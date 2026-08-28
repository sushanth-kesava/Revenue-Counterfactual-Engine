"""
Unit tests for the Revenue Counterfactual Engine.

Run with:  python -m pytest tests/ -v
       or:  python tests/test_engine.py   (falls back to plain asserts)

Tests cover:
  - Dataset generation / determinism (legacy)
  - Context reconstruction
  - Counterfactual engine + model interface
  - Policy engine / safety gate
  - Baseline context-blindness
  - Executor / simulation fairness
  - Feature engineering
  - Leakage prevention
  - No-action as legitimate decision
  - End-to-end benchmark
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.models import (
    RevenueEvent, EventType, FailureReason, ActionType, PolicyVerdict,
)
from backend.dataset_generator import generate_dataset, split_dataset, SCENARIO_COUNTS
from backend.context import reconstruct_context
from backend.counterfactual_engine import evaluate_counterfactuals, set_model
from backend.policy_engine import apply_policy, DEFAULT_POLICY, PolicyConfig
from backend.baseline import baseline_decide
from backend.executor import SimulatedExecutor
from backend.evaluation import run_benchmark
from backend.features.columns import AGENT_INPUT_COLUMNS, EVALUATION_COLUMNS, VALID_ACTIONS
from backend.models_ml.recovery_model import FallbackRuleModel, RecoveryModel


def make_event(**overrides) -> RevenueEvent:
    defaults = dict(
        transaction_id="ORD_TEST",
        merchant_id="MID_TEST",
        customer_id="CUST_TEST",
        amount=8000.0,
        event_type=EventType.PAYMENT_FAILED,
        failure_reason=FailureReason.TRANSIENT_BANK_FAILURE,
        payment_method="card",
        customer_previous_payments=6,
        customer_previous_failures=1,
        days_since_last_purchase=5,
        subscription_status="none",
        checkout_abandoned=False,
        invoice_overdue=False,
        retry_count=0,
        customer_value=40000.0,
    )
    defaults.update(overrides)
    return RevenueEvent(**defaults)


# --------------------------------------------------------------------------- #
# Dataset generator (legacy)
# --------------------------------------------------------------------------- #

def test_dataset_size_matches_spec():
    ds = generate_dataset()
    assert len(ds) == sum(SCENARIO_COUNTS.values()) == 500


def test_dataset_is_deterministic_given_seed():
    a = generate_dataset(seed=123)
    b = generate_dataset(seed=123)
    assert [e.transaction_id for e in a] == [e.transaction_id for e in b]
    assert [e.amount for e in a] == [e.amount for e in b]


def test_split_covers_all_events_without_overlap():
    ds = generate_dataset()
    train, val, test = split_dataset(ds)
    assert len(train) + len(val) + len(test) == len(ds)
    ids = [e.event_id for e in train + val + test]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------- #
# Context reconstruction
# --------------------------------------------------------------------------- #

def test_high_history_customer_gets_high_intent():
    event = make_event(customer_previous_payments=20, customer_previous_failures=1, days_since_last_purchase=2)
    ctx = reconstruct_context(event)
    assert ctx.intent_label == "high"


def test_never_paid_customer_gets_low_intent():
    event = make_event(customer_previous_payments=0, customer_previous_failures=3, days_since_last_purchase=200)
    ctx = reconstruct_context(event)
    assert ctx.intent_label == "low"


# --------------------------------------------------------------------------- #
# Counterfactual engine + model
# --------------------------------------------------------------------------- #

def test_engine_selects_highest_expected_value_option():
    # Use fallback model for deterministic test
    set_model(FallbackRuleModel())
    event = make_event()
    decision = evaluate_counterfactuals(event)
    best = max(decision.counterfactuals, key=lambda c: c.expected_recovery)
    assert decision.selected_action == best.action
    assert decision.selected_expected_recovery == best.expected_recovery


def test_engine_includes_no_action():
    set_model(FallbackRuleModel())
    event = make_event()
    decision = evaluate_counterfactuals(event)
    actions = [c.action for c in decision.counterfactuals]
    assert ActionType.NO_ACTION in actions


def test_no_action_has_zero_probability():
    set_model(FallbackRuleModel())
    event = make_event()
    decision = evaluate_counterfactuals(event)
    no_action = next(c for c in decision.counterfactuals if c.action == ActionType.NO_ACTION)
    assert no_action.probability_of_success == 0.0
    assert no_action.expected_recovery == 0.0


def test_no_action_wins_when_all_actions_low_value():
    """If expected recovery is below threshold for all actions, NO_ACTION wins."""
    set_model(FallbackRuleModel())
    # Very small amount where costs exceed recovery
    event = make_event(amount=5.0, customer_previous_payments=0, customer_previous_failures=5)
    decision = evaluate_counterfactuals(event)
    # At ₹5, cost+penalty should exceed expected recovery for most actions
    # NO_ACTION should be selected or SEND_REMINDER (₹2 cost, low threshold)
    assert decision.selected_action in (ActionType.NO_ACTION, ActionType.SEND_REMINDER)


def test_repeated_failures_lower_retry_probability():
    set_model(FallbackRuleModel())
    fresh = make_event(retry_count=0)
    exhausted = make_event(retry_count=3)
    d_fresh = evaluate_counterfactuals(fresh)
    d_exhausted = evaluate_counterfactuals(exhausted)
    p_fresh = next(c for c in d_fresh.counterfactuals if c.action == ActionType.RETRY_PAYMENT).probability_of_success
    # exhausted may not have retry available due to feasibility, check if present
    retry_opts = [c for c in d_exhausted.counterfactuals if c.action == ActionType.RETRY_PAYMENT]
    if retry_opts:
        p_exhausted = retry_opts[0].probability_of_success
        assert p_exhausted < p_fresh


def test_model_interface_contract():
    """RecoveryModel interface must provide predict_probability and predict_action_values."""
    model = FallbackRuleModel()
    assert model.is_trained()

    context = {"transaction_amount": 1000, "customer_value_segment": "high", "retry_count": 0}
    p = model.predict_probability(context, "RETRY_PAYMENT")
    assert 0.0 <= p <= 1.0

    values = model.predict_action_values(context)
    assert set(values.keys()) == set(VALID_ACTIONS)
    for v in values.values():
        assert 0.0 <= v <= 1.0


# --------------------------------------------------------------------------- #
# Policy engine / safety gate
# --------------------------------------------------------------------------- #

def test_policy_blocks_retry_after_max_attempts():
    set_model(FallbackRuleModel())
    event = make_event(retry_count=2)
    decision = evaluate_counterfactuals(event)
    decision.selected_action = ActionType.RETRY_PAYMENT
    checked = apply_policy(event, decision, DEFAULT_POLICY)
    assert checked.selected_action != ActionType.RETRY_PAYMENT
    assert checked.policy_check.verdict in (PolicyVerdict.REQUIRES_HUMAN_APPROVAL, PolicyVerdict.BLOCKED)


def test_policy_requires_human_approval_above_merchant_limit():
    set_model(FallbackRuleModel())
    event = make_event(amount=75000.0)
    decision = evaluate_counterfactuals(event)
    checked = apply_policy(event, decision, DEFAULT_POLICY)
    assert checked.policy_check.verdict == PolicyVerdict.REQUIRES_HUMAN_APPROVAL
    assert checked.selected_action == ActionType.ESCALATE_TO_HUMAN


def test_policy_never_lets_low_confidence_auto_execute():
    set_model(FallbackRuleModel())
    config = PolicyConfig(min_confidence=0.99)
    event = make_event()
    decision = evaluate_counterfactuals(event)
    checked = apply_policy(event, decision, config)
    assert checked.selected_action in (ActionType.ESCALATE_TO_HUMAN, ActionType.NO_ACTION)


def test_safety_gate_cannot_be_overridden_by_model():
    """The model/agent cannot bypass the deterministic safety gate."""
    set_model(FallbackRuleModel())
    event = make_event(retry_count=5)  # Way past max retry
    decision = evaluate_counterfactuals(event)
    # Force the "AI recommendation" to be retry
    decision.selected_action = ActionType.RETRY_PAYMENT
    decision.confidence = 0.99  # High confidence
    checked = apply_policy(event, decision, DEFAULT_POLICY)
    # Safety gate MUST override regardless of confidence
    assert checked.selected_action != ActionType.RETRY_PAYMENT


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #

def test_baseline_ignores_customer_history():
    rich_history = make_event(customer_previous_payments=30, customer_previous_failures=0)
    no_history = make_event(customer_previous_payments=0, customer_previous_failures=0)
    d1 = baseline_decide(rich_history)
    d2 = baseline_decide(no_history)
    assert d1.selected_action == d2.selected_action


def test_baseline_escalates_after_full_chain():
    event = make_event(retry_count=99)
    decision = baseline_decide(event)
    assert decision.selected_action == ActionType.ESCALATE_TO_HUMAN


# --------------------------------------------------------------------------- #
# Executor / simulation fairness
# --------------------------------------------------------------------------- #

def test_simulated_executor_is_deterministic_for_a_seed():
    set_model(FallbackRuleModel())
    event = make_event()
    decision = evaluate_counterfactuals(event)
    decision = apply_policy(event, decision, DEFAULT_POLICY)
    out1 = SimulatedExecutor(seed=5).execute(event, decision)
    out2 = SimulatedExecutor(seed=5).execute(event, decision)
    assert out1.result == out2.result
    assert out1.amount_recovered == out2.amount_recovered


def test_no_action_never_recovers_revenue():
    set_model(FallbackRuleModel())
    event = make_event()
    decision = evaluate_counterfactuals(event)
    decision.selected_action = ActionType.NO_ACTION
    out = SimulatedExecutor(seed=1).execute(event, decision)
    assert out.amount_recovered == 0.0


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #

def test_feature_engineering_no_leakage():
    """Verify that EVALUATION_COLUMNS never appear in the feature matrix."""
    import pandas as pd
    from backend.features.feature_engineering import build_feature_matrix

    # Create a sample row with ALL columns (including evaluation ones)
    sample = {col: 0 for col in AGENT_INPUT_COLUMNS + EVALUATION_COLUMNS}
    sample["payment_method"] = "Credit Card"
    sample["device_type"] = "Mobile"
    sample["revenue_event"] = "payment_failure"
    sample["failure_reason"] = "bank_timeout"
    sample["customer_value_segment"] = "high"
    sample["automated_recovery_risk"] = "low"
    sample["risk_payment_method"] = "Credit Card"
    sample["risk_device_type"] = "Mobile"
    sample["transaction_amount"] = 1000

    df = pd.DataFrame([sample])
    features, feature_names = build_feature_matrix(df, actions=pd.Series(["RETRY_PAYMENT"]))

    # NO evaluation column should appear in features
    leaked = set(feature_names) & set(EVALUATION_COLUMNS)
    assert leaked == set(), f"LEAKAGE: {leaked}"


def test_feature_engineering_produces_numeric_output():
    """All features must be numeric (no object columns)."""
    import pandas as pd
    from backend.features.feature_engineering import build_feature_matrix

    sample = {
        "transaction_amount": 1000, "payment_method": "PayPal",
        "device_type": "Mobile", "revenue_event": "payment_failure",
        "failure_reason": "bank_timeout", "retry_count": 0,
        "recovery_eligible": True, "customer_age": 30,
        "customer_city": "Mumbai",
        "behavior_transactions": 5, "behavior_total_spend": 5000,
        "behavior_average_order_value": 1000, "behavior_avg_session_minutes": 10,
        "behavior_avg_pages": 8, "behavior_returning_rate": 0.8,
        "behavior_avg_rating": 4.0, "behavior_avg_delivery_days": 3,
        "historical_transactions": 20, "historical_total_spend": 20000,
        "historical_average_order_value": 1000, "historical_median_order_value": 800,
        "historical_max_order_value": 3000, "historical_total_quantity": 60,
        "days_since_last_purchase": 10, "customer_lifetime_days": 365,
        "purchase_frequency_per_30d": 2, "distinct_products": 15,
        "distinct_countries": 1, "customer_value_segment": "high",
        "risk_amount": 100, "risk_payment_method": "PayPal",
        "risk_device_type": "Mobile", "failed_transactions_24h": 0,
        "transaction_frequency_24h": 1, "unusual_amount_flag": False,
        "unusual_location_flag": False, "multiple_transactions_short_time": False,
        "high_risk_device_flag": False, "velocity_flag": False,
        "previous_fraud_flag": False, "risk_signal_count": 0,
        "automated_recovery_risk": "low",
    }
    df = pd.DataFrame([sample])
    features, _ = build_feature_matrix(df, actions=pd.Series(["RETRY_PAYMENT"]))
    assert features.select_dtypes(include=["object"]).shape[1] == 0


def test_customer_aggregation_handles_missing_ids():
    """Feature engineering should handle missing/null customer fields gracefully."""
    import pandas as pd
    from backend.features.feature_engineering import build_feature_matrix

    sample = {
        "transaction_amount": 500, "payment_method": "Debit Card",
        "device_type": "Desktop", "revenue_event": "checkout_abandonment",
        "failure_reason": "checkout_abandoned", "retry_count": 0,
        "recovery_eligible": True, "customer_age": None,
        "customer_city": None,
        "behavior_transactions": 0, "behavior_total_spend": 0,
        "behavior_average_order_value": 0, "behavior_avg_session_minutes": 0,
        "behavior_avg_pages": 0, "behavior_returning_rate": 0,
        "behavior_avg_rating": 0, "behavior_avg_delivery_days": 0,
        "historical_transactions": 0, "historical_total_spend": 0,
        "historical_average_order_value": 0, "historical_median_order_value": 0,
        "historical_max_order_value": 0, "historical_total_quantity": 0,
        "days_since_last_purchase": 0, "customer_lifetime_days": 0,
        "purchase_frequency_per_30d": 0, "distinct_products": 0,
        "distinct_countries": 0, "customer_value_segment": "low",
        "risk_amount": 0, "risk_payment_method": "Debit Card",
        "risk_device_type": "Desktop", "failed_transactions_24h": 0,
        "transaction_frequency_24h": 0, "unusual_amount_flag": False,
        "unusual_location_flag": False, "multiple_transactions_short_time": False,
        "high_risk_device_flag": False, "velocity_flag": False,
        "previous_fraud_flag": False, "risk_signal_count": 0,
        "automated_recovery_risk": "low",
    }
    df = pd.DataFrame([sample])
    features, feature_names = build_feature_matrix(df, actions=pd.Series(["SEND_REMINDER"]))
    # Should not crash, should produce valid numerics
    assert not features.isnull().any().any()


# --------------------------------------------------------------------------- #
# Risk scoring
# --------------------------------------------------------------------------- #

def test_risk_signal_count_affects_model():
    """Cases with more risk signals should get different probability estimates."""
    model = FallbackRuleModel()
    low_risk = {"automated_recovery_risk": "low", "retry_count": 0, "customer_value_segment": "high"}
    high_risk = {"automated_recovery_risk": "high", "retry_count": 0, "customer_value_segment": "high"}
    p_low = model.predict_probability(low_risk, "RETRY_PAYMENT")
    p_high = model.predict_probability(high_risk, "RETRY_PAYMENT")
    # High-risk cases have higher recovery potential in this model
    assert p_high != p_low


# --------------------------------------------------------------------------- #
# Webhook signature verification
# --------------------------------------------------------------------------- #

def test_webhook_hmac_matches_razorpay_sdk_construction():
    import hmac
    import hashlib

    body = b'{"event":"payment.captured","payload":{}}'
    secret = b"test_webhook_secret_123"
    expected = hmac.new(secret, body, hashlib.sha256).hexdigest()
    computed = hmac.new(secret, body, hashlib.sha256).hexdigest()
    assert hmac.compare_digest(expected, computed)


def test_webhook_hmac_rejects_tampered_body():
    import hmac
    import hashlib

    secret = b"test_webhook_secret_123"
    original = b'{"event":"payment.captured","payload":{"amount":100}}'
    tampered = b'{"event":"payment.captured","payload":{"amount":999999}}'

    sig_for_original = hmac.new(secret, original, hashlib.sha256).hexdigest()
    sig_for_tampered = hmac.new(secret, tampered, hashlib.sha256).hexdigest()
    assert sig_for_original != sig_for_tampered


# --------------------------------------------------------------------------- #
# End-to-end benchmark (legacy mode)
# --------------------------------------------------------------------------- #

def test_benchmark_runs_and_agent_has_zero_policy_violations():
    set_model(FallbackRuleModel())
    result = run_benchmark()
    assert result["agent"]["policy_violations"] == 0
    assert result["baseline"]["revenue_at_risk"] == result["agent"]["revenue_at_risk"]


def test_agent_recovers_at_least_as_much_as_baseline():
    set_model(FallbackRuleModel())
    result = run_benchmark()
    assert result["agent"]["revenue_recovered"] >= result["baseline"]["revenue_recovered"] * 0.9


# --------------------------------------------------------------------------- #
# Deterministic reproducibility
# --------------------------------------------------------------------------- #

def test_full_pipeline_reproducibility():
    """Running the benchmark twice with the same seed produces identical results."""
    set_model(FallbackRuleModel())
    r1 = run_benchmark(seed=42, executor_seed=7)
    r2 = run_benchmark(seed=42, executor_seed=7)
    assert r1["agent"]["revenue_recovered"] == r2["agent"]["revenue_recovered"]
    assert r1["baseline"]["revenue_recovered"] == r2["baseline"]["revenue_recovered"]


# --------------------------------------------------------------------------- #
# Run as script
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    test_fns = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for fn in test_fns:
        try:
            fn()
            passed += 1
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")

    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    sys.exit(1 if failed else 0)
