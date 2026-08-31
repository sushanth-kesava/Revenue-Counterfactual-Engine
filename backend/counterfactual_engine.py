"""
Steps 7-8 — Counterfactual Generation and Scoring (DATA-DRIVEN).

This is the intellectual core of the project. For every revenue event the
engine does NOT jump straight to an action. It:

  1. Generates the set of feasible interventions for this event type.
  2. Calls the RecoveryModel to estimate P(recovery | action, context)
     for each feasible action — driven by ML trained on real data, not
     hand-coded priors.
  3. Scores each option:
         expected_recovery = amount * P(success) - cost - risk_penalty
  4. Selects the option with the highest expected utility, respecting
     that NO_ACTION is a legitimate choice.

The output is a fully-populated Decision (see models.py) *before* any
policy check or execution happens — decision-making and authorization are
deliberately kept as separate stages.

ARCHITECTURE:
    Context → RecoveryModel → P(recovery | action, context) per action
    → Expected Monetary Value → Best Action (including NO_ACTION)
"""
from __future__ import annotations

from .models import (
    RevenueEvent,
    RevenueAutopsy,
    ActionType,
    CounterfactualOption,
    Decision,
    FailureReason,
    EventType,
    PolicyCheck,
    PolicyVerdict,
)
from .context import CustomerContext, reconstruct_context
from .risk_engine import run_autopsy
from .models_ml.config import DEFAULT_CONFIG, DecisionConfig
from .models_ml.recovery_model import RecoveryModel, get_model
from .features.columns import VALID_ACTIONS

# Module-level model instance (loaded once, reused)
_MODEL: RecoveryModel | None = None


def _get_model() -> RecoveryModel:
    """Lazy-load the recovery model."""
    global _MODEL
    if _MODEL is None:
        _MODEL = get_model()
    return _MODEL


def set_model(model: RecoveryModel) -> None:
    """Override the model (useful for testing or live model swaps)."""
    global _MODEL
    _MODEL = model


def _feasible_actions(event: RevenueEvent) -> list[ActionType]:
    """Which actions are structurally possible for this event type."""
    actions = [ActionType.SEND_REMINDER, ActionType.ESCALATE_TO_HUMAN, ActionType.NO_ACTION]

    can_retry = (
        event.event_type in (EventType.PAYMENT_FAILED, EventType.SUBSCRIPTION_FAILED, EventType.REPEATED_FAILURE)
        and event.failure_reason
        in (FailureReason.TRANSIENT_BANK_FAILURE, FailureReason.NETWORK_TIMEOUT, FailureReason.ISSUER_DECLINE)
    )
    if can_retry:
        actions.append(ActionType.RETRY_PAYMENT)

    can_link = event.event_type in (
        EventType.CHECKOUT_ABANDONED,
        EventType.INVOICE_OVERDUE,
        EventType.SUBSCRIPTION_FAILED,
        EventType.PAYMENT_FAILED,
    ) or event.failure_reason == FailureReason.CARD_EXPIRED
    if can_link:
        actions.append(ActionType.CREATE_PAYMENT_LINK)

    return actions


def _derive_customer_value_segment(event: RevenueEvent) -> str:
    """Derive customer value segment from numeric customer_value."""
    # Use the attribute if it exists as a string (from benchmark CSV)
    seg = getattr(event, "customer_value_segment", None)
    if seg and isinstance(seg, str):
        return seg
    # Otherwise derive from numeric customer_value
    cv = getattr(event, "customer_value", 0) or 0
    if cv >= 50000: return "very_high"
    if cv >= 15000: return "high"
    if cv >= 5000: return "medium"
    return "low"


def _event_to_context_dict(event: RevenueEvent, ctx: CustomerContext) -> dict:
    """Convert event + reconstructed context into a flat dict for the model."""
    return {
        "transaction_amount": event.amount,
        "payment_method": event.payment_method,
        "device_type": getattr(event, "device_type", "Desktop"),
        "revenue_event": event.event_type.value,
        "failure_reason": event.failure_reason.value,
        "retry_count": event.retry_count,
        "recovery_eligible": not getattr(event, "high_risk_case", False),
        "customer_value_segment": _derive_customer_value_segment(event),
        "customer_age": getattr(event, "customer_age", 35.0),
        "customer_city": getattr(event, "customer_city", "Unknown"),
        "behavior_transactions": getattr(event, "behavior_transactions", event.customer_previous_payments),
        "behavior_total_spend": getattr(event, "behavior_total_spend", event.customer_value),
        "behavior_average_order_value": getattr(event, "behavior_average_order_value", event.amount),
        "behavior_avg_session_minutes": getattr(event, "behavior_avg_session_minutes", 5.0),
        "behavior_avg_pages": getattr(event, "behavior_avg_pages", 5.0),
        "behavior_returning_rate": getattr(event, "behavior_returning_rate", 1.0 if ctx.is_repeat_customer else 0.0),
        "behavior_avg_rating": getattr(event, "behavior_avg_rating", 3.5),
        "behavior_avg_delivery_days": getattr(event, "behavior_avg_delivery_days", 4.0),
        "historical_transactions": getattr(event, "historical_transactions", event.customer_previous_payments + event.customer_previous_failures),
        "historical_total_spend": getattr(event, "historical_total_spend", event.customer_value),
        "historical_average_order_value": getattr(event, "historical_average_order_value", event.amount),
        "historical_median_order_value": getattr(event, "historical_median_order_value", event.amount * 0.8),
        "historical_max_order_value": getattr(event, "historical_max_order_value", event.amount * 1.5), 
        "historical_total_quantity": getattr(event, "historical_total_quantity", event.customer_previous_payments * 3),
        "days_since_last_purchase": event.days_since_last_purchase,
        "customer_lifetime_days": getattr(event, "customer_lifetime_days", 90),
        "purchase_frequency_per_30d": getattr(event, "purchase_frequency_per_30d", 2.0),
        "distinct_products": getattr(event, "distinct_products", 5),
        "distinct_countries": getattr(event, "distinct_countries", 1),
        "risk_amount": getattr(event, "risk_amount", event.amount * 0.1),
        "risk_payment_method": getattr(event, "risk_payment_method", event.payment_method),
        "risk_device_type": getattr(event, "risk_device_type", "Desktop"),
        "failed_transactions_24h": getattr(event, "failed_transactions_24h", event.customer_previous_failures),
        "transaction_frequency_24h": getattr(event, "transaction_frequency_24h", 1),
        "unusual_amount_flag": getattr(event, "unusual_amount_flag", False),
        "unusual_location_flag": getattr(event, "unusual_location_flag", False),
        "multiple_transactions_short_time": getattr(event, "multiple_transactions_short_time", False),
        "high_risk_device_flag": getattr(event, "high_risk_device_flag", False),
        "velocity_flag": getattr(event, "velocity_flag", False),
        "previous_fraud_flag": getattr(event, "previous_fraud_flag", False),
        "risk_signal_count": getattr(event, "risk_signal_count", 0),
    }


def evaluate_counterfactuals(
    event: RevenueEvent,
    config: DecisionConfig | None = None,
) -> Decision:
    """Full Steps 3, 7, 8: reconstruct context, call the recovery model
    to score every feasible intervention, and select the best expected-value
    action (including NO_ACTION as a legitimate option)."""
    config = config or DEFAULT_CONFIG
    model = _get_model()
    ctx = reconstruct_context(event)
    autopsy = run_autopsy(event, ctx)
    context_dict = _event_to_context_dict(event, ctx)

    # Derive risk tier from risk_signal_count (not the leaky automated_recovery_risk)
    rsc = context_dict.get("risk_signal_count", 0)
    if rsc >= 3: risk_tier = "high"
    elif rsc >= 1: risk_tier = "medium"
    else: risk_tier = "low"

    options: list[CounterfactualOption] = []
    for action in _feasible_actions(event):
        action_str = action.value

        # Call the ML model for probability estimation
        p = model.predict_probability(context_dict, action_str)

        # Costs from configuration
        cost = config.intervention_costs.get(action_str)
        risk_multiplier = config.risk_penalties.get_multiplier(risk_tier)
        risk_penalty = round(risk_multiplier * event.amount, 2)

        # Expected recovery = amount * P(success) - cost - risk_penalty
        expected_recovery = round(event.amount * p - cost - risk_penalty, 2)
        expected_recovery = max(0.0, expected_recovery)

        # Customer friction (informational, not used in scoring)
        friction_map = {
            ActionType.RETRY_PAYMENT: 0.10,
            ActionType.CREATE_PAYMENT_LINK: 0.30,
            ActionType.SEND_REMINDER: 0.20,
            ActionType.ESCALATE_TO_HUMAN: 0.15,
            ActionType.NO_ACTION: 0.0,
        }

        options.append(
            CounterfactualOption(
                action=action,
                probability_of_success=p,
                intervention_cost=cost,
                risk_penalty=risk_penalty,
                customer_friction=friction_map.get(action, 0.0),
                expected_recovery=expected_recovery,
            )
        )

    # Select best action — NO_ACTION is a real option
    best = max(options, key=lambda o: o.expected_recovery)

    # Check minimum intervention threshold: if best action's incremental
    # value over NO_ACTION is below threshold, prefer NO_ACTION
    no_action_opt = next((o for o in options if o.action == ActionType.NO_ACTION), None)
    if (
        no_action_opt is not None
        and best.action != ActionType.NO_ACTION
        and best.expected_recovery - (no_action_opt.expected_recovery) < config.min_intervention_threshold
    ):
        best = no_action_opt

    best_possible = max(
        (o.expected_recovery for o in options if o.action != ActionType.NO_ACTION),
        default=best.expected_recovery,
    )

    # Confidence: based on model probability spread and context richness
    history_depth = getattr(event, "historical_transactions", 0) or (
        event.customer_previous_payments + event.customer_previous_failures
    )
    prob_spread = max(o.probability_of_success for o in options) - min(o.probability_of_success for o in options)
    confidence = round(
        min(0.97, 0.40 + 0.30 * min(1.0, history_depth / 20.0) + 0.20 * prob_spread + 0.10 * (ctx.intent_score)),
        3,
    )

    return Decision(
        event_id=event.event_id,
        transaction_id=event.transaction_id,
        autopsy=autopsy,
        counterfactuals=sorted(options, key=lambda o: -o.expected_recovery),
        selected_action=best.action,
        selected_expected_recovery=best.expected_recovery,
        best_possible_expected_recovery=best_possible,
        confidence=confidence,
        policy_check=PolicyCheck(verdict=PolicyVerdict.PASSED, reasons=[]),  # filled by policy_engine
    )
