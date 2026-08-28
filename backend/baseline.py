"""
Baseline system — the conventional fixed-rule workflow described in the
spec (Section 15):

    Payment failed -> Retry -> Reminder -> Escalate

This system does NOT reconstruct customer context, does NOT evaluate
counterfactuals, and does NOT compute expected value. It applies the same
linear workflow to every event of a given type, regardless of customer
history or transaction value. It exists purely so the counterfactual
agent has something honest to be benchmarked against on the same
held-out dataset (evaluation.py).
"""
from __future__ import annotations

from .models import (
    RevenueEvent,
    Decision,
    RevenueAutopsy,
    CounterfactualOption,
    ActionType,
    PolicyCheck,
    PolicyVerdict,
    EventType,
    FailureReason,
)

# The fixed, non-adaptive rule table. Every event type maps to exactly one
# next action based only on retry_count -- no context, no scoring.
_FIXED_RULES = {
    EventType.PAYMENT_FAILED: [ActionType.RETRY_PAYMENT, ActionType.RETRY_PAYMENT, ActionType.SEND_REMINDER, ActionType.ESCALATE_TO_HUMAN],
    EventType.REPEATED_FAILURE: [ActionType.SEND_REMINDER, ActionType.ESCALATE_TO_HUMAN],
    EventType.CHECKOUT_ABANDONED: [ActionType.SEND_REMINDER, ActionType.SEND_REMINDER, ActionType.ESCALATE_TO_HUMAN],
    EventType.SUBSCRIPTION_FAILED: [ActionType.RETRY_PAYMENT, ActionType.SEND_REMINDER, ActionType.ESCALATE_TO_HUMAN],
    EventType.INVOICE_OVERDUE: [ActionType.SEND_REMINDER, ActionType.SEND_REMINDER, ActionType.ESCALATE_TO_HUMAN],
}

# Flat, non-adaptive probability priors used only to compute a comparable
# "expected recovery" figure for the ledger / dashboard -- the baseline
# itself never looks at these numbers when deciding what to do.
_FLAT_PROBABILITY = {
    ActionType.RETRY_PAYMENT: 0.45,
    ActionType.SEND_REMINDER: 0.25,
    ActionType.ESCALATE_TO_HUMAN: 0.50,
    ActionType.CREATE_PAYMENT_LINK: 0.40,
    ActionType.NO_ACTION: 0.0,
}


def baseline_decide(event: RevenueEvent) -> Decision:
    """Applies the fixed workflow: pick the step in the rule chain indexed
    by retry_count, with no counterfactual comparison."""
    chain = _FIXED_RULES.get(event.event_type, [ActionType.SEND_REMINDER, ActionType.ESCALATE_TO_HUMAN])
    step = min(event.retry_count, len(chain) - 1)
    action = chain[step]

    p = _FLAT_PROBABILITY[action]
    expected_recovery = round(max(0.0, event.amount * p), 2)

    option = CounterfactualOption(
        action=action,
        probability_of_success=p,
        intervention_cost=0.0,
        risk_penalty=0.0,
        customer_friction=0.0,
        expected_recovery=expected_recovery,
    )

    autopsy = RevenueAutopsy(
        root_cause=event.failure_reason,
        customer_intent="not_evaluated",
        recovery_eligibility="not_evaluated",
        previous_success_rate=0.0,
        notes="Fixed-rule baseline does not perform context reconstruction.",
    )

    return Decision(
        event_id=event.event_id,
        transaction_id=event.transaction_id,
        autopsy=autopsy,
        counterfactuals=[option],
        selected_action=action,
        selected_expected_recovery=expected_recovery,
        best_possible_expected_recovery=expected_recovery,  # baseline has no alternatives to compare
        confidence=1.0,  # baseline is not confidence-aware; always "executes"
        policy_check=PolicyCheck(verdict=PolicyVerdict.PASSED, reasons=["fixed_rule_no_policy_gate"]),
    )
