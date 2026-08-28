"""
Step 9 — Policy and Safety Gate.

The LLM / decision engine must never have unrestricted control over
money-moving actions. This module is a deterministic, non-AI policy
engine that validates a proposed Decision before it is allowed to reach
the Action Executor. It never asks "is this a good idea" -- that question
was already answered by the counterfactual engine -- it only asks
"is this action permitted".

All rules are simple, explicit, and testable in isolation, which is the
point: the audit trail should never depend on interpreting model output.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import RevenueEvent, Decision, ActionType, PolicyCheck, PolicyVerdict


@dataclass
class PolicyConfig:
    max_retry_count: int = 2                    # block RETRY once retry_count reaches this
    human_approval_amount_threshold: float = 50000.0  # INR, matches "merchant_limit"
    min_confidence: float = 0.35                # below this -> escalate instead of auto-execute
    max_reminders_per_case: int = 3


DEFAULT_POLICY = PolicyConfig()


def check_policy(event: RevenueEvent, decision: Decision, config: PolicyConfig = DEFAULT_POLICY) -> PolicyCheck:
    reasons: list[str] = []
    action = decision.selected_action

    # Payment already succeeded elsewhere in the meantime -> stop all recovery.
    # (In this offline pipeline this is checked by the caller before invoking
    # policy; kept here as a documented rule for completeness / future wiring.)

    if action == ActionType.RETRY_PAYMENT and event.retry_count >= config.max_retry_count:
        reasons.append(f"retry_count ({event.retry_count}) >= max_retry_count ({config.max_retry_count})")
        return _fallback_to_escalation(reasons)

    if event.amount > config.human_approval_amount_threshold:
        reasons.append(
            f"transaction_value (₹{event.amount:,.0f}) > merchant_limit "
            f"(₹{config.human_approval_amount_threshold:,.0f})"
        )
        return PolicyCheck(verdict=PolicyVerdict.REQUIRES_HUMAN_APPROVAL, reasons=reasons)

    if decision.confidence < config.min_confidence:
        reasons.append(f"confidence ({decision.confidence:.2f}) < threshold ({config.min_confidence:.2f})")
        return _fallback_to_escalation(reasons)

    if decision.autopsy.recovery_eligibility == "NONE":
        reasons.append("recovery_eligibility classified as NONE")
        return PolicyCheck(verdict=PolicyVerdict.BLOCKED, reasons=reasons)

    reasons.append("all checks passed")
    return PolicyCheck(verdict=PolicyVerdict.PASSED, reasons=reasons)


def _fallback_to_escalation(reasons: list[str]) -> PolicyCheck:
    reasons.append("falling back to ESCALATE_TO_HUMAN")
    return PolicyCheck(verdict=PolicyVerdict.REQUIRES_HUMAN_APPROVAL, reasons=reasons)


def apply_policy(event: RevenueEvent, decision: Decision, config: PolicyConfig = DEFAULT_POLICY) -> Decision:
    """Runs the policy check and, if the original action was not passed
    outright, rewrites the decision's selected_action to the safe fallback
    (ESCALATE_TO_HUMAN or NO_ACTION for hard blocks) — the executor only
    ever sees policy-approved actions."""
    check = check_policy(event, decision, config)
    decision.policy_check = check

    if check.verdict == PolicyVerdict.BLOCKED:
        decision.selected_action = ActionType.NO_ACTION
    elif check.verdict == PolicyVerdict.REQUIRES_HUMAN_APPROVAL:
        decision.selected_action = ActionType.ESCALATE_TO_HUMAN

    return decision
