"""
Step 1 — Revenue Event Detection (risk scoring) and
Step 2 — Revenue Autopsy (root-cause diagnosis).
"""
from __future__ import annotations

from .models import RevenueEvent, RevenueAutopsy, FailureReason
from .context import CustomerContext, reconstruct_context


def assign_risk_value(event: RevenueEvent) -> float:
    """Revenue-at-risk value for prioritization. For most events this is
    simply the transaction amount, but repeated failures and overdue
    invoices carry additional weight since they are more likely to convert
    to permanent loss the longer they go unresolved."""
    multiplier = 1.0
    if event.retry_count >= 2:
        multiplier += 0.15 * event.retry_count
    if event.invoice_overdue:
        multiplier += 0.10 * min(event.days_since_last_purchase / 30.0, 3.0)
    return round(event.amount * multiplier, 2)


_RECOVERY_ELIGIBILITY_BY_INTENT = {
    "high": "HIGH",
    "medium": "MEDIUM",
    "low": "LOW",
}


def run_autopsy(event: RevenueEvent, ctx: CustomerContext | None = None) -> RevenueAutopsy:
    """Diagnose *why* the revenue is at risk and how eligible it is for recovery."""
    ctx = ctx or reconstruct_context(event)

    eligibility = _RECOVERY_ELIGIBILITY_BY_INTENT[ctx.intent_label]

    # Hard overrides: some root causes are structurally close to unrecoverable
    # regardless of intent score (e.g. an expired card can't just be retried).
    if event.failure_reason == FailureReason.CARD_EXPIRED and event.retry_count == 0:
        eligibility = "MEDIUM" if eligibility == "HIGH" else eligibility
    if event.customer_previous_payments == 0 and event.customer_previous_failures >= 2:
        eligibility = "NONE" if ctx.intent_score < 0.15 else eligibility

    notes = (
        f"{event.customer_previous_payments} previous successful payments, "
        f"{event.customer_previous_failures} previous failures "
        f"({ctx.success_rate:.1%} success rate). "
        f"Root cause classified as {event.failure_reason.value}."
    )

    return RevenueAutopsy(
        root_cause=event.failure_reason,
        customer_intent=ctx.intent_label,
        recovery_eligibility=eligibility,
        previous_success_rate=ctx.success_rate,
        notes=notes,
    )
