"""
Step 3 — Customer Intent Reconstruction.

Turns raw historical signals on a RevenueEvent into a normalized intent
score and label. This is intentionally a transparent, deterministic
scoring function (not an LLM call) so it can be audited and unit tested;
the AI reasoning layer in counterfactual_engine.py consumes this output
rather than re-deriving it.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import RevenueEvent


@dataclass
class CustomerContext:
    success_rate: float
    intent_score: float  # 0-1
    intent_label: str  # "high" | "medium" | "low"
    is_high_value: bool
    is_repeat_customer: bool
    recency_penalty: float  # 0-1, higher = more stale


def reconstruct_context(event: RevenueEvent) -> CustomerContext:
    total = event.customer_previous_payments + event.customer_previous_failures
    success_rate = event.customer_previous_payments / total if total > 0 else 0.0

    is_repeat_customer = event.customer_previous_payments >= 2
    is_high_value = event.customer_value >= 15000 or event.amount >= 20000

    # Recency: intent decays the longer since the customer last transacted.
    recency_penalty = min(1.0, event.days_since_last_purchase / 180.0)

    # Weighted intent score. Weights are explicit and tunable, not hidden
    # inside a black-box model, so the "why" is always inspectable.
    score = (
        0.45 * success_rate
        + 0.20 * min(1.0, event.customer_previous_payments / 10.0)
        - 0.20 * min(1.0, event.customer_previous_failures / 5.0)
        - 0.15 * recency_penalty
    )
    # Repeated-failure and stale/never-paying customers get an extra penalty.
    if event.retry_count >= 2:
        score -= 0.10
    if total == 0:
        score -= 0.15

    score = max(0.0, min(1.0, score))

    if score >= 0.55:
        label = "high"
    elif score >= 0.3:
        label = "medium"
    else:
        label = "low"

    return CustomerContext(
        success_rate=round(success_rate, 3),
        intent_score=round(score, 3),
        intent_label=label,
        is_high_value=is_high_value,
        is_repeat_customer=is_repeat_customer,
        recency_penalty=round(recency_penalty, 3),
    )
