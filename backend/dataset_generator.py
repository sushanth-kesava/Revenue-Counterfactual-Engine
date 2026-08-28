"""
Synthetic Revenue Event Dataset Generator.

Produces the >=500 record benchmark dataset described in the project spec,
split across the seven scenario categories, with realistic-ish correlated
fields (e.g. high previous-failure customers are more likely to see repeated
failures; high checkout abandonment correlates with lower recorded intent).

The generator is deterministic given a seed, so the same dataset can be
regenerated for grading / re-runs.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .models import RevenueEvent, EventType, FailureReason

# Category -> record count, matching the spec's suggested breakdown.
SCENARIO_COUNTS: dict[str, int] = {
    "temporary_payment_failure": 100,
    "checkout_abandonment": 100,
    "failed_subscription": 75,
    "overdue_invoice": 75,
    "repeated_payment_failure": 50,
    "recoverable_historical_cases": 50,
    "non_recoverable_cases": 50,
}

PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet", "emi"]


def _rand_amount(rng: random.Random, low: float, high: float) -> float:
    # Log-ish distribution so most transactions are small with a long tail
    # of high-value ones, which is what makes prioritization meaningful.
    x = rng.random() ** 2.2
    return round(low + x * (high - low), 2)


def _make_event(rng: random.Random, category: str, idx: int) -> RevenueEvent:
    merchant_id = f"MID_{rng.randint(1, 40):04d}"
    customer_id = f"CUST_{rng.randint(1, 5000):06d}"
    payment_method = rng.choice(PAYMENT_METHODS)

    if category == "temporary_payment_failure":
        event_type = EventType.PAYMENT_FAILED
        failure_reason = rng.choice(
            [FailureReason.TRANSIENT_BANK_FAILURE, FailureReason.NETWORK_TIMEOUT]
        )
        prev_payments = rng.randint(3, 20)
        prev_failures = rng.randint(0, 1)
        retry_count = 0
        amount = _rand_amount(rng, 300, 25000)
        checkout_abandoned = False
        invoice_overdue = False
        subscription_status = "none"
        days_since_last_purchase = rng.randint(1, 30)

    elif category == "checkout_abandonment":
        event_type = EventType.CHECKOUT_ABANDONED
        failure_reason = FailureReason.CUSTOMER_ABANDONED
        prev_payments = rng.randint(0, 10)
        prev_failures = rng.randint(0, 2)
        retry_count = 0
        amount = _rand_amount(rng, 200, 40000)
        checkout_abandoned = True
        invoice_overdue = False
        subscription_status = "none"
        days_since_last_purchase = rng.randint(0, 90)

    elif category == "failed_subscription":
        event_type = EventType.SUBSCRIPTION_FAILED
        failure_reason = FailureReason.SUBSCRIPTION_LAPSED
        prev_payments = rng.randint(2, 36)
        prev_failures = rng.randint(0, 3)
        retry_count = rng.randint(0, 1)
        amount = _rand_amount(rng, 199, 4999)
        checkout_abandoned = False
        invoice_overdue = False
        subscription_status = "lapsed"
        days_since_last_purchase = rng.randint(1, 45)

    elif category == "overdue_invoice":
        event_type = EventType.INVOICE_OVERDUE
        failure_reason = FailureReason.INVOICE_UNPAID
        prev_payments = rng.randint(1, 15)
        prev_failures = rng.randint(0, 2)
        retry_count = 0
        amount = _rand_amount(rng, 5000, 200000)
        checkout_abandoned = False
        invoice_overdue = True
        subscription_status = "none"
        days_since_last_purchase = rng.randint(15, 120)

    elif category == "repeated_payment_failure":
        event_type = EventType.REPEATED_FAILURE
        failure_reason = rng.choice(
            [FailureReason.ISSUER_DECLINE, FailureReason.INSUFFICIENT_FUNDS, FailureReason.CARD_EXPIRED]
        )
        prev_payments = rng.randint(0, 8)
        prev_failures = rng.randint(3, 6)
        retry_count = rng.randint(2, 4)
        amount = _rand_amount(rng, 300, 30000)
        checkout_abandoned = False
        invoice_overdue = False
        subscription_status = "none"
        days_since_last_purchase = rng.randint(1, 20)

    elif category == "recoverable_historical_cases":
        # Strong history, single blip -> should be an "easy win" for the agent.
        event_type = rng.choice([EventType.PAYMENT_FAILED, EventType.SUBSCRIPTION_FAILED])
        failure_reason = FailureReason.TRANSIENT_BANK_FAILURE
        prev_payments = rng.randint(8, 40)
        prev_failures = 1
        retry_count = 0
        amount = _rand_amount(rng, 500, 50000)
        checkout_abandoned = False
        invoice_overdue = False
        subscription_status = "active" if event_type == EventType.SUBSCRIPTION_FAILED else "none"
        days_since_last_purchase = rng.randint(1, 10)

    else:  # non_recoverable_cases
        event_type = rng.choice(
            [EventType.PAYMENT_FAILED, EventType.CHECKOUT_ABANDONED, EventType.REPEATED_FAILURE]
        )
        failure_reason = rng.choice(
            [FailureReason.CARD_EXPIRED, FailureReason.ISSUER_DECLINE, FailureReason.UNKNOWN]
        )
        prev_payments = 0
        prev_failures = rng.randint(2, 5)
        retry_count = rng.randint(0, 3)
        amount = _rand_amount(rng, 200, 15000)
        checkout_abandoned = event_type == EventType.CHECKOUT_ABANDONED
        invoice_overdue = False
        subscription_status = "none"
        days_since_last_purchase = rng.randint(90, 400)

    total_hist = prev_payments + prev_failures
    success_rate = prev_payments / total_hist if total_hist > 0 else 0.0
    customer_value = round(prev_payments * _rand_amount(rng, 200, 8000) * (0.5 + success_rate), 2)

    return RevenueEvent(
        transaction_id=f"ORD_{10000 + idx}",
        merchant_id=merchant_id,
        customer_id=customer_id,
        amount=amount,
        event_type=event_type,
        failure_reason=failure_reason,
        payment_method=payment_method,
        customer_previous_payments=prev_payments,
        customer_previous_failures=prev_failures,
        days_since_last_purchase=days_since_last_purchase,
        subscription_status=subscription_status,
        checkout_abandoned=checkout_abandoned,
        invoice_overdue=invoice_overdue,
        retry_count=retry_count,
        customer_value=customer_value,
    )


def generate_dataset(seed: int = 42, counts: dict[str, int] | None = None) -> list[RevenueEvent]:
    """Generate the full synthetic dataset. Deterministic for a given seed."""
    rng = random.Random(seed)
    counts = counts or SCENARIO_COUNTS
    events: list[RevenueEvent] = []
    idx = 0
    for category, n in counts.items():
        for _ in range(n):
            events.append(_make_event(rng, category, idx))
            idx += 1
    rng.shuffle(events)
    return events


def split_dataset(
    events: list[RevenueEvent], train: float = 0.6, val: float = 0.2
) -> tuple[list[RevenueEvent], list[RevenueEvent], list[RevenueEvent]]:
    """Split into train/calibration, validation, and held-out test sets."""
    n = len(events)
    n_train = int(n * train)
    n_val = int(n * val)
    return events[:n_train], events[n_train:n_train + n_val], events[n_train + n_val:]


if __name__ == "__main__":
    import json
    import os

    ds = generate_dataset()
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "events.json")
    with open(out_path, "w") as f:
        json.dump([e.to_dict() for e in ds], f, indent=2)
    print(f"Generated {len(ds)} events -> {out_path}")
