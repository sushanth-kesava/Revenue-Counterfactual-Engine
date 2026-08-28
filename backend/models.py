"""
Core data models for the Revenue Counterfactual Engine.

These are deliberately plain dataclasses (not ORM models) so the engine
logic in this package can be unit tested and run completely offline,
independent of whatever persistence layer (Postgres, SQLite, in-memory)
wraps it in a given deployment.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #

class EventType(str, Enum):
    PAYMENT_FAILED = "payment_failed"
    CHECKOUT_ABANDONED = "checkout_abandoned"
    SUBSCRIPTION_FAILED = "subscription_failed"
    INVOICE_OVERDUE = "invoice_overdue"
    REPEATED_FAILURE = "repeated_failure"


class FailureReason(str, Enum):
    TRANSIENT_BANK_FAILURE = "transient_bank_failure"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    CARD_EXPIRED = "card_expired"
    ISSUER_DECLINE = "issuer_decline"
    NETWORK_TIMEOUT = "network_timeout"
    CUSTOMER_ABANDONED = "customer_abandoned"
    SUBSCRIPTION_LAPSED = "subscription_lapsed"
    INVOICE_UNPAID = "invoice_unpaid"
    UNKNOWN = "unknown"


class ActionType(str, Enum):
    RETRY_PAYMENT = "RETRY_PAYMENT"
    CREATE_PAYMENT_LINK = "CREATE_PAYMENT_LINK"
    SEND_REMINDER = "SEND_REMINDER"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"
    NO_ACTION = "NO_ACTION"


class PolicyVerdict(str, Enum):
    PASSED = "PASSED"
    BLOCKED = "BLOCKED"
    REQUIRES_HUMAN_APPROVAL = "REQUIRES_HUMAN_APPROVAL"


class ExecutionResult(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PENDING = "PENDING"
    NOT_EXECUTED = "NOT_EXECUTED"


# --------------------------------------------------------------------------- #
# Core records
# --------------------------------------------------------------------------- #

@dataclass
class RevenueEvent:
    """A single revenue-at-risk event ingested from the merchant transaction stream."""
    transaction_id: str
    merchant_id: str
    customer_id: str
    amount: float
    event_type: EventType
    failure_reason: FailureReason
    payment_method: str

    customer_previous_payments: int
    customer_previous_failures: int
    days_since_last_purchase: int
    subscription_status: str  # active | lapsed | none
    checkout_abandoned: bool
    invoice_overdue: bool
    retry_count: int
    customer_value: float  # lifetime value, used for prioritization

    # Optional real-world contact fields — populated for live events coming
    # off an actual merchant webhook/CRM, absent in the synthetic dataset.
    # Only sent to Razorpay's payment_link.create() when actually present
    # (see executor.py) — never sent as blank strings.
    customer_email: str | None = None
    customer_phone: str | None = None

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_id: str = field(default_factory=lambda: f"EVT_{uuid.uuid4().hex[:10].upper()}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        d["failure_reason"] = self.failure_reason.value
        return d


@dataclass
class CounterfactualOption:
    """One candidate intervention and its evaluated expected outcome."""
    action: ActionType
    probability_of_success: float
    intervention_cost: float
    risk_penalty: float
    customer_friction: float  # 0-1, higher = more annoying to customer
    expected_recovery: float  # transaction_value * probability - cost - risk_penalty

    def to_dict(self) -> dict:
        d = asdict(self)
        d["action"] = self.action.value
        return d


@dataclass
class RevenueAutopsy:
    """Step 2 of the workflow: diagnosis of *why* revenue is at risk."""
    root_cause: FailureReason
    customer_intent: str  # "high" | "medium" | "low"
    recovery_eligibility: str  # "HIGH" | "MEDIUM" | "LOW" | "NONE"
    previous_success_rate: float
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["root_cause"] = self.root_cause.value
        return d


@dataclass
class PolicyCheck:
    verdict: PolicyVerdict
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"verdict": self.verdict.value, "reasons": self.reasons}


@dataclass
class Decision:
    """The engine's full decision for one event: everything computed before execution."""
    event_id: str
    transaction_id: str
    autopsy: RevenueAutopsy
    counterfactuals: list[CounterfactualOption]
    selected_action: ActionType
    selected_expected_recovery: float
    best_possible_expected_recovery: float  # max over all counterfactuals, pre-policy
    confidence: float
    policy_check: PolicyCheck

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "transaction_id": self.transaction_id,
            "autopsy": self.autopsy.to_dict(),
            "counterfactuals": [c.to_dict() for c in self.counterfactuals],
            "selected_action": self.selected_action.value,
            "selected_expected_recovery": self.selected_expected_recovery,
            "best_possible_expected_recovery": self.best_possible_expected_recovery,
            "confidence": self.confidence,
            "policy_check": self.policy_check.to_dict(),
        }


@dataclass
class ExecutionOutcome:
    """Step 11: what actually happened after the approved action was executed."""
    result: ExecutionResult
    amount_recovered: float
    executed_action: ActionType
    executed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["result"] = self.result.value
        d["executed_action"] = self.executed_action.value
        return d


@dataclass
class LedgerEntry:
    """One immutable row of the Counterfactual Ledger — the full audit trail for one event."""
    event: RevenueEvent
    decision: Decision
    outcome: ExecutionOutcome
    prediction_error: float  # actual_recovered - selected_expected_recovery
    revenue_left_on_table: float  # best_possible_expected_recovery - actual_recovered (floored at 0)
    system: str = "counterfactual_agent"  # or "fixed_rule_baseline"

    def to_dict(self) -> dict:
        return {
            "event": self.event.to_dict(),
            "decision": self.decision.to_dict(),
            "outcome": self.outcome.to_dict(),
            "prediction_error": self.prediction_error,
            "revenue_left_on_table": self.revenue_left_on_table,
            "system": self.system,
        }
