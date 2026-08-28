"""
Step 12 — Counterfactual Ledger.

The append-only audit trail for every decision the system has made. Kept
as a thin, storage-agnostic wrapper: `JSONLedger` is used both for the
offline benchmark/dataset pipeline (data/ledger.json) and, separately, as
the LIVE ledger for real decide -> execute -> webhook-reconcile cycles
(data/live_ledger.json — see backend/main.py). The same `LedgerEntry`
records can be persisted to Postgres in a real deployment without
changing anything upstream.

NOTE on concurrency: this JSON-file implementation is fine for a demo or
low-volume pilot, but is NOT safe for concurrent writers (no file locking,
last-write-wins). A real deployment handling live webhook traffic should
swap this for a Postgres-backed ledger with a unique constraint on
transaction_id — the interface (record / find_by_transaction_id /
update_outcome / save) is deliberately narrow so that swap doesn't touch
any other module.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field

from .models import LedgerEntry, ExecutionOutcome


@dataclass
class JSONLedger:
    path: str
    entries: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, entry: LedgerEntry) -> None:
        with self._lock:
            self.entries.append(entry)
            self._save_locked()

    def find_by_transaction_id(self, transaction_id: str):
        for entry in self.entries:
            if entry.event.transaction_id == transaction_id:
                return entry
        return None

    def update_outcome(self, transaction_id: str, outcome: ExecutionOutcome):
        """Used by the webhook handler to move an entry from PENDING to a
        final SUCCESS/FAILURE outcome once Razorpay confirms what actually
        happened, and to recompute the derived metrics that depend on it."""
        with self._lock:
            entry = self.find_by_transaction_id(transaction_id)
            if entry is None:
                return None
            entry.outcome = outcome
            entry.prediction_error = round(outcome.amount_recovered - entry.decision.selected_expected_recovery, 2)
            entry.revenue_left_on_table = max(
                0.0, round(entry.decision.best_possible_expected_recovery - outcome.amount_recovered, 2)
            )
            self._save_locked()
            return entry

    def load(self) -> None:
        if not os.path.exists(self.path):
            self.entries = []
            return
        with open(self.path) as f:
            raw = json.load(f)
        self.entries = [_ledger_entry_from_dict(r) for r in raw]

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump([e.to_dict() for e in self.entries], f, indent=2)

    def __len__(self) -> int:
        return len(self.entries)


def _ledger_entry_from_dict(raw: dict) -> LedgerEntry:
    """Reconstructs a LedgerEntry from its to_dict() form. This is the
    inverse of LedgerEntry.to_dict() in models.py, kept here (rather than
    as a from_dict on each dataclass) so the (de)serialization boundary
    lives in one place, next to the store that needs it."""
    from .models import (
        RevenueEvent, EventType, FailureReason, RevenueAutopsy, CounterfactualOption,
        ActionType, PolicyCheck, PolicyVerdict, Decision, ExecutionResult,
    )

    ev = raw["event"]
    event = RevenueEvent(
        transaction_id=ev["transaction_id"], merchant_id=ev["merchant_id"], customer_id=ev["customer_id"],
        amount=ev["amount"], event_type=EventType(ev["event_type"]), failure_reason=FailureReason(ev["failure_reason"]),
        payment_method=ev["payment_method"], customer_previous_payments=ev["customer_previous_payments"],
        customer_previous_failures=ev["customer_previous_failures"], days_since_last_purchase=ev["days_since_last_purchase"],
        subscription_status=ev["subscription_status"], checkout_abandoned=ev["checkout_abandoned"],
        invoice_overdue=ev["invoice_overdue"], retry_count=ev["retry_count"], customer_value=ev["customer_value"],
        customer_email=ev.get("customer_email"), customer_phone=ev.get("customer_phone"),
        created_at=ev["created_at"], event_id=ev["event_id"],
    )
    d = raw["decision"]
    decision = Decision(
        event_id=d["event_id"], transaction_id=d["transaction_id"],
        autopsy=RevenueAutopsy(
            root_cause=FailureReason(d["autopsy"]["root_cause"]), customer_intent=d["autopsy"]["customer_intent"],
            recovery_eligibility=d["autopsy"]["recovery_eligibility"], previous_success_rate=d["autopsy"]["previous_success_rate"],
            notes=d["autopsy"].get("notes", ""),
        ),
        counterfactuals=[
            CounterfactualOption(
                action=ActionType(c["action"]), probability_of_success=c["probability_of_success"],
                intervention_cost=c["intervention_cost"], risk_penalty=c["risk_penalty"],
                customer_friction=c["customer_friction"], expected_recovery=c["expected_recovery"],
            ) for c in d["counterfactuals"]
        ],
        selected_action=ActionType(d["selected_action"]), selected_expected_recovery=d["selected_expected_recovery"],
        best_possible_expected_recovery=d["best_possible_expected_recovery"], confidence=d["confidence"],
        policy_check=PolicyCheck(verdict=PolicyVerdict(d["policy_check"]["verdict"]), reasons=d["policy_check"]["reasons"]),
    )
    o = raw["outcome"]
    outcome = ExecutionOutcome(
        result=ExecutionResult(o["result"]), amount_recovered=o["amount_recovered"],
        executed_action=ActionType(o["executed_action"]), executed_at=o["executed_at"],
    )
    return LedgerEntry(
        event=event, decision=decision, outcome=outcome,
        prediction_error=raw["prediction_error"], revenue_left_on_table=raw["revenue_left_on_table"],
        system=raw.get("system", "counterfactual_agent"),
    )
