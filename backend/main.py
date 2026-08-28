"""
FastAPI application exposing the Revenue Counterfactual Engine.

Razorpay-Integrated Endpoints:
  GET  /api/health
  GET  /api/benchmark                 -> full baseline-vs-agent benchmark (cached)
  GET  /api/dashboard                 -> summary numbers for the top dashboard cards
  GET  /api/ledger?system=agent|baseline&limit=N   -> paged ledger entries
  GET  /api/decision/{case_id}        -> single-case decision trace
  POST /api/decide                    -> submit a raw event JSON, get a live decision back
  POST /api/execute                   -> decide + execute through live executor
  POST /api/razorpay/recover          -> Razorpay-native: accepts payment.failed payload format
  GET  /api/live-ledger               -> real-time ledger of /api/execute calls
  GET  /api/segments                  -> segment breakdown metrics
  POST /api/webhook/razorpay          -> webhook receiver for Razorpay reconciliation

Run with:  uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import os
import hmac
import hashlib
from typing import Optional

from fastapi import FastAPI, HTTPException, Body, Request, Header
from fastapi.middleware.cors import CORSMiddleware

from .evaluation import run_benchmark, run_data_driven_benchmark
from .models import RevenueEvent, EventType, FailureReason, ActionType, ExecutionResult, ExecutionOutcome, LedgerEntry
from .counterfactual_engine import evaluate_counterfactuals
from .policy_engine import apply_policy, DEFAULT_POLICY
from .executor import get_default_executor
from .ledger import JSONLedger
from .models_ml.recovery_model import get_model
from .models_ml.config import DEFAULT_CONFIG

app = FastAPI(title="Revenue Counterfactual Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_BENCHMARK_CACHE: Optional[dict] = None
_BENCHMARK_CSV = os.path.join(_DATA_DIR, "revenue_recovery_benchmark.csv")

# Live ledger
_live_ledger = JSONLedger(path=os.path.join(_DATA_DIR, "live_ledger.json"))
_live_ledger.load()


def _get_benchmark(force_refresh: bool = False) -> dict:
    global _BENCHMARK_CACHE
    cache_path = os.path.join(_DATA_DIR, "benchmark.json")
    if not force_refresh and _BENCHMARK_CACHE is not None:
        return _BENCHMARK_CACHE
    if not force_refresh and os.path.exists(cache_path):
        with open(cache_path) as f:
            _BENCHMARK_CACHE = json.load(f)
            return _BENCHMARK_CACHE
    # Try data-driven benchmark first
    if os.path.exists(_BENCHMARK_CSV):
        _BENCHMARK_CACHE = run_data_driven_benchmark(_BENCHMARK_CSV)
    else:
        _BENCHMARK_CACHE = run_benchmark()
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(_BENCHMARK_CACHE, f, indent=2)
    return _BENCHMARK_CACHE


def _parse_event(payload: dict) -> RevenueEvent:
    try:
        return RevenueEvent(
            transaction_id=payload["transaction_id"],
            merchant_id=payload.get("merchant_id", "MID_LIVE"),
            customer_id=payload.get("customer_id", "CUST_LIVE"),
            amount=float(payload["amount"]),
            event_type=EventType(payload["event_type"]),
            failure_reason=FailureReason(payload.get("failure_reason", "unknown")),
            payment_method=payload.get("payment_method", "card"),
            customer_previous_payments=int(payload.get("customer_previous_payments", 0)),
            customer_previous_failures=int(payload.get("customer_previous_failures", 0)),
            days_since_last_purchase=int(payload.get("days_since_last_purchase", 0)),
            subscription_status=payload.get("subscription_status", "none"),
            checkout_abandoned=bool(payload.get("checkout_abandoned", False)),
            invoice_overdue=bool(payload.get("invoice_overdue", False)),
            retry_count=int(payload.get("retry_count", 0)),
            customer_value=float(payload.get("customer_value", 0)),
            customer_email=payload.get("customer_email"),
            customer_phone=payload.get("customer_phone"),
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid event payload: {e}")


@app.get("/api/health")
def health():
    model = get_model()
    return {
        "status": "ok",
        "model_loaded": model.is_trained(),
        "model_type": type(model).__name__,
    }


@app.get("/api/benchmark")
def get_benchmark(refresh: bool = False):
    return _get_benchmark(force_refresh=refresh)


@app.get("/api/dashboard")
def get_dashboard():
    bench = _get_benchmark()
    agent = bench["agent"]
    baseline = bench["baseline"]
    return {
        "revenue_at_risk": agent["revenue_at_risk"],
        "recovered_revenue": agent["revenue_recovered"],
        "recovery_rate": agent["recovery_rate"],
        "incremental_revenue": agent.get("incremental_revenue", 0),
        "uplift_pct": agent.get("uplift_pct", 0),
        "intervention_rate": agent.get("intervention_rate", 100),
        "escalation_rate": agent["escalation_rate"],
        "no_action_rate": agent.get("no_action_rate", 0),
        "unnecessary_interventions": agent.get("unnecessary_interventions", 0),
        "policy_violations": agent["policy_violations"],
        "avg_recovered_per_case": agent.get("avg_recovered_per_case", 0),
        "baseline_comparison": baseline,
        "dataset_sizes": bench["dataset_sizes"],
    }


@app.get("/api/segments")
def get_segments():
    bench = _get_benchmark()
    return bench.get("segments", {})


@app.get("/api/ledger")
def get_ledger(system: str = "agent", limit: int = 25, offset: int = 0):
    bench = _get_benchmark()
    key = "agent_ledger" if system == "agent" else "baseline_ledger"
    entries = bench.get(key, [])
    return {"total": len(entries), "entries": entries[offset:offset + limit]}


@app.get("/api/decision/{case_id}")
def get_decision_trace(case_id: str, system: str = "agent"):
    bench = _get_benchmark()
    key = "agent_ledger" if system == "agent" else "baseline_ledger"
    for entry in bench.get(key, []):
        if entry.get("case_id") == case_id:
            return entry
        # Legacy format compatibility
        if isinstance(entry, dict) and entry.get("event", {}).get("transaction_id") == case_id:
            return entry
    raise HTTPException(status_code=404, detail=f"No {system} ledger entry for {case_id}")


@app.post("/api/decide")
def decide(event: dict = Body(...)):
    """Run the counterfactual engine on an event payload.

    Returns structured decision with:
      - case_id
      - recommended_action
      - confidence
      - expected_recovery
      - candidate_actions (with probabilities and expected values)
      - safety_constraints
      - risk_tier
      - reasoning
    """
    revenue_event = _parse_event(event)
    decision = evaluate_counterfactuals(revenue_event)
    decision = apply_policy(revenue_event, decision, DEFAULT_POLICY)

    # Build the clean API response (no internal chain-of-thought)
    candidates = []
    for cf in decision.counterfactuals:
        candidates.append({
            "action": cf.action.value,
            "probability": round(cf.probability_of_success, 4),
            "expected_recovery": round(cf.expected_recovery, 2),
            "intervention_cost": cf.intervention_cost,
            "risk_penalty": cf.risk_penalty,
        })

    return {
        "case_id": decision.event_id,
        "recommended_action": decision.selected_action.value,
        "expected_recovery": round(decision.selected_expected_recovery, 2),
        "confidence": round(decision.confidence, 4),
        "candidate_actions": candidates,
        "safety_constraints": decision.policy_check.reasons,
        "risk_tier": getattr(revenue_event, "automated_recovery_risk", "medium") if hasattr(revenue_event, "automated_recovery_risk") else "medium",
        "reasoning": [
            f"Root cause: {decision.autopsy.root_cause.value}",
            f"Customer intent: {decision.autopsy.customer_intent}",
            f"Recovery eligibility: {decision.autopsy.recovery_eligibility}",
            f"Best action probability: {decision.counterfactuals[0].probability_of_success:.2%}" if decision.counterfactuals else "",
        ],
    }


@app.post("/api/execute")
def execute(event: dict = Body(...)):
    """Decide AND execute: full decide -> policy -> execute cycle."""
    revenue_event = _parse_event(event)

    existing = _live_ledger.find_by_transaction_id(revenue_event.transaction_id)
    if existing is not None:
        return {"idempotent_replay": True, "entry": existing.to_dict()}

    decision = evaluate_counterfactuals(revenue_event)
    decision = apply_policy(revenue_event, decision, DEFAULT_POLICY)

    executor = get_default_executor()
    outcome = executor.execute(revenue_event, decision)

    entry = LedgerEntry(
        event=revenue_event,
        decision=decision,
        outcome=outcome,
        prediction_error=round(outcome.amount_recovered - decision.selected_expected_recovery, 2),
        revenue_left_on_table=max(0.0, round(decision.best_possible_expected_recovery - outcome.amount_recovered, 2)),
        system="counterfactual_agent",
    )
    _live_ledger.record(entry)
    return {"idempotent_replay": False, "entry": entry.to_dict()}


@app.get("/api/live-ledger")
def get_live_ledger(limit: int = 50, offset: int = 0):
    entries = [e.to_dict() for e in _live_ledger.entries]
    return {"total": len(entries), "entries": entries[offset:offset + limit]}


def _extract_transaction_id(entity: dict) -> Optional[str]:
    """Pulls transaction_id from a Razorpay entity payload."""
    notes = entity.get("notes") or {}
    if isinstance(notes, dict) and notes.get("transaction_id"):
        return notes["transaction_id"]
    if entity.get("reference_id"):
        return entity["reference_id"]
    receipt = entity.get("receipt") or ""
    if receipt.startswith("retry_"):
        return receipt[len("retry_"):]
    return None


@app.post("/api/webhook/razorpay")
async def razorpay_webhook(request: Request, x_razorpay_signature: Optional[str] = Header(None)):
    """Receiver for Razorpay webhook events.

    Verifies HMAC over raw bytes, finds the matching live-ledger entry,
    and updates its outcome from PENDING to final.
    """
    raw_body = await request.body()

    webhook_secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET")
    if webhook_secret and x_razorpay_signature:
        expected_sig = hmac.new(
            webhook_secret.encode(), raw_body, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected_sig, x_razorpay_signature):
            raise HTTPException(status_code=400, detail="Invalid webhook signature")

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    event_type = payload.get("event", "")
    entity = payload.get("payload", {})

    # Navigate to the actual entity
    for key in ("payment", "payment_link", "order"):
        if key in entity and "entity" in entity[key]:
            entity = entity[key]["entity"]
            break

    transaction_id = _extract_transaction_id(entity)
    if not transaction_id:
        raise HTTPException(status_code=400, detail="Could not extract transaction_id from webhook payload")

    is_success = event_type in ("payment.captured", "payment_link.paid", "order.paid")
    amount_recovered = entity.get("amount", 0) / 100.0 if is_success else 0.0

    outcome = ExecutionOutcome(
        result=ExecutionResult.SUCCESS if is_success else ExecutionResult.FAILURE,
        amount_recovered=amount_recovered,
        executed_action=ActionType.RETRY_PAYMENT,  # best guess from context
    )

    updated = _live_ledger.update_outcome(transaction_id, outcome)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"No live-ledger entry for transaction {transaction_id}")

    return {"status": "reconciled", "transaction_id": transaction_id, "outcome": outcome.result.value}


# ============================================================================
# RAZORPAY-NATIVE INTEGRATION ENDPOINTS
# ============================================================================

# Mapping Razorpay error codes to our failure reasons
_RZP_ERROR_MAP = {
    "BAD_REQUEST_ERROR": FailureReason.TRANSIENT_BANK_FAILURE,
    "GATEWAY_ERROR": FailureReason.NETWORK_TIMEOUT,
    "SERVER_ERROR": FailureReason.NETWORK_TIMEOUT,
}

_RZP_REASON_MAP = {
    "payment_failed": FailureReason.TRANSIENT_BANK_FAILURE,
    "bank_declined": FailureReason.ISSUER_DECLINE,
    "insufficient_balance": FailureReason.INSUFFICIENT_FUNDS,
    "network_error": FailureReason.NETWORK_TIMEOUT,
    "card_expired": FailureReason.CARD_EXPIRED,
}


@app.post("/api/razorpay/recover")
def razorpay_recover(payload: dict = Body(...)):
    """
    Razorpay-native endpoint: accepts a payment.failed event payload
    in Razorpay's actual webhook format and returns a recovery decision.

    This is what you'd wire up as a webhook handler in production:
    1. Razorpay sends payment.failed
    2. This endpoint diagnoses, evaluates, decides
    3. Returns the recommended action + executes if auto_execute=true

    Example payload (Razorpay's format):
    {
      "event": "payment.failed",
      "payload": {
        "payment": {
          "entity": {
            "id": "pay_abc123",
            "amount": 895000,  // paise
            "currency": "INR",
            "method": "card",
            "error_code": "BAD_REQUEST_ERROR",
            "error_reason": "payment_failed",
            "notes": {"customer_id": "cust_123"}
          }
        }
      }
    }
    """
    # Parse Razorpay's nested format
    event_type = payload.get("event", "")
    payment_entity = payload.get("payload", {}).get("payment", {}).get("entity", {})

    if not payment_entity:
        raise HTTPException(status_code=422, detail="Missing payload.payment.entity")

    # Convert Razorpay payment to our RevenueEvent format
    amount_inr = payment_entity.get("amount", 0) / 100.0  # paise → ₹
    error_code = payment_entity.get("error_code", "")
    error_reason = payment_entity.get("error_reason", "")
    notes = payment_entity.get("notes", {})

    failure_reason = _RZP_REASON_MAP.get(error_reason, _RZP_ERROR_MAP.get(error_code, FailureReason.UNKNOWN))

    revenue_event = RevenueEvent(
        transaction_id=payment_entity.get("id", f"pay_{os.urandom(4).hex()}"),
        merchant_id=notes.get("merchant_id", "MID_LIVE"),
        customer_id=notes.get("customer_id", "CUST_UNKNOWN"),
        amount=amount_inr,
        event_type=EventType.PAYMENT_FAILED,
        failure_reason=failure_reason,
        payment_method=payment_entity.get("method", "card"),
        customer_previous_payments=int(notes.get("previous_payments", 0)),
        customer_previous_failures=int(notes.get("previous_failures", 0)),
        days_since_last_purchase=int(notes.get("days_since_last_purchase", 0)),
        subscription_status=notes.get("subscription_status", "none"),
        checkout_abandoned=False,
        invoice_overdue=False,
        retry_count=int(notes.get("retry_count", 0)),
        customer_value=float(notes.get("customer_value", 0)),
        customer_email=notes.get("email"),
        customer_phone=notes.get("contact"),
    )

    decision = evaluate_counterfactuals(revenue_event)
    decision = apply_policy(revenue_event, decision, DEFAULT_POLICY)

    # Build Razorpay-friendly response
    return {
        "payment_id": payment_entity.get("id"),
        "amount": amount_inr,
        "decision": {
            "action": decision.selected_action.value,
            "confidence": round(decision.confidence, 3),
            "expected_recovery_inr": round(decision.selected_expected_recovery, 2),
            "reasoning": f"Root cause: {failure_reason.value}, best action probability: {decision.counterfactuals[0].probability_of_success:.0%}" if decision.counterfactuals else "",
        },
        "alternatives": [
            {"action": cf.action.value, "expected_inr": round(cf.expected_recovery, 2), "probability": round(cf.probability_of_success, 3)}
            for cf in decision.counterfactuals[:4]
        ],
        "safety": {
            "verdict": decision.policy_check.verdict.value,
            "reasons": decision.policy_check.reasons,
        },
        "next_step": _action_to_razorpay_call(decision.selected_action),
    }


def _action_to_razorpay_call(action: ActionType) -> str:
    """Map our action to the Razorpay SDK call that would execute it."""
    return {
        ActionType.RETRY_PAYMENT: "razorpay.order.create() → present checkout to customer",
        ActionType.CREATE_PAYMENT_LINK: "razorpay.payment_link.create() → send via SMS/email",
        ActionType.SEND_REMINDER: "Send notification (SMS/email/WhatsApp) with payment link",
        ActionType.ESCALATE_TO_HUMAN: "Route to support team — do not auto-charge",
        ActionType.NO_ACTION: "No intervention — event does not justify cost of action",
    }.get(action, "Unknown action")
