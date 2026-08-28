"""
Step 10-11 — Agent Execution and Outcome Observation.

This module executes a policy-approved action and observes the real
result. Two backends are supported:

  * RazorpayLiveExecutor  — wraps the real `razorpay` Python SDK against
    Razorpay TEST MODE keys. Use this in a real deployment / demo where
    network access to api.razorpay.com and valid test keys are available.

  * SimulatedExecutor     — a deterministic outcome simulator used for the
    offline benchmark (Section 15-16 of the spec: 500+ synthetic events,
    held-out test set). It draws a Bernoulli outcome using the SAME
    probability the counterfactual engine estimated for the selected
    action, plus independently-sampled noise, so that the benchmark
    reflects estimation error rather than being tautological.

Both implement the same `execute()` interface, so swapping between them
(e.g. via an environment variable) requires no changes anywhere else in
the pipeline.
"""
from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod

from .models import RevenueEvent, Decision, ActionType, ExecutionResult, ExecutionOutcome
from .context import reconstruct_context
from .ground_truth import true_probability


class BaseExecutor(ABC):
    @abstractmethod
    def execute(self, event: RevenueEvent, decision: Decision) -> ExecutionOutcome:
        ...


class SimulatedExecutor(BaseExecutor):
    """Deterministic (seedable) outcome simulator for offline evaluation."""

    def __init__(self, seed: int = 7, noise_std: float = 0.08):
        self._rng = random.Random(seed)
        self.noise_std = noise_std

    def execute(self, event: RevenueEvent, decision: Decision) -> ExecutionOutcome:
        action = decision.selected_action

        if action in (ActionType.NO_ACTION,):
            return ExecutionOutcome(
                result=ExecutionResult.NOT_EXECUTED, amount_recovered=0.0, executed_action=action
            )

        # Outcomes are simulated against the CANONICAL ground-truth
        # probability for this (event, action) pair — never against
        # whichever system's own estimate chose the action. This is what
        # keeps the baseline-vs-agent benchmark honest: reality doesn't
        # care what a system believed, only what action it took.
        ctx = reconstruct_context(event)
        p_true = true_probability(event, action, ctx)

        if action == ActionType.ESCALATE_TO_HUMAN:
            success = self._rng.random() < p_true
            recovered = event.amount if success else 0.0
            return ExecutionOutcome(
                result=ExecutionResult.SUCCESS if success else ExecutionResult.PENDING,
                amount_recovered=recovered,
                executed_action=action,
            )

        # Add independent sampling noise on top of the canonical
        # probability to represent real-world stochasticity beyond what
        # any model (baseline or agent) could account for.
        p_sampled = max(0.01, min(0.98, p_true + self._rng.gauss(0, self.noise_std)))

        success = self._rng.random() < p_sampled
        recovered = round(event.amount, 2) if success else 0.0

        return ExecutionOutcome(
            result=ExecutionResult.SUCCESS if success else ExecutionResult.FAILURE,
            amount_recovered=recovered,
            executed_action=action,
        )


class RazorpayLiveExecutor(BaseExecutor):
    """Wraps the real Razorpay Python SDK in TEST MODE.

    Requires:
        pip install razorpay
        RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET env vars set to TEST keys
        (they always start with rzp_test_...)

    NOT used by the offline benchmark — this is the path wired up for the
    live demo / Phase 5 of the roadmap, where approved actions actually
    hit Razorpay's sandbox.
    """

    def __init__(self):
        try:
            import razorpay  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "razorpay SDK not installed. Run `pip install razorpay` to use "
                "RazorpayLiveExecutor, or use SimulatedExecutor for offline runs."
            ) from e

        key_id = os.environ.get("RAZORPAY_KEY_ID")
        key_secret = os.environ.get("RAZORPAY_KEY_SECRET")
        if not key_id or not key_secret:
            raise RuntimeError(
                "RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET not set. Get TEST MODE keys "
                "from the Razorpay dashboard (Settings -> API Keys)."
            )
        if not key_id.startswith("rzp_test_"):
            raise RuntimeError("Refusing to run with a non-test Razorpay key in this executor.")

        self.client = razorpay.Client(auth=(key_id, key_secret))
        # SDK-native retry with exponential backoff + jitter on ConnectionError/
        # Timeout — real revenue recovery traffic should not fail a decision
        # cycle over a single dropped connection.
        self.client.enable_retry(True)
        self._razorpay = razorpay

    def execute(self, event: RevenueEvent, decision: Decision) -> ExecutionOutcome:
        from razorpay.errors import BadRequestError, GatewayError, ServerError
        import requests as _requests

        action = decision.selected_action
        amount_paise = int(round(event.amount * 100))

        try:
            if action == ActionType.RETRY_PAYMENT:
                # Razorpay's API does not let you replay a failed payment_id
                # directly — a "retry" is modeled as a fresh Order against the
                # same transaction reference, which the customer (or a saved
                # payment method / auto-recurring charge, if set up) then pays.
                order = self.client.order.create(
                    {
                        "amount": amount_paise,
                        "currency": "INR",
                        "receipt": f"retry_{event.transaction_id}",
                        "notes": {
                            "source": "revenue_counterfactual_engine",
                            "reason": "retry",
                            "transaction_id": event.transaction_id,
                        },
                    }
                )
                # Real capture confirmation only arrives via the payment.captured /
                # payment.failed webhook (see main.py) — this executor's job ends
                # at "the retry attempt was created", not at "money moved".
                return ExecutionOutcome(result=ExecutionResult.PENDING, amount_recovered=0.0, executed_action=action)

            if action == ActionType.CREATE_PAYMENT_LINK:
                payload = {
                    "amount": amount_paise,
                    "currency": "INR",
                    "description": f"Complete your payment for {event.transaction_id}",
                    "reference_id": event.transaction_id,  # required to reconcile the payment_link.paid webhook
                    "notify": {"sms": True, "email": True},
                    "notes": {"source": "revenue_counterfactual_engine", "transaction_id": event.transaction_id},
                }
                # Only send customer contact details we actually have — sending
                # blank strings is worse than omitting the field, and Razorpay's
                # own docs note these are optional.
                customer = {}
                if getattr(event, "customer_email", None):
                    customer["email"] = event.customer_email
                if getattr(event, "customer_phone", None):
                    customer["contact"] = event.customer_phone
                if customer:
                    payload["customer"] = customer

                link = self.client.payment_link.create(payload)
                return ExecutionOutcome(result=ExecutionResult.PENDING, amount_recovered=0.0, executed_action=action)

            if action == ActionType.SEND_REMINDER:
                # Reminder delivery is out of Razorpay's scope — wire to your
                # notification provider (SMS/email/WhatsApp) here.
                return ExecutionOutcome(result=ExecutionResult.PENDING, amount_recovered=0.0, executed_action=action)

            if action == ActionType.ESCALATE_TO_HUMAN:
                return ExecutionOutcome(result=ExecutionResult.PENDING, amount_recovered=0.0, executed_action=action)

            return ExecutionOutcome(result=ExecutionResult.NOT_EXECUTED, amount_recovered=0.0, executed_action=action)

        except (BadRequestError, GatewayError, ServerError) as e:
            # A real Razorpay API error (bad params, gateway down, etc). Fail
            # the execution cleanly into the ledger rather than crashing the
            # request — the policy engine already approved the *decision*;
            # this is an execution-layer failure, and should be visible in
            # the ledger as such, not silently swallowed or a 500.
            return ExecutionOutcome(
                result=ExecutionResult.FAILURE, amount_recovered=0.0, executed_action=action
            )
        except _requests.exceptions.RequestException:
            # Network-level failure that survived the SDK's own retry policy.
            return ExecutionOutcome(
                result=ExecutionResult.FAILURE, amount_recovered=0.0, executed_action=action
            )


def get_default_executor() -> BaseExecutor:
    """Selects the live Razorpay executor if credentials are configured,
    otherwise falls back to the offline simulator."""
    if os.environ.get("RAZORPAY_KEY_ID") and os.environ.get("RAZORPAY_KEY_SECRET"):
        try:
            return RazorpayLiveExecutor()
        except RuntimeError:
            pass
    return SimulatedExecutor()
