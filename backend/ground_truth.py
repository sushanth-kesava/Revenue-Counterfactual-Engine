"""
Synthetic Benchmark Environment (formerly "ground_truth.py").

IMPORTANT DISTINCTION:
This module provides a SYNTHETIC counterfactual outcome model for
offline benchmarking. It does NOT represent real observed causal effects.

It is used by executor.SimulatedExecutor to sample outcomes for the
offline benchmark. The "true probability" here is the benchmark's
canonical outcome model — a fixed function that neither the agent nor
the baseline can access or influence.

This ensures:
  1. The benchmark is fair: outcomes don't depend on which system chose
     the action.
  2. Prediction error is meaningful (the agent's model ≠ the benchmark's
     canonical model).

TERMINOLOGY:
  - "benchmark_probability" = P(recovery) in the synthetic environment
  - NOT "ground truth" in the causal-inference sense
  - NOT observed real-world recovery rates

Only executor.SimulatedExecutor should import this module.
"""
from __future__ import annotations

from .models import RevenueEvent, ActionType, FailureReason
from .context import CustomerContext, reconstruct_context


def benchmark_probability(event: RevenueEvent, action: ActionType, ctx: CustomerContext | None = None) -> float:
    """
    Canonical recovery probability in the synthetic benchmark environment.

    This is deliberately DIFFERENT from the ML model's predictions — the
    gap between the model's estimate and this function is what makes the
    benchmark non-tautological.
    """
    ctx = ctx or reconstruct_context(event)

    base = {
        ActionType.RETRY_PAYMENT: {
            FailureReason.TRANSIENT_BANK_FAILURE: 0.75,
            FailureReason.NETWORK_TIMEOUT: 0.70,
            FailureReason.ISSUER_DECLINE: 0.32,
            FailureReason.INSUFFICIENT_FUNDS: 0.22,
        }.get(event.failure_reason, 0.28),
        ActionType.CREATE_PAYMENT_LINK: 0.48,
        ActionType.SEND_REMINDER: 0.26,
        ActionType.ESCALATE_TO_HUMAN: 0.58,
        ActionType.NO_ACTION: 0.0,
    }[action]

    adj = base
    adj += 0.22 * (ctx.intent_score - 0.5)
    if action == ActionType.RETRY_PAYMENT:
        adj -= 0.13 * event.retry_count
    if action == ActionType.CREATE_PAYMENT_LINK and event.checkout_abandoned:
        adj += 0.07
    if action == ActionType.SEND_REMINDER and ctx.is_repeat_customer:
        adj += 0.04
    if ctx.recency_penalty > 0.5:
        adj -= 0.12

    return max(0.01, min(0.98, adj))


# Legacy alias for backward compatibility with existing executor.py imports
true_probability = benchmark_probability
