"""
Configuration for the counterfactual decision engine.

All tunable parameters are centralized here — intervention costs,
risk penalties, and model hyperparameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InterventionCosts:
    """Configurable costs for each action type.

    Costs represent the operational/business cost of taking an action,
    independent of whether the action succeeds.
    """
    RETRY_PAYMENT: float = 5.0          # small processing cost
    CREATE_PAYMENT_LINK: float = 8.0    # generation + delivery cost
    SEND_REMINDER: float = 2.0          # minimal cost (SMS/email)
    ESCALATE_TO_HUMAN: float = 50.0     # human agent time cost
    NO_ACTION: float = 0.0              # zero cost

    def get(self, action: str) -> float:
        return getattr(self, action, 0.0)


@dataclass
class RiskPenalties:
    """Risk-based penalties applied to expected utility calculation.

    Higher penalties discourage actions on risky transactions where
    the downside (e.g. false positive charge, customer friction)
    outweighs the expected recovery.
    """
    high_risk_multiplier: float = 0.15     # penalty factor for high-risk cases
    medium_risk_multiplier: float = 0.05   # penalty factor for medium-risk
    low_risk_multiplier: float = 0.0       # no penalty for low-risk

    def get_multiplier(self, risk_tier: str) -> float:
        mapping = {
            "high": self.high_risk_multiplier,
            "medium": self.medium_risk_multiplier,
            "low": self.low_risk_multiplier,
        }
        return mapping.get(risk_tier, self.medium_risk_multiplier)


@dataclass
class DecisionConfig:
    """Full decision engine configuration."""
    intervention_costs: InterventionCosts = field(default_factory=InterventionCosts)
    risk_penalties: RiskPenalties = field(default_factory=RiskPenalties)

    # Minimum expected incremental value over NO_ACTION to justify intervention
    min_intervention_threshold: float = 10.0

    # Model confidence threshold below which we defer to NO_ACTION or ESCALATE
    min_model_confidence: float = 0.20

    # Random seed for reproducibility
    seed: int = 42


DEFAULT_CONFIG = DecisionConfig()
