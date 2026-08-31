"""
Recovery Propensity Model.

Interface and implementations for estimating:
    P(recovery | action, context)

The model is called by the counterfactual engine to score each candidate
action's expected recovery probability.

Architecture:
    Context + Action → RecoveryModel → P(recovery | action, context)

Implementations:
- HistGBMRecoveryModel: Production model (requires sklearn)
- NumpyLogisticModel: Portable model trained with numpy (no sklearn needed)
- FallbackRuleModel: Hand-calibrated fallback when no trained model exists
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from ..features.columns import VALID_ACTIONS, AGENT_INPUT_COLUMNS
from ..features.feature_engineering import build_feature_matrix


class RecoveryModel(ABC):
    """Abstract interface for recovery probability estimation."""

    @abstractmethod
    def predict_probability(self, context: dict, action: str) -> float:
        """Predict P(recovery | action, context) for a single case."""
        ...

    @abstractmethod
    def predict_action_values(self, context: dict) -> dict[str, float]:
        """Predict P(recovery | action, context) for ALL actions at once."""
        ...

    @abstractmethod
    def is_trained(self) -> bool:
        """Whether the model has been trained / loaded from artifacts."""
        ...


class NumpyLogisticModel(RecoveryModel):
    """
    Portable logistic regression model using only numpy.

    Loads pre-trained weights from JSON (no pickle, no sklearn dependency).
    This is the bootstrap model that ships with the repo — upgrade to
    HistGBMRecoveryModel by running `python run_pipeline.py` with sklearn installed.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self._model_dir = model_dir or os.path.join(
            os.path.dirname(__file__), "artifacts"
        )
        self._weights: Optional[np.ndarray] = None
        self._bias: float = 0.0
        self._feat_mean: Optional[np.ndarray] = None
        self._feat_std: Optional[np.ndarray] = None
        self._feature_names: list[str] = []

    def is_trained(self) -> bool:
        return self._weights is not None

    def load(self) -> bool:
        weights_path = os.path.join(self._model_dir, "recovery_model_weights.json")
        if not os.path.exists(weights_path):
            return False

        with open(weights_path, "r") as f:
            data = json.load(f)

        self._weights = np.array(data["weights"])
        self._bias = data["bias"]
        self._feat_mean = np.array(data["feature_mean"])
        self._feat_std = np.array(data["feature_std"])
        self._feature_names = data["feature_names"]
        return True

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        X_norm = (X - self._feat_mean) / self._feat_std
        return self._sigmoid(X_norm @ self._weights + self._bias)

    def predict_probability(self, context: dict, action: str) -> float:
        if not self.is_trained():
            raise RuntimeError("Model not loaded.")

        df = pd.DataFrame([context])
        actions = pd.Series([action])
        features, _ = build_feature_matrix(df, actions=actions)
        features_aligned = self._align_features(features)
        proba = self._predict_raw(features_aligned.values.astype(np.float64))
        return float(proba[0])

    def predict_action_values(self, context: dict) -> dict[str, float]:
        if not self.is_trained():
            raise RuntimeError("Model not loaded.")
        return {action: self.predict_probability(context, action) for action in VALID_ACTIONS}

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        for col in self._feature_names:
            if col not in features.columns:
                features[col] = 0
        return features[self._feature_names].fillna(0)


class HistGBMRecoveryModel(RecoveryModel):
    """
    HistGradientBoosting-based recovery model (requires sklearn).

    Use `python run_pipeline.py` to train this model.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self._model = None
        self._feature_names: list[str] = []
        self._model_dir = model_dir or os.path.join(
            os.path.dirname(__file__), "artifacts"
        )

    def is_trained(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        import pickle
        model_path = os.path.join(self._model_dir, "recovery_model.pkl")
        meta_path = os.path.join(self._model_dir, "model_metadata.json")

        if not os.path.exists(model_path):
            return False

        with open(model_path, "rb") as f:
            self._model = pickle.load(f)

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self._feature_names = meta.get("feature_names", [])
        return True

    def save(self) -> None:
        import pickle
        os.makedirs(self._model_dir, exist_ok=True)
        with open(os.path.join(self._model_dir, "recovery_model.pkl"), "wb") as f:
            pickle.dump(self._model, f)
        with open(os.path.join(self._model_dir, "model_metadata.json"), "w") as f:
            json.dump({"feature_names": self._feature_names, "model_type": "HistGradientBoostingClassifier", "n_features": len(self._feature_names)}, f, indent=2)

    def train(self, X: pd.DataFrame, y: pd.Series, feature_names: list[str]) -> dict:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score, accuracy_score

        self._feature_names = feature_names
        X_aligned = X[feature_names] if set(feature_names).issubset(X.columns) else X

        self._model = HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, learning_rate=0.1,
            min_samples_leaf=50, max_bins=255, random_state=42,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
        )
        self._model.fit(X_aligned.values, y.values)

        y_pred_proba = self._model.predict_proba(X_aligned.values)[:, 1]
        return {
            "train_auc": round(roc_auc_score(y, y_pred_proba), 4),
            "train_accuracy": round(accuracy_score(y, self._model.predict(X_aligned.values)), 4),
            "n_samples": len(y),
            "n_features": len(feature_names),
            "n_iterations": self._model.n_iter_,
        }

    def predict_probability(self, context: dict, action: str) -> float:
        if not self.is_trained():
            raise RuntimeError("Model not trained. Call train() or load() first.")
        df = pd.DataFrame([context])
        actions = pd.Series([action])
        features, _ = build_feature_matrix(df, actions=actions)
        features_aligned = self._align_features(features)
        proba = self._model.predict_proba(features_aligned.values)[:, 1]
        return float(proba[0])

    def predict_action_values(self, context: dict) -> dict[str, float]:
        if not self.is_trained():
            raise RuntimeError("Model not trained.")
        return {action: self.predict_probability(context, action) for action in VALID_ACTIONS}

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        for col in self._feature_names:
            if col not in features.columns:
                features[col] = 0
        return features[self._feature_names].fillna(0)


class FallbackRuleModel(RecoveryModel):
    """
    Fallback model using hand-calibrated priors.
    Used when no trained model is available.
    """

    BASE_PROBS = {
        "RETRY_PAYMENT": 0.50,
        "CREATE_PAYMENT_LINK": 0.45,
        "SEND_REMINDER": 0.28,
        "ESCALATE_TO_HUMAN": 0.55,
        "NO_ACTION": 0.0,
    }

    # Failure-reason-specific base probabilities for RETRY
    RETRY_BY_FAILURE = {
        "transient_bank_failure": 0.72,
        "network_timeout": 0.68,
        "issuer_decline": 0.35,
        "insufficient_funds": 0.25,
        "temporary_bank_error": 0.70,
        "bank_timeout": 0.68,
    }

    def is_trained(self) -> bool:
        return True

    def predict_probability(self, context: dict, action: str) -> float:
        base = self.BASE_PROBS.get(action, 0.0)
        if action == "NO_ACTION":
            return 0.0

        # Failure-reason-specific adjustment for RETRY
        if action == "RETRY_PAYMENT":
            failure_reason = context.get("failure_reason", "")
            base = self.RETRY_BY_FAILURE.get(failure_reason, base)

        adj = base
        segment = context.get("customer_value_segment", "medium")
        if segment == "very_high": adj += 0.08
        elif segment == "high": adj += 0.04
        elif segment == "low": adj -= 0.05

        retry_count = context.get("retry_count", 0)
        if action == "RETRY_PAYMENT": adj -= 0.10 * retry_count

        # Use risk_signal_count for risk adjustment (not the leaky automated_recovery_risk)
        rsc = context.get("risk_signal_count", 0)
        if rsc >= 3: adj -= 0.08   # high risk = lower recovery chance
        elif rsc == 0: adj += 0.03  # no risk signals = slight boost

        # Returning customer boost
        hist_txns = context.get("historical_transactions", 0)
        if hist_txns >= 5:
            adj += 0.05

        return max(0.02, min(0.95, adj))

    def predict_action_values(self, context: dict) -> dict[str, float]:
        return {action: self.predict_probability(context, action) for action in VALID_ACTIONS}


def get_model(model_dir: Optional[str] = None) -> RecoveryModel:
    """Factory: try HistGBM (pkl) → NumpyLogistic (json) → FallbackRules."""
    model_dir = model_dir or os.path.join(os.path.dirname(__file__), "artifacts")

    # Try sklearn model first (best performance)
    try:
        hgbm = HistGBMRecoveryModel(model_dir=model_dir)
        if hgbm.load():
            return hgbm
    except (ImportError, Exception):
        pass

    # Try portable numpy model
    numpy_model = NumpyLogisticModel(model_dir=model_dir)
    if numpy_model.load():
        return numpy_model

    # Fallback to rules
    return FallbackRuleModel()
