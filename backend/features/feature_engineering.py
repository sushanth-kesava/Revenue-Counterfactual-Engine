"""
Feature Engineering Module.

Transforms raw benchmark/event data into model-ready feature vectors.
All transformations are deterministic and reproducible.

The feature engineering pipeline:
1. Categorical encoding (one-hot for low-cardinality, label for high-cardinality)
2. Numerical features (as-is, already pre-computed in the benchmark)
3. Boolean flags (cast to int)
4. Risk composite features
5. Customer value composite features
6. Action encoding (for the model: action is an input alongside context)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from .columns import AGENT_INPUT_COLUMNS, EVALUATION_COLUMNS, VALID_ACTIONS


# Categorical columns that need encoding
CATEGORICAL_COLUMNS = [
    "payment_method",
    "device_type",
    "revenue_event",
    "failure_reason",
    "customer_value_segment",
    "automated_recovery_risk",
    "risk_payment_method",
    "risk_device_type",
]

# Boolean columns to cast to int
BOOLEAN_COLUMNS = [
    "recovery_eligible",
    "unusual_amount_flag",
    "unusual_location_flag",
    "multiple_transactions_short_time",
    "high_risk_device_flag",
    "velocity_flag",
    "previous_fraud_flag",
]

# Numerical columns (used as-is)
NUMERICAL_COLUMNS = [
    "transaction_amount",
    "retry_count",
    "customer_age",
    "behavior_transactions",
    "behavior_total_spend",
    "behavior_average_order_value",
    "behavior_avg_session_minutes",
    "behavior_avg_pages",
    "behavior_returning_rate",
    "behavior_avg_rating",
    "behavior_avg_delivery_days",
    "historical_transactions",
    "historical_total_spend",
    "historical_average_order_value",
    "historical_median_order_value",
    "historical_max_order_value",
    "historical_total_quantity",
    "days_since_last_purchase",
    "customer_lifetime_days",
    "purchase_frequency_per_30d",
    "distinct_products",
    "distinct_countries",
    "risk_amount",
    "failed_transactions_24h",
    "transaction_frequency_24h",
    "risk_signal_count",
]


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute composite/derived features from raw columns."""
    out = df.copy()

    # Customer value score: normalized composite
    out["customer_value_score"] = (
        np.log1p(out["historical_total_spend"])
        * np.log1p(out["historical_transactions"])
        / (1 + out["days_since_last_purchase"] / 30.0)
    )

    # Risk composite: count of risk flags active
    risk_flags = [
        "unusual_amount_flag", "unusual_location_flag",
        "multiple_transactions_short_time", "high_risk_device_flag",
        "velocity_flag", "previous_fraud_flag",
    ]
    out["risk_flag_count"] = out[risk_flags].sum(axis=1)

    # Spend ratio: current transaction vs historical average
    out["spend_ratio"] = out["transaction_amount"] / (out["historical_average_order_value"] + 1.0)

    # Recency score: inverse of days since last purchase (bounded)
    out["recency_score"] = 1.0 / (1.0 + out["days_since_last_purchase"] / 30.0)

    # Session engagement score
    out["engagement_score"] = (
        out["behavior_avg_session_minutes"] * out["behavior_avg_pages"] / 100.0
    )

    # Transaction amount bucket (log-scale)
    out["log_transaction_amount"] = np.log1p(out["transaction_amount"])

    return out


def encode_action(action: str) -> dict:
    """One-hot encode a single action for model input."""
    return {f"action_{a}": int(a == action) for a in VALID_ACTIONS}


def build_feature_matrix(
    df: pd.DataFrame,
    actions: Optional[pd.Series] = None,
    fit_encoder: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a complete feature matrix from raw benchmark data.

    Parameters
    ----------
    df : DataFrame with at least AGENT_INPUT_COLUMNS present.
    actions : Series of action strings (e.g. 'RETRY_PAYMENT'). If provided,
              action one-hot columns are added.
    fit_encoder : If True, returns the fitted state (for saving). Currently
                  uses pandas get_dummies which is stateless.

    Returns
    -------
    (feature_df, feature_names) : The encoded feature matrix and ordered column names.
    """
    # Validate no leakage
    leaked = set(df.columns) & set(EVALUATION_COLUMNS)
    # We don't error if they're present in the source df (they will be), we just don't use them
    
    # Start with available input columns
    available = [c for c in AGENT_INPUT_COLUMNS if c in df.columns]
    work = df[available].copy()

    # Compute derived features
    work = compute_derived_features(work)

    # Encode booleans as int
    for col in BOOLEAN_COLUMNS:
        if col in work.columns:
            work[col] = work[col].astype(int)

    # One-hot encode categoricals
    for col in CATEGORICAL_COLUMNS:
        if col in work.columns:
            dummies = pd.get_dummies(work[col], prefix=col, dtype=int)
            work = pd.concat([work.drop(columns=[col]), dummies], axis=1)

    # Drop high-cardinality text columns that slipped through
    text_cols = work.select_dtypes(include=["object"]).columns.tolist()
    if text_cols:
        work = work.drop(columns=text_cols)

    # Add action encoding if provided
    if actions is not None:
        action_dummies = pd.get_dummies(actions, prefix="action", dtype=int)
        # Ensure all action columns exist
        for a in VALID_ACTIONS:
            col_name = f"action_{a}"
            if col_name not in action_dummies.columns:
                action_dummies[col_name] = 0
        work = pd.concat([work.reset_index(drop=True), action_dummies.reset_index(drop=True)], axis=1)

    # Fill any NaN with 0 (e.g. risk_score is all null)
    work = work.fillna(0)

    feature_names = work.columns.tolist()
    return work, feature_names


def extract_features_for_case(row: dict, action: str) -> dict:
    """
    Extract features for a single case + action combination.
    Used at inference time (API / live decisions).

    Parameters
    ----------
    row : dict with at least the AGENT_INPUT_COLUMNS fields
    action : the action being evaluated

    Returns
    -------
    dict of feature_name -> value
    """
    df = pd.DataFrame([row])
    actions = pd.Series([action])
    features, _ = build_feature_matrix(df, actions=actions)
    return features.iloc[0].to_dict()
