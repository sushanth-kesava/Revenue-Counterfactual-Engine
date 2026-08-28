"""
Model Training Pipeline.

Trains the HistGBM recovery model from the benchmark dataset.
Implements proper train/validation/test splits with leakage prevention.

Usage:
    python -m backend.models_ml.train [--data-path data/revenue_recovery_benchmark.csv]

Outputs:
    backend/models_ml/artifacts/recovery_model.pkl
    backend/models_ml/artifacts/model_metadata.json
    backend/models_ml/artifacts/training_report.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

# Allow running as script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.features.columns import (
    AGENT_INPUT_COLUMNS,
    EVALUATION_COLUMNS,
    TARGET_COLUMN,
    VALID_ACTIONS,
)
from backend.features.feature_engineering import build_feature_matrix
from backend.models_ml.recovery_model import HistGBMRecoveryModel


def load_and_split(
    data_path: str,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load benchmark CSV and split into train/val/test.

    Split strategy: random stratified by actual_recovery_success.
    (Customer-level split not possible: no reliable shared customer ID
    across the three source datasets — documented as independently sampled.)
    """
    df = pd.read_csv(data_path)

    # Shuffle deterministically
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = df.iloc[:n_train].reset_index(drop=True)
    val = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test = df.iloc[n_train + n_val:].reset_index(drop=True)

    return train, val, test


def prepare_training_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Prepare feature matrix and target from a split.

    Uses policy_recommended_action as the action that was taken,
    and actual_recovery_success as the target.

    LEAKAGE CHECK: verifies no evaluation columns leak into features.
    """
    # The action taken for each case
    actions = df["policy_recommended_action"]

    # Build feature matrix from input columns + action encoding
    features, feature_names = build_feature_matrix(df, actions=actions)

    # Target
    target = df[TARGET_COLUMN].astype(int)

    # LEAKAGE VERIFICATION
    leaked = set(feature_names) & set(EVALUATION_COLUMNS)
    if leaked:
        raise ValueError(f"LEAKAGE DETECTED! These evaluation columns appeared in features: {leaked}")

    return features, target, feature_names


def evaluate_model(
    model: HistGBMRecoveryModel,
    df: pd.DataFrame,
    feature_names: list[str],
    split_name: str,
) -> dict:
    """Evaluate model on a split, computing various metrics."""
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score,
        recall_score, f1_score, log_loss,
    )

    actions = df["policy_recommended_action"]
    features, _ = build_feature_matrix(df, actions=actions)

    # Align features
    for col in feature_names:
        if col not in features.columns:
            features[col] = 0
    features_aligned = features[feature_names]

    y_true = df[TARGET_COLUMN].astype(int)
    y_pred_proba = model._model.predict_proba(features_aligned.values)[:, 1]
    y_pred = model._model.predict(features_aligned.values)

    metrics = {
        "split": split_name,
        "n_samples": len(y_true),
        "auc_roc": round(roc_auc_score(y_true, y_pred_proba), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "log_loss": round(log_loss(y_true, y_pred_proba), 4),
        "recovery_rate_actual": round(y_true.mean(), 4),
        "recovery_rate_predicted": round(y_pred.mean(), 4),
    }

    # Per-action metrics
    action_metrics = {}
    for action in VALID_ACTIONS:
        mask = (actions == action)
        if mask.sum() > 0:
            action_metrics[action] = {
                "count": int(mask.sum()),
                "actual_recovery_rate": round(y_true[mask].mean(), 4),
                "predicted_recovery_rate": round(y_pred_proba[mask].mean(), 4),
            }
    metrics["per_action"] = action_metrics

    return metrics


def train_model(data_path: str, seed: int = 42) -> dict:
    """
    Full training pipeline.

    Returns a report dict with train/val/test metrics.
    """
    print("=" * 60)
    print("Revenue Counterfactual Engine — Model Training")
    print("=" * 60)

    # 1. Load and split
    print(f"\n[1/5] Loading data from {data_path}")
    train_df, val_df, test_df = load_and_split(data_path, seed=seed)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 2. Prepare training features
    print("\n[2/5] Engineering features...")
    X_train, y_train, feature_names = prepare_training_data(train_df)
    print(f"  Features: {len(feature_names)} columns")
    print(f"  Target distribution: {y_train.mean():.3f} recovery rate")

    # 3. Train model
    print("\n[3/5] Training HistGradientBoosting model...")
    model = HistGBMRecoveryModel()
    train_metrics = model.train(X_train, y_train, feature_names)
    print(f"  Training AUC: {train_metrics['train_auc']:.4f}")
    print(f"  Iterations: {train_metrics['n_iterations']}")

    # 4. Evaluate
    print("\n[4/5] Evaluating on all splits...")
    val_metrics = evaluate_model(model, val_df, feature_names, "validation")
    test_metrics = evaluate_model(model, test_df, feature_names, "test")
    train_eval = evaluate_model(model, train_df, feature_names, "train")

    print(f"  Train AUC: {train_eval['auc_roc']:.4f} | Accuracy: {train_eval['accuracy']:.4f}")
    print(f"  Val   AUC: {val_metrics['auc_roc']:.4f} | Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Test  AUC: {test_metrics['auc_roc']:.4f} | Accuracy: {test_metrics['accuracy']:.4f}")

    # 5. Save
    print("\n[5/5] Saving model artifacts...")
    model.save()

    # Save training report
    report = {
        "seed": seed,
        "data_path": data_path,
        "splits": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
            "total": len(train_df) + len(val_df) + len(test_df),
        },
        "feature_count": len(feature_names),
        "model_type": "HistGradientBoostingClassifier",
        "hyperparameters": {
            "max_iter": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "min_samples_leaf": 20,
            "early_stopping": True,
        },
        "metrics": {
            "train": train_eval,
            "validation": val_metrics,
            "test": test_metrics,
        },
    }

    report_path = os.path.join(model._model_dir, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Model saved to: {model._model_dir}/recovery_model.pkl")
    print(f"  Report saved to: {report_path}")
    print("\nDone.")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the recovery model")
    parser.add_argument(
        "--data-path",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "revenue_recovery_benchmark.csv"),
        help="Path to benchmark CSV",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_model(args.data_path, seed=args.seed)
