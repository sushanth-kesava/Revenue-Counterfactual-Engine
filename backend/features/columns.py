"""
Column configuration for leakage prevention.

AGENT_INPUT_COLUMNS: Features the model/agent is allowed to see at decision time.
EVALUATION_COLUMNS: Target/outcome columns that must NEVER be used as model inputs.

This separation is the primary leakage-prevention mechanism.
"""

# Features available to the model at decision time
AGENT_INPUT_COLUMNS = [
    # Transaction context
    "transaction_amount",
    "payment_method",
    "device_type",
    "revenue_event",
    "failure_reason",
    "retry_count",
    "recovery_eligible",

    # Customer demographics
    "customer_age",
    "customer_city",

    # Current behavior signals
    "behavior_transactions",
    "behavior_total_spend",
    "behavior_average_order_value",
    "behavior_avg_session_minutes",
    "behavior_avg_pages",
    "behavior_returning_rate",
    "behavior_avg_rating",
    "behavior_avg_delivery_days",

    # Historical customer value
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
    "customer_value_segment",

    # Risk signals
    "risk_amount",
    "risk_payment_method",
    "risk_device_type",
    "failed_transactions_24h",
    "transaction_frequency_24h",
    "unusual_amount_flag",
    "unusual_location_flag",
    "multiple_transactions_short_time",
    "high_risk_device_flag",
    "velocity_flag",
    "previous_fraud_flag",
    "risk_signal_count",
]

# Columns that encode the action taken (used as model input alongside context)
ACTION_COLUMN = "action"

# Columns that must NEVER be used as model inputs - these are targets/labels
EVALUATION_COLUMNS = [
    "retry_expected_probability",
    "payment_link_expected_probability",
    "reminder_expected_probability",
    "escalation_expected_probability",
    "retry_expected_recovery",
    "payment_link_expected_recovery",
    "reminder_expected_recovery",
    "escalation_expected_recovery",
    "counterfactual_optimal_action_unconstrained",
    "estimated_optimal_recovery_unconstrained",
    "policy_recommended_action",
    "selected_action_probability",
    "actual_outcome",
    "actual_recovery_success",
    "actual_recovered_amount",
    "baseline_action",
    "baseline_expected_recovery",
    "baseline_recovery_success",
    "baseline_actual_recovered_amount",
]

# Target column for the recovery model
TARGET_COLUMN = "actual_recovery_success"

# Metadata columns (not features, not targets)
METADATA_COLUMNS = [
    "case_id",
    "favorite_product",  # too high cardinality / not useful for model
    "high_risk_case",    # derived from risk signals, could leak
]

# Valid action types for the counterfactual engine
VALID_ACTIONS = [
    "RETRY_PAYMENT",
    "CREATE_PAYMENT_LINK",
    "SEND_REMINDER",
    "ESCALATE_TO_HUMAN",
    "NO_ACTION",
]

# Columns excluded from model features due to potential leakage.
# 'automated_recovery_risk' is excluded because it is counter-intuitively
# correlated with recovery success (high "risk" = 82% recovery rate),
# suggesting it encodes outcome information. It was likely derived from
# recovery suitability rather than representing genuine risk.
EXCLUDED_LEAKY_COLUMNS = [
    "automated_recovery_risk",
    "high_risk_case",      # inverse of recovery_eligible, outcome-correlated
    "recovery_eligible",   # suspiciously anti-correlated with actual recovery
]
