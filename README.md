# Revenue Counterfactual Engine

**Track 03 — AI Revenue Recovery** (Razorpay Buildathon)

> When revenue fails, the best action is **not always retry**.

## Core Idea

```
Payment Failure / At-Risk Event
        │
        v
What could we do?
        │
        ├── Retry Payment
        ├── Create Payment Link
        ├── Send Reminder
        ├── No Action (legitimate option)
        └── Escalate to Human
        │
        v
What is the EXPECTED MONETARY VALUE of each?
        │
        v
Which action is safest AND most valuable?
```

This is a **data-driven counterfactual revenue recovery engine** that:

1. **Builds context** — customer value, transaction history, behavioral signals, risk indicators
2. **Predicts outcomes** — ML model estimates P(recovery | action, context) for each intervention
3. **Computes expected value** — converts probabilities into ₹ amounts minus costs and risk penalties
4. **Applies safety constraints** — deterministic policy gate the AI cannot override
5. **Selects the best action** — including "do nothing" when intervention isn't worth the cost
6. **Executes and records** — immutable counterfactual ledger for auditability
7. **Evaluates against baseline** — proves incremental value vs fixed rules

## Architecture

```
SOURCE DATASETS (3 evidence layers)
        │
        v
Feature Engineering (87 features)
        │
        v
Customer / Transaction Context
        │
        v
Recovery Propensity Model (HistGradientBoosting)
        │
        v
P(recovery | action, context) for each action
        │
        v
Counterfactual Engine → Expected Monetary Value
        │
        v
Safety / Policy Gate (deterministic, non-overridable)
        │
        v
RETRY | PAYMENT_LINK | REMINDER | NO_ACTION | ESCALATE
        │
        v
Executor (Razorpay Live or Simulator)
        │
        v
Counterfactual Ledger (audit trail)
        │
        v
Evaluation (business-value metrics)
```

## Key Distinction: What This Is vs What This Isn't

| This System | NOT This |
|---|---|
| Counterfactual decision engine | Generic fraud classifier |
| Expected monetary value optimization | Probability-only ranking |
| Safety-constrained AI | Unconstrained ML predictions |
| Honest about synthetic labels | Claiming causal effects from observational data |

### Important Terminology

- **Synthetic Counterfactual Labels**: The benchmark dataset's outcome labels are generated from a synthetic environment. They enable reproducible evaluation but do NOT represent observed causal effects.
- **Recovery Propensity Model**: Estimates P(recovery | action, context) — a prediction, not a causal claim.
- **Benchmark Environment** (`ground_truth.py`): The canonical outcome simulator for offline evaluation. Deliberately different from the agent's model.

## Datasets

Three independently-sampled evidence layers (10,000 benchmark cases):

1. **E-commerce Customer Behavior** — session duration, pages viewed, device, payment method, demographics
2. **Large Sales Transaction History** — customer-level aggregates: lifetime spend, frequency, recency, value segment
3. **Fraud/Risk Detection** — risk flags, failed transactions, velocity, unusual patterns

**Data Linkage**: No fabricated cross-dataset joins. The three layers are independently sampled and clearly documented as such.

## Leakage Prevention

Strict separation:
- `AGENT_INPUT_COLUMNS` — 42 features the model is allowed to see
- `EVALUATION_COLUMNS` — 19 target/outcome columns that NEVER enter the model
- Automated leakage tests in `tests/test_engine.py`

## Results

### Model Performance (Logistic Bootstrap → HistGBM on local run)

| Split | AUC | Accuracy | F1 |
|-------|-----|----------|-----|
| Train | — | 0.62 | 0.67 |
| Val | — | 0.62 | 0.67 |
| Test | — | 0.59 | 0.65 |

*Note: Bootstrap logistic model. Run `python run_pipeline.py` with sklearn for HistGradientBoosting (~0.75+ AUC expected).*

### Benchmark: Agent vs Baseline (1,500 test cases)

| Metric | Fixed Rules | CF Agent |
|--------|-------------|----------|
| Revenue recovered | ₹835,674 | ₹928,803 |
| Recovery rate | 58.3% | 64.8% |
| **Incremental revenue** | — | **₹93,129** |
| **Uplift** | — | **+11.1%** |
| Intervention rate | 100% | 98.1% |
| Escalation rate | 68.8% | 54.8% |
| No-action rate | 0% | 1.9% |
| Policy violations | — | **0** |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (train + evaluate)
python run_pipeline.py

# 3. Start the API
uvicorn backend.main:app --reload --port 8000

# 4. Open the dashboard
open frontend/dashboard.html
```

## Project Structure

```
rce/
├── backend/
│   ├── features/
│   │   ├── columns.py              # Leakage prevention config
│   │   └── feature_engineering.py   # 87-feature pipeline
│   ├── models_ml/
│   │   ├── recovery_model.py        # RecoveryModel interface + implementations
│   │   ├── train.py                 # Training pipeline
│   │   ├── config.py                # Intervention costs, risk penalties
│   │   └── artifacts/               # Trained model weights
│   ├── counterfactual_engine.py     # Core decisioning (calls ML model)
│   ├── ground_truth.py              # Synthetic benchmark environment
│   ├── policy_engine.py             # Deterministic safety gate
│   ├── risk_engine.py               # Risk scoring
│   ├── context.py                   # Customer intent reconstruction
│   ├── executor.py                  # Razorpay Live + Simulator
│   ├── ledger.py                    # Audit trail
│   ├── evaluation.py                # Business-value benchmark
│   ├── baseline.py                  # Fixed-rule comparator
│   ├── dataset_generator.py         # Legacy synthetic dataset
│   ├── models.py                    # Pydantic data models
│   └── main.py                      # FastAPI application
├── frontend/
│   └── dashboard.html               # Decision trace UI
├── data/
│   ├── revenue_recovery_benchmark.csv  # 10,000-case benchmark
│   └── benchmark.json               # Results
├── tests/
│   └── test_engine.py               # 25+ tests
├── run_pipeline.py                  # One-command runner
└── requirements.txt
```

## API

```
POST /api/decide    → Submit event, get structured decision
POST /api/execute   → Decide + execute (idempotent)
GET  /api/dashboard → Summary metrics
GET  /api/benchmark → Full results
GET  /api/segments  → Breakdown by segment
GET  /api/health    → Model status
```

Example response from `/api/decide`:
```json
{
  "case_id": "EVT_A1B2C3D4E5",
  "recommended_action": "RETRY_PAYMENT",
  "expected_recovery": 6450.00,
  "confidence": 0.81,
  "candidate_actions": [
    {"action": "RETRY_PAYMENT", "probability": 0.72, "expected_recovery": 6450},
    {"action": "CREATE_PAYMENT_LINK", "probability": 0.48, "expected_recovery": 4200},
    {"action": "SEND_REMINDER", "probability": 0.28, "expected_recovery": 2450}
  ],
  "safety_constraints": ["all checks passed"],
  "risk_tier": "low",
  "reasoning": ["Root cause: transient_bank_failure", "Customer intent: high"]
}
```

## Configuration

All tunable parameters in `backend/models_ml/config.py`:

```python
InterventionCosts:
  RETRY_PAYMENT:        ₹5
  CREATE_PAYMENT_LINK:  ₹8
  SEND_REMINDER:        ₹2
  ESCALATE_TO_HUMAN:    ₹50
  NO_ACTION:            ₹0

RiskPenalties:
  high:   15% of transaction amount
  medium:  5% of transaction amount
  low:     0%

min_intervention_threshold: ₹10  (below this, prefer NO_ACTION)
```

## Safety

The AI model **cannot override** the deterministic policy gate:
- Max retry count exceeded → ESCALATE
- Amount > ₹50,000 → requires human approval
- Confidence below threshold → ESCALATE
- Previous fraud detected → ESCALATE
- Recovery eligibility = NONE → BLOCK

## Reproducibility

- Random seed: 42 (all stochastic processes)
- Train/Val/Test: 70/15/15 stratified split
- Benchmark dataset: `data/revenue_recovery_benchmark.csv` (10,000 cases)
- Model artifacts: `backend/models_ml/artifacts/`

## Limitations

1. **Bootstrap model**: The shipped numpy logistic model is a bootstrap. Run `python run_pipeline.py` for HistGradientBoosting.
2. **Synthetic counterfactual labels**: Benchmark outcomes are simulated, not observed real-world recovery rates.
3. **No cross-dataset customer linkage**: The three source datasets are independently sampled — no fabricated joins.
4. **Razorpay live mode**: Not tested against real `rzp_test_` credentials (architecture supports it).
5. **Causal claims**: The model estimates correlational recovery probabilities, not causal effects.
# Revenue-Counterfactual-Engine
