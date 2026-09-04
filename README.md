# Revenue Counterfactual Engine

**Track 03 — AI Revenue Recovery** (Razorpay Buildathon)

> When revenue fails, the best action is **not always retry**.

---

## What It Does

When a payment fails on Razorpay, this engine **doesn't blindly retry**. It evaluates every possible recovery action, predicts the expected monetary value of each, and picks the one that maximizes revenue — all within safety constraints.

```
Payment fails on Razorpay
        │
   ① WEBHOOK IN ─── Razorpay sends payment.failed to your server
        │
   ② DIAGNOSE ───── Root cause + customer context (87 features)
        │
   ③ EVALUATE ───── ML model predicts ₹ value of each action
        │
   ④ SAFETY GATE ── Deterministic policy check (AI cannot override)
        │
   ⑤ EXECUTE ────── Razorpay API: retry / payment link / remind / escalate
        │
   ⑥ RECONCILE ──── Razorpay webhooks back with the result

```

### Recovery Actions

| Action | What Happens | When It's Best |
| --- | --- | --- |
| `RETRY_PAYMENT` | `razorpay.order.create()` — new order for same amount | Bank timeouts, transient failures |
| `CREATE_PAYMENT_LINK` | `razorpay.payment_link.create()` — SMS/email link to customer | Expired cards, method failures |
| `SEND_REMINDER` | SMS/email/WhatsApp notification | Abandoned checkouts, overdue invoices |
| `ESCALATE_TO_HUMAN` | Route to support team | High-risk, fraud flags, high-value |

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### 1. Install Dependencies

```bash
cd rce
pip install -r requirements.txt

```

### 2. Train Model & Run Benchmark

```bash
python run_pipeline.py

```

This will:

- Train the HistGradientBoosting recovery model on 70% of the 10,000-case benchmark dataset
- Evaluate agent vs fixed-rule baseline on the 15% held-out test set
- Save results to `data/benchmark.json`
- Print the comparison table

### 3. Start the API Server

```bash
uvicorn backend.main:app --reload --port 8000

```

### 4. Open the Dashboard

Open `frontend/dashboard.html` in your browser.

> **Important:** The dashboard fetches live data from `http://localhost:8000`. The backend server (step 3) must be running, otherwise API calls like `/api/decide` will fail with a 405 error.

---

## Razorpay Integration (3 Steps)

To connect to real Razorpay (test mode):

### Step 1 — Get Test Keys

1. Go to [Razorpay Dashboard](https://dashboard.razorpay.com) → **Test Mode**
2. **Settings → API Keys → Generate Key**
3. Copy `Key ID` (`rzp_test_...`) and `Key Secret`

### Step 2 — Set Environment Variables

```bash
export RAZORPAY_KEY_ID=rzp_test_xxxxxxxxxxxx
export RAZORPAY_KEY_SECRET=your_key_secret_here
export RAZORPAY_WEBHOOK_SECRET=your_webhook_secret   # optional

```

### Step 3 — Configure Webhook on Razorpay

1. Razorpay Dashboard → **Settings → Webhooks → Add New Webhook**
2. **URL:** `https://your-server.com/api/webhook/razorpay`
3. **Events:** `payment.failed`, `payment.captured`, `payment_link.paid`

> Without env vars, the engine auto-falls back to `SimulatedExecutor` (offline benchmark mode). With env vars set, it auto-switches to `RazorpayLiveExecutor`.

---

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Model status check |
| `GET` | `/api/dashboard` | Summary metrics for dashboard cards |
| `GET` | `/api/benchmark` | Full agent vs baseline benchmark results |
| `GET` | `/api/segments` | Breakdown by failure reason, risk tier, etc. |
| `GET` | `/api/ledger?system=agent&limit=25` | Paginated decision ledger |
| `GET` | `/api/decision/{case_id}` | Single-case decision trace |
| `GET` | `/api/live-ledger` | Real-time ledger of live executions |
| `POST` | `/api/decide` | Submit event JSON → get structured decision |
| `POST` | `/api/execute` | Decide + execute through live/simulated executor |
| `POST` | `/api/razorpay/recover` | Accepts Razorpay `payment.failed` webhook format |
| `POST` | `/api/webhook/razorpay` | Webhook receiver for Razorpay reconciliation |

### Example: `/api/decide`

**Request:**

```json
{
  "transaction_id": "ORD_DEMO_001",
  "amount": 8950,
  "event_type": "payment_failed",
  "failure_reason": "transient_bank_failure",
  "payment_method": "card",
  "customer_previous_payments": 8,
  "customer_previous_failures": 1,
  "days_since_last_purchase": 5,
  "retry_count": 0,
  "customer_value": 42000
}

```

**Response:**

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

---

## How the Engine Works

### Decision Formula

For each possible action:

```
expected_recovery = amount × P(success | action, context) − intervention_cost − risk_penalty

```

The agent picks the action with the **highest expected recovery**.

### ML Model

- **Algorithm:** HistGradientBoosting (scikit-learn)
- **Input:** 42 features (customer behavior, transaction history, risk signals)
- **Output:** P(recovery | action, context) per action
- **Training:** 70/15/15 train/val/test split on 10,000 benchmark cases

### Safety Gate (Non-Overridable)

The ML model's recommendation passes through a **deterministic policy gate** that the AI cannot bypass:

| Condition | Action |
| --- | --- |
| Previous fraud detected | → ESCALATE |
| Risk signal count ≥ 5 | → ESCALATE |
| Amount > ₹50,000 | → Requires human approval |
| Max retry count exceeded | → ESCALATE |
| Confidence below threshold | → ESCALATE |
| Recovery eligibility = NONE | → BLOCK |

**Policy violations: always 0.** The safety gate is deterministic and sits outside the model.

### Leakage Prevention

Strict column separation enforced in `backend/features/columns.py`:

- `AGENT_INPUT_COLUMNS` — 42 features the model is allowed to see
- `EVALUATION_COLUMNS` — 19 target/outcome columns that NEVER enter the model
- Automated leakage tests in `tests/test_engine.py`

---

## Configuration

All tunable parameters in `backend/models_ml/config.py`:

```python
InterventionCosts:
  RETRY_PAYMENT:        ₹5
  CREATE_PAYMENT_LINK:  ₹8
  SEND_REMINDER:        ₹2
  ESCALATE_TO_HUMAN:    ₹50

RiskPenalties:
  high:    4% of transaction amount
  medium:  1% of transaction amount
  low:     0%

```

---

## Project Structure

```
rce/
├── backend/
│   ├── features/
│   │   ├── columns.py                # Leakage prevention config
│   │   └── feature_engineering.py    # 87-feature pipeline
│   ├── models_ml/
│   │   ├── recovery_model.py         # RecoveryModel interface + implementations
│   │   ├── train.py                  # Training pipeline
│   │   ├── config.py                 # Intervention costs, risk penalties
│   │   └── artifacts/                # Trained model weights
│   ├── counterfactual_engine.py      # Core decisioning (calls ML model)
│   ├── ground_truth.py               # Synthetic benchmark environment
│   ├── policy_engine.py              # Deterministic safety gate
│   ├── risk_engine.py                # Risk scoring
│   ├── context.py                    # Customer intent reconstruction
│   ├── executor.py                   # Razorpay Live + Simulator
│   ├── ledger.py                     # Audit trail
│   ├── evaluation.py                 # Business-value benchmark
│   ├── baseline.py                   # Fixed-rule comparator
│   ├── dataset_generator.py          # Legacy synthetic dataset
│   ├── models.py                     # Pydantic data models
│   └── main.py                       # FastAPI application
├── frontend/
│   └── dashboard.html                # Decision trace UI
├── data/
│   ├── revenue_recovery_benchmark.csv  # 10,000-case benchmark
│   └── benchmark.json                # Evaluation results
├── tests/
│   └── test_engine.py                # 25+ unit tests
├── run_pipeline.py                   # One-command runner
├── requirements.txt                  # Python dependencies
└── pyproject.toml                    # Project metadata

```

---

## Running Tests

```bash
python -m pytest tests/ -v

```

---

## Datasets

Three independently-sampled evidence layers (10,000 benchmark cases):

1. **E-commerce Customer Behavior** — session duration, pages viewed, device, payment method, demographics
2. **Large Sales Transaction History** — customer-level aggregates: lifetime spend, frequency, recency, value segment
3. **Fraud/Risk Detection** — risk flags, failed transactions, velocity, unusual patterns

---

## Reproducibility

- Random seed: `42` (all stochastic processes)
- Train/Val/Test: 70/15/15 split
- Benchmark dataset: `data/revenue_recovery_benchmark.csv` (10,000 cases)
- Model artifacts: `backend/models_ml/artifacts/`

---

## Limitations

1. **Synthetic counterfactual labels** — Benchmark outcomes are simulated, not observed real-world recovery rates
2. **No cross-dataset customer linkage** — The three source datasets are independently sampled
3. **Razorpay live mode** — Architecture supports it but not tested against real `rzp_test_` credentials
4. **Causal claims** — The model estimates correlational recovery probabilities, not causal effects

---

## Tech Stack

| Component | Technology |
| --- | --- |
| Backend | FastAPI + Python |
| ML Model | scikit-learn HistGradientBoosting |
| Data | pandas + numpy |
| Payment API | Razorpay Python SDK |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Vercel (FastAPI entrypoint) |

