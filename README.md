# Revenue Counterfactual Engine

Track 03 - AI Revenue Recovery (Razorpay Buildathon)

When revenue fails, the best action is not always retry.

## What It Does

When a payment fails on Razorpay, this engine doesn't blindly retry. It looks at who the customer is, why the payment failed, and what each recovery option is actually worth in rupees. Then it picks the best one.

## Architecture

```
                         RAZORPAY
                            |
                    payment.failed webhook
                            |
                            v
              +----------------------------+
              |   Webhook Receiver         |
              |   main.py                  |
              |   HMAC signature verify    |
              +----------------------------+
                            |
                            v
  +------------------+    +----------------------------+    +------------------+
  | Customer Behavior|    |   Context Builder          |    | Risk Engine      |
  | Transaction Hist.|===>|   context.py               |<===| risk_engine.py   |
  | Fraud/Risk Data  |    |   87 features engineered   |    | RevenueAutopsy   |
  +------------------+    +----------------------------+    +------------------+
                            |
                            v
              +----------------------------+
              |   Recovery Model           |
              |   HistGradientBoosting     |
              |   recovery_model.py        |
              |                            |
              |   outputs per action:      |
              |   P(recovery | action,ctx) |
              +----------------------------+
                            |
                            v
              +----------------------------+
              |   Counterfactual Engine    |
              |   counterfactual_engine.py |
              |                            |
              |   EMV = Amt x P - Cost     |
              |         - Risk Penalty     |
              |                            |
              |   picks highest EMV action |
              +----------------------------+
                            |
                            v
              +----------------------------+
              |   Safety Gate              |     The AI cannot
              |   policy_engine.py         |     override this.
              |                            |     Policy violations: 0
              |   fraud? -> ESCALATE       |     (guaranteed)
              |   >50K?  -> HUMAN APPROVAL |
              |   retries maxed? -> ESCAL. |
              +----------------------------+
                            |
                            v
  +------------------+    +----------------------------+    +------------------+
  | SimulatedExecutor|    |   Executor                 |    |  RAZORPAY API    |
  | (offline bench.) |<---|   executor.py              |--->|  order.create()  |
  +------------------+    |   auto-selects backend     |    |  payment_link()  |
                           +----------------------------+    +------------------+
                            |
                            v
  +------------------+    +----------------------------+    +------------------+
  | Baseline Compare |    |   Evaluation Harness       |    | Counterfactual   |
  | baseline.py      |<---|   evaluation.py            |--->| Ledger           |
  | retry->retry->esc|    |   agent vs baseline        |    | ledger.py        |
  +------------------+    +----------------------------+    +------------------+
                            |
                            v
              +----------------------------+
              |   FastAPI Server           |
              |   main.py - 11 endpoints   |
              |                            |
              |   /api/decide   (POST)     |
              |   /api/execute  (POST)     |
              |   /api/dashboard (GET)     |
              |   /api/benchmark (GET)     |
              +----------------------------+
                            |
                            v
              +----------------------------+
              |   Dashboard                |
              |   frontend/dashboard.html  |
              |   live metrics from API    |
              +----------------------------+

```

Every box maps to an actual file in the project.

The four recovery actions it chooses from:

- **Retry Payment** - creates a new Razorpay order for the same amount. Best for bank timeouts and transient failures.
- **Create Payment Link** - sends an SMS/email payment link to the customer. Best for expired cards and method failures.
- **Send Reminder** - SMS, email, or WhatsApp notification. Best for abandoned checkouts and overdue invoices.
- **Escalate to Human** - routes to the support team. Best for high-risk cases, fraud flags, and high-value transactions.

## Quick Start

You need Python 3.10+ and pip.

```bash
# install dependencies
cd rce
pip install -r requirements.txt

# train the model and run the benchmark
python run_pipeline.py

# start the API server
uvicorn backend.main:app --reload --port 8000

```

Then open `frontend/dashboard.html` in your browser.

The dashboard fetches live data from `http://localhost:8000`, so the backend server needs to be running. If you see a 405 error on the "Agent Decision" panel, it means the server isn't up yet.

`run_pipeline.py` does three things: trains the HistGradientBoosting model on 70% of the 10,000-case dataset, evaluates the agent against the fixed-rule baseline on 15% held-out test cases, and saves the results to `data/benchmark.json`.

## Razorpay Integration

To connect to real Razorpay in test mode:

**1. Get test keys** from the Razorpay Dashboard (Settings > API Keys > Generate Key). Make sure you're in Test Mode. You'll get a Key ID starting with `rzp_test_` and a Key Secret.

**2. Set environment variables:**

```bash
export RAZORPAY_KEY_ID=rzp_test_xxxxxxxxxxxx
export RAZORPAY_KEY_SECRET=your_key_secret
export RAZORPAY_WEBHOOK_SECRET=your_webhook_secret  # optional, for signature verification

```

**3. Configure the webhook** in Razorpay Dashboard > Settings > Webhooks > Add New Webhook. Set the URL to `https://your-server.com/api/webhook/razorpay` and enable `payment.failed`, `payment.captured`, and `payment_link.paid` events.

Without the environment variables, the engine falls back to `SimulatedExecutor` for offline benchmarking. With them set, it automatically uses `RazorpayLiveExecutor` and makes real API calls against Razorpay's test sandbox.

## API

The server exposes 11 endpoints:

| Method | Endpoint | What it does |
| --- | --- | --- |
| GET | /api/health | Check if the model is loaded |
| GET | /api/dashboard | Summary metrics for the dashboard |
| GET | /api/benchmark | Full agent vs baseline results |
| GET | /api/segments | Breakdown by failure reason, risk tier, etc. |
| GET | /api/ledger | Paginated decision ledger (pass `system=agent` or `baseline`, `limit=25`) |
| GET | /api/decision/{case_id} | Full decision trace for a single case |
| GET | /api/live-ledger | Real-time ledger of live /api/execute calls |
| POST | /api/decide | Submit an event, get back a decision |
| POST | /api/execute | Decide and execute (creates Razorpay orders/links) |
| POST | /api/razorpay/recover | Accepts Razorpay's native payment.failed payload format |
| POST | /api/webhook/razorpay | Receives Razorpay reconciliation webhooks |

Example request to `/api/decide`:

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

Example response:

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

## How the Engine Works

For each possible action, the engine computes:

```
expected_recovery = amount x P(success | action, context) - intervention_cost - risk_penalty

```

It picks the action with the highest expected recovery.

**The ML model** is a HistGradientBoosting classifier from scikit-learn. It takes 42 features (customer behavior, transaction history, risk signals) and outputs P(recovery | action, context) for each action. It's trained on a 70/15/15 split of the 10,000-case benchmark dataset.

**The safety gate** is a deterministic policy check that sits outside the model. The AI cannot bypass it. The rules are simple:

- Previous fraud detected: escalate
- Risk signal count >= 5: escalate
- Amount over 50,000 INR: requires human approval
- Max retry count exceeded: escalate
- Model confidence below threshold: escalate
- Recovery eligibility is NONE: block entirely

Policy violations in the benchmark: zero. This is an architectural guarantee, not an aspiration.

**Leakage prevention** is enforced through strict column separation in `backend/features/columns.py`. The model sees 42 input features. The 19 evaluation/outcome columns never enter the model. This is tested automatically.

## Results

Benchmark output from the current pipeline run, evaluated on 1,500 held-out test cases:

| Metric | Fixed Rules | Agent |
| --- | --- | --- |
| Revenue recovered | 835,674 | 928,803 |
| Recovery rate | 58.3% | 64.8% |
| Incremental revenue | - | +93,129 |
| Uplift | - | +11.1% |
| Intervention rate | 100% | 98.1% |
| Escalation rate | 68.8% | 54.8% |
| No-action rate | 0% | 1.9% |
| Policy violations | - | 0 |

These numbers come from running `python run_pipeline.py` with the current dataset, model, and configuration. They are not hard-coded. If you change the dataset, retrain the model, or adjust the configuration, the numbers will change. The dashboard displays whatever the current evaluation produces.

This is a benchmark on synthetic counterfactual data, not a production performance guarantee.

## Dashboard

The dashboard at `frontend/dashboard.html` pulls metrics live from the backend API. Everything it displays - revenue at risk, agent recovered, baseline recovered, incremental revenue, uplift, recovery rate, escalation rate, policy violations - is dynamically calculated from the current evaluation run.

If you retrain the model or change the config, restart the server and refresh the dashboard. The numbers update automatically.

## Configuration

All tunable parameters live in `backend/models_ml/config.py`:

```
Intervention costs:
  Retry Payment:        5 INR
  Create Payment Link:  8 INR
  Send Reminder:        2 INR
  Escalate to Human:    50 INR

Risk penalties:
  High risk:   4% of transaction amount
  Medium risk: 1% of transaction amount
  Low risk:    0%

```

## Project Structure

```
rce/
  backend/
    features/
      columns.py                 - leakage prevention config
      feature_engineering.py     - 87-feature pipeline
    models_ml/
      recovery_model.py          - model interface and implementations
      train.py                   - training pipeline
      config.py                  - costs, penalties, thresholds
      artifacts/                 - saved model weights
    counterfactual_engine.py     - core decision engine
    ground_truth.py              - synthetic benchmark environment
    policy_engine.py             - deterministic safety gate
    risk_engine.py               - risk scoring
    context.py                   - customer intent reconstruction
    executor.py                  - Razorpay live executor + simulator
    ledger.py                    - audit trail
    evaluation.py                - benchmark evaluation harness
    baseline.py                  - fixed-rule comparator
    dataset_generator.py         - legacy synthetic dataset generator
    models.py                    - data models
    main.py                      - FastAPI server
  frontend/
    dashboard.html               - decision trace dashboard
  data/
    revenue_recovery_benchmark.csv  - 10,000-case benchmark dataset
    benchmark.json               - evaluation results
  tests/
    test_engine.py               - 25+ unit tests
  run_pipeline.py                - one-command pipeline runner
  requirements.txt
  pyproject.toml

```

## Tests

```bash
python -m pytest tests/ -v

```

## Datasets

The benchmark uses 10,000 cases built from three independently sampled evidence layers:

1. E-commerce customer behavior (sessions, device, pages, payment method, demographics)
2. Sales transaction history (lifetime spend, frequency, recency, value segment)
3. Fraud and risk signals (risk flags, failed transactions, velocity, unusual patterns)

These are independently sampled. There are no fabricated cross-dataset joins.

## Reproducibility

The benchmark is reproducible when you keep the same:

- Dataset (`data/revenue_recovery_benchmark.csv`)
- Random seed (42, used everywhere)
- Train/val/test split (70/15/15)
- Model implementation and saved weights (`backend/models_ml/`)
- Configuration (`backend/models_ml/config.py`)
- Evaluation logic (`backend/evaluation.py`)

Change any of these and the numbers will change.

## Limitations

1. **Synthetic data.** The benchmark outcomes come from a synthetic counterfactual environment. They allow reproducible evaluation but don't represent observed real-world recovery rates or causal effects.
2. **Benchmark is not production.** The metrics show how the system behaves under controlled conditions. They should not be read as guaranteed production performance or guaranteed uplift in a live environment.
3. **Prediction, not causation.** The model estimates the probability of recovery given an action and context. It does not establish that the action caused the recovery.
4. **Independent datasets.** The three source datasets are independently sampled. There are no fabricated cross-dataset customer links.
5. **Razorpay live mode untested.** The architecture supports live Razorpay API calls, but this hasn't been tested against real `rzp_test_` credentials.
6. **Not an LLM.** The model is a HistGradientBoosting classifier from scikit-learn. It predicts recovery probabilities from structured features. It is not a large language model.

## Tech Stack

- Backend: FastAPI + Python
- ML: scikit-learn (HistGradientBoosting)
- Data: pandas, numpy
- Payments: Razorpay Python SDK
- Frontend: HTML, CSS, JS (no framework)
- Deployment: Vercel

