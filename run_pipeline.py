#!/usr/bin/env python3
"""
One-command pipeline runner for the Data-Driven Revenue Counterfactual Engine.

    python run_pipeline.py

Steps:
  1. Copies the benchmark dataset to data/ if not already there
  2. Trains the recovery model (HistGradientBoosting) on 70% train split
  3. Evaluates baseline vs. counterfactual agent on 15% held-out test set
  4. Saves data/benchmark.json with full results
  5. Prints the comparison table

Then:
  - Open frontend/dashboard.html in a browser (works offline with bundled data)
  - Or: uvicorn backend.main:app --reload (live API for dashboard)
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BENCHMARK_CSV = os.path.join(DATA_DIR, "revenue_recovery_benchmark.csv")


def main():
    print("Revenue Counterfactual Engine — Data-Driven Pipeline")
    print("=" * 60)
    print()

    # Step 0: Ensure benchmark CSV exists in data/
    if not os.path.exists(BENCHMARK_CSV):
        # Check common locations
        candidates = [
            os.path.join(os.path.dirname(__file__), "revenue_recovery_benchmark.csv"),
            os.path.join(os.path.dirname(__file__), "attached_files", "revenue_recovery_benchmark.csv"),
        ]
        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break
        if found:
            os.makedirs(DATA_DIR, exist_ok=True)
            shutil.copy2(found, BENCHMARK_CSV)
            print(f"[0/4] Copied benchmark CSV to {BENCHMARK_CSV}")
        else:
            print(f"ERROR: Benchmark CSV not found at {BENCHMARK_CSV}")
            print("       Place revenue_recovery_benchmark.csv in the data/ directory.")
            sys.exit(1)
    else:
        print(f"[0/4] Benchmark CSV found: {BENCHMARK_CSV}")

    # Step 1: Train the model
    print()
    from backend.models_ml.train import train_model
    report = train_model(BENCHMARK_CSV)

    # Step 2: Run the data-driven benchmark
    print()
    print("=" * 60)
    print("Running Data-Driven Benchmark (Agent vs Baseline)")
    print("=" * 60)
    from backend.evaluation import run_data_driven_benchmark
    bench = run_data_driven_benchmark(BENCHMARK_CSV)

    # Save benchmark
    os.makedirs(DATA_DIR, exist_ok=True)
    bench_path = os.path.join(DATA_DIR, "benchmark.json")
    with open(bench_path, "w") as f:
        json.dump(bench, f, indent=2)
    print(f"\n[✓] Benchmark saved to {bench_path}")

    # Step 3: Print comparison table
    print()
    print("=" * 60)
    print("BENCHMARK RESULTS: Fixed Rules vs Counterfactual Agent")
    print("=" * 60)
    print()

    sizes = bench["dataset_sizes"]
    print(f"Dataset: {sizes['total']:,} total | {sizes['train']:,} train | "
          f"{sizes['validation']:,} val | {sizes['test']:,} test")
    print()

    b, a = bench["baseline"], bench["agent"]
    header = f"{'Metric':<32}{'Fixed Rules':>18}{'CF Agent':>18}"
    print(header)
    print("-" * len(header))
    rows = [
        ("Revenue at risk", f"₹{b['revenue_at_risk']:,.0f}", f"₹{a['revenue_at_risk']:,.0f}"),
        ("Revenue recovered", f"₹{b['revenue_recovered']:,.0f}", f"₹{a['revenue_recovered']:,.0f}"),
        ("Recovery rate", f"{b['recovery_rate']:.1f}%", f"{a['recovery_rate']:.1f}%"),
        ("Incremental revenue", "—", f"₹{a.get('incremental_revenue', 0):,.0f}"),
        ("Uplift %", "—", f"{a.get('uplift_pct', 0):.1f}%"),
        ("Intervention rate", f"{b.get('intervention_rate', 100):.1f}%", f"{a.get('intervention_rate', 0):.1f}%"),
        ("Unnecessary interventions", str(b.get('unnecessary_interventions', 0)), str(a.get('unnecessary_interventions', 0))),
        ("Escalation rate", f"{b['escalation_rate']:.1f}%", f"{a['escalation_rate']:.1f}%"),
        ("No-action rate", f"{b.get('no_action_rate', 0):.1f}%", f"{a.get('no_action_rate', 0):.1f}%"),
        ("Policy violations", str(b['policy_violations']), str(a['policy_violations'])),
        ("Avg recovered/case", f"₹{b.get('avg_recovered_per_case', 0):,.0f}", f"₹{a.get('avg_recovered_per_case', 0):,.0f}"),
    ]
    for label, bv, av in rows:
        print(f"{label:<32}{bv:>18}{av:>18}")

    # Model performance summary
    print()
    print("=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    test_metrics = report["metrics"]["test"]
    print(f"  AUC-ROC:   {test_metrics['auc_roc']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")

    print()
    print("Next steps:")
    print("  - Open frontend/dashboard.html in a browser")
    print("  - Or: uvicorn backend.main:app --reload --port 8000")
    print()


if __name__ == "__main__":
    main()
