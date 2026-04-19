#!/usr/bin/env python3
"""
compare_models.py
------------------
Master evaluation runner for DoctorG.
Runs all evaluation scripts and prints a consolidated comparison table.

Usage:
    python compare_models.py
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

try:
    from tabulate import tabulate
except ImportError:
    print("Install tabulate: pip install tabulate")
    sys.exit(1)

METRICS_DIR = Path(__file__).parent
RESULTS_DIR = METRICS_DIR / "results"


def run_script(script_name: str) -> dict:
    """Run a metrics script and capture key output values."""
    script_path = METRICS_DIR / script_name
    print(f"\n{'─' * 60}")
    print(f"  Running: {script_name}")
    print(f"{'─' * 60}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,
        text=True
    )
    return {"script": script_name, "exit_code": result.returncode}


def collect_latest_results() -> dict:
    """Read the most recent CSV result file from each evaluation."""
    summary = {}
    prefixes = {
        "classification": "Triage Classification",
        "generation": "Text Generation",
        "biomarker": "Biomarker Extraction",
        "rag": "RAG Pipeline",
    }

    for prefix, label in prefixes.items():
        files = sorted(RESULTS_DIR.glob(f"{prefix}_*.csv"), reverse=True)
        if not files:
            summary[label] = {"status": "No results yet"}
            continue

        import csv
        with open(files[0]) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            summary[label] = {"status": "Empty result"}
            continue

        # Compute averages for numeric columns
        numeric_keys = [k for k in rows[0] if k != "id"]
        avgs = {}
        for k in numeric_keys:
            try:
                vals = [float(r[k]) for r in rows if r[k] not in ("", None)]
                avgs[k] = f"{sum(vals) / len(vals):.4f}" if vals else "N/A"
            except ValueError:
                avgs[k] = rows[0][k]  # Non-numeric, just take first value

        summary[label] = avgs

    return summary


def print_comparison_table(summary: dict):
    print("\n")
    print("╔" + "═" * 70 + "╗")
    print("║" + "  DOCTORG — FULL EVALUATION COMPARISON TABLE".center(70) + "║")
    print("║" + f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(70) + "║")
    print("╚" + "═" * 70 + "╝")

    for eval_name, metrics in summary.items():
        print(f"\n  📊 {eval_name}")
        print("  " + "─" * 50)

        if "status" in metrics:
            print(f"  {metrics['status']}")
            continue

        rows = [[k, v] for k, v in metrics.items()]
        print(tabulate(rows, headers=["Metric", "Score"], tablefmt="simple",
                       colalign=("left", "right")))

    print("\n" + "═" * 72)
    print("  ✅ All evaluations complete. Results saved to metrics/results/")
    print("=" * 72)


def save_comparison_report(summary: dict):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"comparison_report_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "summary": summary}, f, indent=2)
    print(f"\n  Full report saved to: {out_path}")


def main():
    print("\n")
    print("╔" + "═" * 60 + "╗")
    print("║" + "  DOCTORG METRICS SUITE — FULL EVALUATION RUN".center(60) + "║")
    print("╚" + "═" * 60 + "╝")

    scripts = [
        "evaluate_classification.py",
        "evaluate_generation.py",
        "evaluate_biomarker.py",
        "evaluate_rag.py",
    ]

    for script in scripts:
        run_script(script)

    print("\n\n" + "═" * 60)
    print("  COLLECTING RESULTS...")
    print("═" * 60)

    summary = collect_latest_results()
    print_comparison_table(summary)
    save_comparison_report(summary)


if __name__ == "__main__":
    main()
