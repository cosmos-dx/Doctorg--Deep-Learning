#!/usr/bin/env python3
"""
evaluate_biomarker.py
-----------------------
Evaluates accuracy of biomarker extraction from medical reports.

Metrics:
- Exact Match Rate: name + value both match
- Name Match Rate: biomarker name identified correctly
- Value Accuracy: numeric value within ±5% of ground truth
- False Positive Rate: extra biomarkers hallucinated
- False Negative Rate: missed biomarkers

Usage:
    python evaluate_biomarker.py
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

SAMPLE_DIR = Path(__file__).parent / "sample_data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Mock extractor (simulates GPT-4o output) ─────────────────────────────────
MOCK_EXTRACTIONS = {
    "bm_001": [
        {"name": "Hemoglobin", "value": 13.5, "unit": "g/dL", "status": "normal"},
        {"name": "WBC", "value": 7.2, "unit": "x10³/µL", "status": "normal"},
        {"name": "Platelets", "value": 250.0, "unit": "x10³/µL", "status": "normal"},
        {"name": "RBC", "value": 4.8, "unit": "million/µL", "status": "normal"},
        {"name": "Hematocrit", "value": 41.0, "unit": "%", "status": "normal"},
        {"name": "MCV", "value": 85.0, "unit": "fL", "status": "normal"},
    ],
    "bm_002": [
        {"name": "Total Cholesterol", "value": 245.0, "unit": "mg/dL", "status": "high"},
        {"name": "LDL Cholesterol", "value": 165.0, "unit": "mg/dL", "status": "high"},
        {"name": "HDL Cholesterol", "value": 38.0, "unit": "mg/dL", "status": "low"},
        {"name": "Triglycerides", "value": 210.0, "unit": "mg/dL", "status": "high"},
    ],
    "bm_003": [
        {"name": "TSH", "value": 0.8, "unit": "mIU/L", "status": "normal"},
        {"name": "Free T4", "value": 1.2, "unit": "ng/dL", "status": "normal"},
        {"name": "Free T3", "value": 3.1, "unit": "pg/mL", "status": "normal"},
    ],
}


def normalize_name(name: str) -> str:
    return name.lower().strip().replace("-", " ").replace("_", " ")


def value_within_tolerance(pred_val, true_val, tolerance=0.05) -> bool:
    if pred_val is None or true_val is None:
        return False
    if true_val == 0:
        return pred_val == 0
    return abs(pred_val - true_val) / abs(true_val) <= tolerance


def evaluate_report(expected: list, predicted: list) -> dict:
    """Per-report evaluation metrics."""
    expected_names = {normalize_name(b["name"]) for b in expected}
    predicted_names = {normalize_name(b["name"]) for b in predicted}

    # Name-level metrics
    true_pos_names = expected_names & predicted_names
    false_pos_names = predicted_names - expected_names
    false_neg_names = expected_names - predicted_names

    name_precision = len(true_pos_names) / len(predicted_names) if predicted_names else 0
    name_recall = len(true_pos_names) / len(expected_names) if expected_names else 0
    name_f1 = (
        2 * name_precision * name_recall / (name_precision + name_recall)
        if (name_precision + name_recall) > 0 else 0
    )

    # Exact match (name + value within tolerance)
    exact_matches = 0
    for exp in expected:
        exp_name = normalize_name(exp["name"])
        for pred in predicted:
            pred_name = normalize_name(pred["name"])
            if exp_name == pred_name:
                if value_within_tolerance(pred.get("value"), exp.get("value")):
                    exact_matches += 1
                break

    exact_match_rate = exact_matches / len(expected) if expected else 0

    # Status accuracy (for matched names)
    status_correct = 0
    status_total = 0
    for exp in expected:
        exp_name = normalize_name(exp["name"])
        for pred in predicted:
            if normalize_name(pred["name"]) == exp_name:
                status_total += 1
                if pred.get("status") == exp.get("status"):
                    status_correct += 1
                break

    status_acc = status_correct / status_total if status_total > 0 else 0

    return {
        "expected_count": len(expected),
        "predicted_count": len(predicted),
        "true_positives": len(true_pos_names),
        "false_positives": len(false_pos_names),
        "false_negatives": len(false_neg_names),
        "name_precision": name_precision,
        "name_recall": name_recall,
        "name_f1": name_f1,
        "exact_match_rate": exact_match_rate,
        "status_accuracy": status_acc,
    }


def run_evaluation(test_cases: list) -> list:
    all_results = []

    for tc in test_cases:
        predicted = MOCK_EXTRACTIONS.get(tc["id"], [])
        expected = tc["expected_biomarkers"]
        metrics = evaluate_report(expected, predicted)
        metrics["id"] = tc["id"]
        all_results.append(metrics)

        print(f"  [{tc['id']}]")
        print(f"    Expected biomarkers : {metrics['expected_count']}")
        print(f"    Predicted biomarkers: {metrics['predicted_count']}")
        print(f"    Name Precision: {metrics['name_precision']:.3f}"
              f"  Recall: {metrics['name_recall']:.3f}"
              f"  F1: {metrics['name_f1']:.3f}")
        print(f"    Exact Match Rate  : {metrics['exact_match_rate']:.3f}")
        print(f"    Status Accuracy   : {metrics['status_accuracy']:.3f}")
        print(f"    False Positives   : {metrics['false_positives']}"
              f"  |  False Negatives: {metrics['false_negatives']}")
        print()

    return all_results


def print_summary(results: list):
    print("═" * 60)
    print("  BIOMARKER EXTRACTION — SUMMARY")
    print("═" * 60)

    keys = ["name_precision", "name_recall", "name_f1", "exact_match_rate", "status_accuracy"]
    for k in keys:
        avg = sum(r[k] for r in results) / len(results)
        print(f"  {k:22s}: {avg:.4f}")

    total_fp = sum(r["false_positives"] for r in results)
    total_fn = sum(r["false_negatives"] for r in results)
    print(f"\n  Total False Positives  : {total_fp}")
    print(f"  Total False Negatives  : {total_fn}")
    print()


def save_results(results: list):
    import csv
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"biomarker_{ts}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  Results saved to: {out_path}")


def main():
    test_path = SAMPLE_DIR / "biomarker_ground_truth.json"
    with open(test_path) as f:
        test_cases = json.load(f)

    print(f"\n  DoctorG Biomarker Extraction Evaluation — {len(test_cases)} reports\n")
    results = run_evaluation(test_cases)
    print_summary(results)
    save_results(results)


if __name__ == "__main__":
    main()
