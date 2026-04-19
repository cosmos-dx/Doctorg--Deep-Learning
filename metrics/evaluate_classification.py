#!/usr/bin/env python3
"""
evaluate_classification.py
--------------------------
Evaluates DoctorG's triage urgency classification using labelled test cases.

Metrics: Precision, Recall, F1-score, Accuracy (per-class and macro avg)
Uses sklearn classification_report for clean formatting.

Usage:
    python evaluate_classification.py [--mock] [--api-url http://localhost:8000]
"""

import json
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add backend to path for imports when running locally
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))

try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import numpy as np
except ImportError:
    print("ERROR: Install scikit-learn: pip install scikit-learn")
    sys.exit(1)

SAMPLE_DIR = Path(__file__).parent / "sample_data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Mock responses (for offline evaluation) ─────────────────────────────────
# These simulate what the model would output based on symptom keywords
def mock_predict_urgency(symptoms: list, message: str) -> str:
    """Simple keyword-based mock to simulate triage classification."""
    combined = (message + " " + " ".join(symptoms)).lower()
    
    emergency_keywords = [
        "chest pain", "left arm", "can't breathe", "stroke", "seizure",
        "loss of consciousness", "worst headache", "throat swelling", "anaphylaxis",
        "drooping", "slurred speech", "severe bleeding"
    ]
    urgent_keywords = [
        "high blood pressure", "180/", "blurred vision", "high fever", "103",
        "rash", "joint pain", "severe"
    ]
    low_keywords = [
        "mild", "runny nose", "sneezing", "indigestion", "bloating",
        "sore throat", "back pain", "dizzy", "nausea", "anxiety", "insomnia"
    ]
    
    for kw in emergency_keywords:
        if kw in combined:
            return "emergency"
    for kw in urgent_keywords:
        if kw in combined:
            return "urgent"
    for kw in low_keywords:
        if kw in combined:
            return "low"
    return "moderate"


def run_mock_evaluation(test_cases: list) -> tuple:
    """Run evaluation using mock predictions (no API call needed)."""
    y_true, y_pred = [], []
    for tc in test_cases:
        true_label = tc["expected_urgency"]
        pred_label = mock_predict_urgency(tc["symptoms"], tc["message"])
        y_true.append(true_label)
        y_pred.append(pred_label)
        print(f"  [{tc['id']}] Expected: {true_label:10s} | Predicted: {pred_label:10s} "
              f"{'✓' if true_label == pred_label else '✗'}")
    return y_true, y_pred


def run_api_evaluation(test_cases: list, api_url: str, token: str) -> tuple:
    """Run evaluation by calling the live Doctorg API."""
    try:
        import requests
    except ImportError:
        print("Install requests: pip install requests")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    y_true, y_pred = [], []

    for tc in test_cases:
        try:
            resp = requests.post(
                f"{api_url}/api/v1/chat/predict",
                json={"symptoms": tc["symptoms"], "message": tc["message"]},
                headers=headers,
                timeout=30
            )
            data = resp.json()
            urgency = data.get("metadata", {}).get("urgency_level", "unknown")
            y_true.append(tc["expected_urgency"])
            y_pred.append(urgency)
            print(f"  [{tc['id']}] Expected: {tc['expected_urgency']:10s} | Got: {urgency:10s} "
                  f"{'✓' if tc['expected_urgency'] == urgency else '✗'}")
        except Exception as e:
            print(f"  [{tc['id']}] API error: {e}")
            y_true.append(tc["expected_urgency"])
            y_pred.append("error")

    return y_true, y_pred


def print_results(y_true: list, y_pred: list):
    """Print formatted classification report and confusion matrix."""
    labels = sorted(set(y_true + y_pred))
    
    print("\n" + "═" * 65)
    print("  DOCTORG TRIAGE CLASSIFICATION — EVALUATION RESULTS")
    print("═" * 65)
    
    acc = accuracy_score(y_true, y_pred)
    print(f"\n  Overall Accuracy: {acc:.1%}\n")
    
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        zero_division=0,
        digits=3
    )
    print(report)
    
    print("\n  Confusion Matrix:")
    print(f"  Labels: {labels}")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"  {cm}\n")
    
    return acc, report


def save_results(y_true, y_pred, acc, report, mode):
    """Save results to CSV."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"classification_{mode}_{ts}.txt"
    with open(out_path, "w") as f:
        f.write(f"Mode: {mode}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"  Results saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DoctorG triage classification")
    parser.add_argument("--mock", action="store_true", default=True,
                        help="Use mock predictions (default). Skip API calls.")
    parser.add_argument("--live", action="store_true",
                        help="Use live API for evaluation")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="API base URL")
    parser.add_argument("--token", default="",
                        help="JWT token for API authentication")
    args = parser.parse_args()

    test_path = SAMPLE_DIR / "triage_test_cases.json"
    with open(test_path) as f:
        test_cases = json.load(f)

    print(f"\n  DoctorG Triage Evaluation — {len(test_cases)} test cases\n")

    if args.live:
        if not args.token:
            print("ERROR: --token required for live evaluation")
            sys.exit(1)
        print("  Mode: LIVE API\n")
        y_true, y_pred = run_api_evaluation(test_cases, args.api_url, args.token)
        mode = "live"
    else:
        print("  Mode: MOCK (keyword-based simulation)\n")
        y_true, y_pred = run_mock_evaluation(test_cases)
        mode = "mock"

    acc, report = print_results(y_true, y_pred)
    save_results(y_true, y_pred, acc, report, mode)


if __name__ == "__main__":
    main()
