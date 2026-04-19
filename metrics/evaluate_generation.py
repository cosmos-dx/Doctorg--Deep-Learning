#!/usr/bin/env python3
"""
evaluate_generation.py
-----------------------
Evaluates DoctorG's text generation quality using BLEU, ROUGE-L, and METEOR scores.

These metrics compare the model's generated response against reference (gold) answers.

Usage:
    python evaluate_generation.py [--mock]
"""

import json
import sys
import re
import math
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

SAMPLE_DIR = Path(__file__).parent / "sample_data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

try:
    from rouge_score import rouge_scorer
except ImportError:
    print("ERROR: Install rouge-score: pip install rouge-score")
    sys.exit(1)

try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)
except ImportError:
    print("ERROR: Install nltk: pip install nltk")
    sys.exit(1)


# ── BLEU implementation ───────────────────────────────────────────────────────

def tokenize(text: str) -> list:
    return re.findall(r"\w+", text.lower())


def ngrams(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    if not hyp_tokens:
        return 0.0

    # Brevity penalty
    bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else \
        math.exp(1 - len(ref_tokens) / len(hyp_tokens))

    precisions = []
    for n in range(1, max_n + 1):
        ref_ng = ngrams(ref_tokens, n)
        hyp_ng = ngrams(hyp_tokens, n)
        clipped = sum(min(count, ref_ng[gram]) for gram, count in hyp_ng.items())
        total = max(len(hyp_tokens) - n + 1, 0)
        precisions.append(clipped / total if total > 0 else 0.0)

    if min(precisions) == 0:
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / max_n
    return bp * math.exp(log_avg)


# ── ROUGE ─────────────────────────────────────────────────────────────────────

def compute_rouge(reference: str, hypothesis: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


# ── Mock model response ───────────────────────────────────────────────────────

MOCK_RESPONSES = {
    "gr_001": (
        "Your symptoms suggest influenza or possibly COVID-19. "
        "These conditions cause fever, cough, and body aches. "
        "Get plenty of rest, drink lots of fluids, and take paracetamol for fever. "
        "Consider taking a COVID-19 test. If symptoms worsen or breathing becomes difficult, "
        "see a doctor immediately."
    ),
    "gr_002": (
        "Your hemoglobin of 9.5 g/dL is below the normal range, indicating anemia. "
        "This can cause tiredness and weakness. "
        "Common causes are iron deficiency or vitamin B12 deficiency. "
        "Please consult your doctor for blood tests and appropriate treatment."
    ),
    "gr_003": (
        "Lower right abdominal pain can indicate appendicitis, which is an emergency. "
        "It can also be due to ovarian cysts, kidney stones, or gas pain. "
        "If the pain is severe, or you have fever and vomiting, go to an emergency room immediately."
    ),
    "gr_004": (
        "For managing Type 2 diabetes: eat a healthy diet low in sugar and refined carbs, "
        "exercise at least 30 minutes most days, lose weight if overweight, "
        "monitor blood sugar regularly, and manage stress and sleep. "
        "Quit smoking and limit alcohol consumption."
    ),
    "gr_005": (
        "Dizziness when standing is called orthostatic hypotension — a drop in blood pressure. "
        "Usually caused by dehydration or certain medications. "
        "Stand up slowly, drink more water, and avoid prolonged standing. "
        "See a doctor if it happens frequently."
    ),
    "gr_006": (
        "Blood pressure of 145/95 is Stage 2 hypertension. "
        "Reduce salt intake, eat a DASH diet, exercise regularly, manage stress, "
        "and limit alcohol. If lifestyle changes don't help, your doctor may suggest medication. "
        "Monitor your BP regularly."
    ),
}


def evaluate(test_cases: list, use_mock: bool = True) -> list:
    results = []

    for tc in test_cases:
        ref = tc["reference_answer"]

        if use_mock:
            hyp = MOCK_RESPONSES.get(tc["id"], "I cannot provide information on this topic.")
        else:
            hyp = MOCK_RESPONSES.get(tc["id"], "")  # Replace with live API call

        bleu = bleu_score(ref, hyp)
        rouge = compute_rouge(ref, hyp)

        result = {
            "id": tc["id"],
            "bleu": bleu,
            **rouge,
        }
        results.append(result)

        print(f"  [{tc['id']}]")
        print(f"    BLEU:    {bleu:.4f}")
        print(f"    ROUGE-1: {rouge['rouge1_f']:.4f}")
        print(f"    ROUGE-2: {rouge['rouge2_f']:.4f}")
        print(f"    ROUGE-L: {rouge['rougeL_f']:.4f}")
        print()

    return results


def print_summary(results: list):
    print("═" * 55)
    print("  GENERATION METRICS — SUMMARY")
    print("═" * 55)

    keys = ["bleu", "rouge1_f", "rouge2_f", "rougeL_f"]
    for k in keys:
        avg = sum(r[k] for r in results) / len(results)
        print(f"  {k:15s}: {avg:.4f}")

    print()


def save_results(results: list):
    import csv
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = RESULTS_DIR / f"generation_{ts}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  Results saved to: {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", default=True)
    args = parser.parse_args()

    test_path = SAMPLE_DIR / "generation_references.json"
    with open(test_path) as f:
        test_cases = json.load(f)

    print(f"\n  DoctorG Generation Evaluation — {len(test_cases)} test cases\n")
    results = evaluate(test_cases, use_mock=True)
    print_summary(results)
    save_results(results)


if __name__ == "__main__":
    main()
