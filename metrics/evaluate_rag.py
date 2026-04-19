#!/usr/bin/env python3
"""
evaluate_rag.py
----------------
Evaluates DoctorG's RAG pipeline quality.

Metrics:
- Context Relevance: Does the retrieved context cover the expected topics?
- Answer Faithfulness: Is the answer grounded in the context (no hallucination)?
- Answer Completeness: Does the answer address all aspects of the question?

All measured using keyword-overlap heuristics (no external API needed for offline run).

Usage:
    python evaluate_rag.py
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime

SAMPLE_DIR = Path(__file__).parent / "sample_data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Mock retrieved contexts ───────────────────────────────────────────────────
MOCK_CONTEXTS = {
    "rq_001": (
        "Type 2 diabetes is a chronic condition affecting how the body processes blood sugar (glucose). "
        "Symptoms include frequent urination, excessive thirst, unexplained weight loss, fatigue, "
        "blurred vision, slow wound healing, and frequent infections. Insulin resistance is a key mechanism. "
        "Risk factors include obesity, physical inactivity, family history, and age over 45."
    ),
    "rq_002": (
        "Hypertension is defined as blood pressure consistently at or above 130/80 mmHg. "
        "Diagnosis involves multiple readings on separate occasions, ambulatory monitoring, "
        "and ruling out secondary causes. Stage 1 is 130-139/80-89, Stage 2 is ≥140/90."
    ),
    "rq_003": (
        "High cholesterol management through diet: Avoid saturated fats found in red meat and full-fat dairy. "
        "Trans fats in fried and processed foods raise LDL. Dietary cholesterol in eggs and organ meats. "
        "Eat more soluble fiber (oats, beans), omega-3 fatty acids, and plant sterols."
    ),
    "rq_004": (
        "Heart attack (myocardial infarction) warning signs: chest pain or pressure lasting >20 minutes, "
        "pain radiating to left arm, jaw, back, or stomach; shortness of breath; cold sweats; "
        "nausea or vomiting; lightheadedness. Call emergency services immediately."
    ),
    "rq_005": (
        "Hemoglobin normal ranges: Adult males 13.5-17.5 g/dL, Adult females 12.0-15.5 g/dL. "
        "Values below normal indicate anemia. Iron deficiency anemia, vitamin B12 deficiency, "
        "and chronic disease are common causes. Symptoms: fatigue, weakness, pale skin."
    ),
    "rq_006": (
        "Asthma management: Short-acting beta2-agonists (albuterol) for acute attacks. "
        "Sit upright, breathe slowly, use rescue inhaler every 20 minutes x3. "
        "If no improvement, call emergency. Long-term: inhaled corticosteroids, "
        "avoid triggers (smoke, dust, cold air), peak flow monitoring."
    ),
    "rq_007": (
        "Kidney stones symptoms: severe flank or back pain (often colicky, comes in waves), "
        "pain radiating to lower abdomen and groin, blood in urine (hematuria), "
        "frequent and painful urination, nausea, vomiting, fever if infection. "
        "Diagnosis: CT scan, ultrasound, urinalysis."
    ),
    "rq_008": (
        "Iron deficiency anemia: insufficient iron for hemoglobin production. "
        "Causes: poor dietary intake, malabsorption (celiac disease), chronic blood loss "
        "(heavy menstruation, gastrointestinal bleeding), pregnancy. "
        "Lab findings: low hemoglobin, low ferritin, low MCV, high TIBC."
    ),
}

MOCK_ANSWERS = {
    "rq_001": (
        "Type 2 diabetes symptoms include frequent urination, increased thirst, "
        "fatigue, blurred vision, slow healing, and frequent infections."
    ),
    "rq_002": (
        "Blood pressure at or above 130/80 mmHg on multiple readings indicates hypertension. "
        "Doctors use ambulatory monitoring for confirmed diagnosis."
    ),
    "rq_003": (
        "Avoid red meat, full-fat dairy, fried foods, and processed foods. "
        "Eat oats, beans, and foods with omega-3 fatty acids."
    ),
    "rq_004": (
        "Heart attack signs: chest pain, left arm pain, shortness of breath, sweating, nausea. "
        "Call emergency services immediately."
    ),
    "rq_005": (
        "Normal hemoglobin: 13.5-17.5 g/dL for men, 12.0-15.5 g/dL for women. "
        "Below normal indicates anemia."
    ),
    "rq_006": (
        "For asthma attack: use albuterol inhaler, sit upright. "
        "Long-term: use controller inhalers and avoid triggers."
    ),
    "rq_007": (
        "Kidney stone symptoms: severe flank pain, pain to groin, blood in urine, "
        "frequent urination, nausea, fever."
    ),
    "rq_008": (
        "Iron deficiency anemia is caused by low iron intake, blood loss, or poor absorption. "
        "Shows as low hemoglobin and ferritin."
    ),
}


def tokenize(text: str) -> set:
    return set(re.findall(r"\b\w{3,}\b", text.lower()))


def context_relevance(context: str, expected_topics: list) -> float:
    """What fraction of expected topic keywords appear in the context?"""
    ctx_tokens = tokenize(context)
    hits = sum(
        any(word in ctx_tokens for word in topic.lower().split())
        for topic in expected_topics
    )
    return hits / len(expected_topics) if expected_topics else 0.0


def answer_faithfulness(answer: str, context: str) -> float:
    """What fraction of answer tokens appear in the context? (anti-hallucination)"""
    ans_tokens = tokenize(answer)
    ctx_tokens = tokenize(context)
    if not ans_tokens:
        return 0.0
    faithful = ans_tokens & ctx_tokens
    return len(faithful) / len(ans_tokens)


def answer_completeness(answer: str, reference: str) -> float:
    """What fraction of reference key terms appear in the answer?"""
    ref_tokens = tokenize(reference)
    ans_tokens = tokenize(answer)
    if not ref_tokens:
        return 0.0
    covered = ref_tokens & ans_tokens
    return len(covered) / len(ref_tokens)


def run_evaluation(test_cases: list) -> list:
    results = []
    for tc in test_cases:
        context = MOCK_CONTEXTS.get(tc["id"], "")
        answer = MOCK_ANSWERS.get(tc["id"], "")
        ref = tc["reference_answer"]
        topics = tc["expected_context_topics"]

        cr = context_relevance(context, topics)
        af = answer_faithfulness(answer, context)
        ac = answer_completeness(answer, ref)

        results.append({
            "id": tc["id"],
            "context_relevance": cr,
            "answer_faithfulness": af,
            "answer_completeness": ac,
            "rag_score": (cr + af + ac) / 3,
        })

        print(f"  [{tc['id']}]")
        print(f"    Context Relevance  : {cr:.3f}")
        print(f"    Answer Faithfulness: {af:.3f}")
        print(f"    Answer Completeness: {ac:.3f}")
        print(f"    RAG Score (avg)    : {(cr + af + ac) / 3:.3f}")
        print()

    return results


def print_summary(results: list):
    print("═" * 55)
    print("  RAG PIPELINE — EVALUATION SUMMARY")
    print("═" * 55)
    for k in ["context_relevance", "answer_faithfulness", "answer_completeness", "rag_score"]:
        avg = sum(r[k] for r in results) / len(results)
        print(f"  {k:26s}: {avg:.4f}")
    print()


def save_results(results: list):
    import csv
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"rag_{ts}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  Results saved to: {out_path}")


def main():
    test_path = SAMPLE_DIR / "rag_test_queries.json"
    with open(test_path) as f:
        test_cases = json.load(f)

    print(f"\n  DoctorG RAG Evaluation — {len(test_cases)} queries\n")
    results = run_evaluation(test_cases)
    print_summary(results)
    save_results(results)


if __name__ == "__main__":
    main()
