# Doctorg — Metrics & Evaluation Suite

This folder contains evaluation scripts to measure the performance of the Doctorg AI medical consultation system.

---

## Scripts

| Script | What it Measures |
|--------|-----------------|
| `evaluate_classification.py` | Triage urgency classification — Precision / Recall / F1 per class |
| `evaluate_rag.py` | RAG pipeline — Context Relevance, Faithfulness, Answer Completeness |
| `evaluate_generation.py` | Text generation quality — BLEU, ROUGE-L, METEOR |
| `evaluate_biomarker.py` | Biomarker extraction accuracy vs. ground truth |
| `compare_models.py` | Runs all evaluations and prints a consolidated comparison table |

---

## Running

```bash
cd metrics

# Single evaluation
python evaluate_classification.py
python evaluate_rag.py
python evaluate_generation.py
python evaluate_biomarker.py

# Full comparison (runs all and saves results/)
python compare_models.py
```

Results are saved as CSV to `metrics/results/`.

---

## Metric Definitions

### Classification Metrics
- **Precision**: Of all cases predicted as "urgent", how many actually were?
- **Recall**: Of all actual "urgent" cases, how many did we catch?
- **F1**: Harmonic mean of Precision and Recall — the main performance indicator
- **Support**: Number of test cases per class

### RAG Metrics
- **Context Relevance**: Does the retrieved context actually relate to the query?
- **Faithfulness**: Is the generated answer grounded in the retrieved context?
- **Answer Completeness**: Does the answer address all aspects of the question?

### Generation Metrics
- **BLEU**: Measures n-gram overlap with reference answers (precision-focused)
- **ROUGE-L**: Measures longest common subsequence with reference (recall-focused)
- **METEOR**: Incorporates synonyms & stemming — more human-aligned than BLEU

### Biomarker Extraction Metrics
- **Exact Match**: Biomarker name + value exactly matches ground truth
- **Partial Match**: Biomarker name matches (value may differ by ±5%)
- **False Positive Rate**: Hallucinated biomarkers not in the report

---

## Sample Data

Located in `sample_data/`:
- `triage_test_cases.json` — Labelled symptom → urgency level pairs
- `rag_test_queries.json` — Medical queries with expected context topics
- `generation_references.json` — Query + reference answer pairs
- `biomarker_ground_truth.json` — Sample reports with verified biomarker extractions
