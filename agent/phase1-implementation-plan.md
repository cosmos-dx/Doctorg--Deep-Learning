DoctorG 2.0 – Phase 1 Implementation Blueprint
Building the Medical Reasoning Brain
1. Backstory – Current State of DoctorG
Already Implemented:
- Symptom to Disease prediction model
- ~8K structured CSV dataset
- ML classification pipeline
- Flask backend (deployed)
- Basic frontend UI

Current Limitations:
- Static predictions
- No reasoning or follow-up
- No personalization
- No memory
- No structured output

Current Behavior:
Input: Symptoms
Output: Disease probabilities

Goal: Transform from prediction tool into Medical Reasoning AI Assistant.
2. Phase 1 Objective
Build:
1. Fine-tuned Medical LLM
2. Persistent Memory Engine (FAISS)
3. Structured Output System
4. Subscription Logic Layer

This becomes the startup’s core intellectual property.
3. Phase 1 Architecture
User
 ↓
Frontend (Next.js)
 ↓
Backend (FastAPI)
 ↓
Medical LLM (Fine-tuned)
 ↓
Memory Retrieval (FAISS)
 ↓
Structured Response Engine
4. Medical LLM Implementation
Base Model: 3B–7B parameter model (LLaMA / BioGPT / Mistral)

Training Method:
- LoRA / QLoRA
- Instruction tuning
- Clinical QA datasets
- PubMed abstracts
- Structured symptom CSV converted to dialogue format

Structured JSON Output Format:
{
  "possible_conditions": [],
  "confidence_level": "",
  "follow_up_questions": [],
  "risk_factors": [],
  "suggested_tests": [],
  "lifestyle_recommendations": []
}
5. Memory Engine Implementation
Technology:
- FAISS (Vector Database)
- Medical embedding model
- PostgreSQL for user metadata

Stored Data:
- Symptom history
- Past conditions
- Lifestyle data
- Lab summaries

Flow:
1. Convert conversation to embedding
2. Store in FAISS
3. Retrieve relevant history
4. Inject into LLM prompt
6. GPU Usage in Phase 1
GPU will be used for:

1. Fine-tuning (LoRA training, mixed precision)
2. Fast inference (low latency response)
3. Embedding generation and indexing

Without GPU: slow training
With GPU: hours instead of days
7. Backend Stack
Training:
- PyTorch
- Transformers
- PEFT (LoRA)
- BitsAndBytes

Inference:
- FastAPI
- FAISS
- PostgreSQL
- Optional: vLLM

Frontend:
- Next.js dashboard
- Health timeline UI
8. Business Logic
Free Tier:
- 5 sessions per month
- No memory
- Limited reasoning depth

Premium Tier (₹299–₹499/month):
- Unlimited sessions
- Memory enabled
- Structured reports
- Follow-up continuity

Backend Gating Logic:
If premium:
   Retrieve memory → Inject → Full reasoning
Else:
   Limited response
9. Core Use Case
Target User: 28-year-old IT professional

Example:
User: I have headache and stress.
DoctorG:
- Asks sleep duration
- Asks hydration
- Retrieves migraine history
- Suggests stress-related headache
- Provides lifestyle suggestions
- Advises doctor if severe
10. Legal Positioning
DoctorG must:
- Include medical disclaimer
- Avoid claiming diagnosis
- Market as AI health assistant

Positioning:
AI-powered preventive health assistant providing educational insights.
11. Development Timeline
Week 1–2: Data preparation
Week 3–4: Fine-tune LLM
Week 5: Structured output integration
Week 6: FAISS memory integration
Week 7: Subscription logic
Week 8: Beta launch
12. Deliverables After Phase 1
- Fine-tuned medical reasoning LLM
- Follow-up questioning
- Structured JSON output
- Persistent memory
- Subscription gating
- GPU-accelerated inference
