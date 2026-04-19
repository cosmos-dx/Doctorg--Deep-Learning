"""
Medical Report Parser Service.
Handles extraction of structured data from uploaded PDF and image medical reports.
Uses PyMuPDF for PDFs and GPT-4o Vision for images/complex layouts.
"""

import os
import base64
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("data/reports")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Classification prompt ──────────────────────────────────────────────
MEDICAL_CLASSIFIER_PROMPT = """You are a document classifier specialising in medical records.

Analyse the following text and determine:
1. Is this a medical/health report? (blood test, CBC, urinalysis, imaging report, pathology, ECG, X-ray report, prescription, discharge summary, etc.)
2. What specific type of report is it?

Reply ONLY with valid JSON in this exact format:
{
  "is_medical": true | false,
  "report_type": "blood_test" | "urinalysis" | "imaging" | "pathology" | "ecg" | "prescription" | "discharge_summary" | "other" | "unknown",
  "confidence": 0.0-1.0,
  "reasoning": "one sentence"
}"""

# ── Biomarker extraction prompt ────────────────────────────────────────
BIOMARKER_EXTRACTION_PROMPT = """You are a medical data extraction specialist.

Extract ALL measurable biomarkers/lab values from this medical report text.
For each value found, provide:
- name: parameter name (standardised, e.g. "Hemoglobin", "WBC", "Creatinine")
- value: numeric value as float (null if not numeric)
- unit: measurement unit as string
- reference_low: lower bound of normal range (null if not stated)
- reference_high: upper bound of normal range (null if not stated)
- status: "normal" | "high" | "low" | "critical" | "unknown"
- report_date: ISO date string if mentioned in report (null otherwise)

Reply ONLY with valid JSON array:
[
  {"name": "Hemoglobin", "value": 13.5, "unit": "g/dL", "reference_low": 12.0, "reference_high": 17.5, "status": "normal", "report_date": null},
  ...
]

If no biomarkers found, return an empty array: []"""

# ── Summary prompt ─────────────────────────────────────────────────────
SUMMARY_PROMPT = """You are a medical report summariser. Write a clear, patient-friendly summary of this medical report.

Guidelines:
- Start with what type of report this is and when it was done (if known)
- Highlight any abnormal values and what they might mean
- Note any values that are within normal range (briefly)
- End with a single sentence: "Please consult your healthcare provider to discuss these results."
- Keep the tone warm, clear, and non-alarming
- 150-200 words maximum

IMPORTANT: This is an AI-generated summary for informational purposes only, not medical advice."""


def save_upload(file_bytes: bytes, filename: str) -> str:
    """Save uploaded file bytes to disk, return absolute path."""
    safe_name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{filename}"
    dest = UPLOAD_DIR / safe_name
    dest.write_bytes(file_bytes)
    logger.info(f"Saved upload to {dest}")
    return str(dest)


def extract_pdf_text(file_path: str) -> str:
    """
    Extract all text from a PDF using PyMuPDF.
    Falls back to empty string on any error.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        doc.close()
        full_text = "\n".join(pages_text).strip()
        logger.info(f"Extracted {len(full_text)} chars from PDF: {file_path}")
        return full_text
    except ImportError:
        logger.warning("PyMuPDF not installed. Cannot parse PDF.")
        return ""
    except Exception as e:
        logger.error(f"PDF extraction failed for {file_path}: {e}")
        return ""


def _encode_image_base64(file_path: str) -> str:
    """Encode image file to base64 for GPT-4o Vision."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def extract_image_text_via_vision(file_path: str, openai_service) -> str:
    """
    Use GPT-4o Vision to extract text from an image-based medical report.
    Returns raw extracted text.
    """
    try:
        import openai as oai_lib
        client = openai_service.client
        ext = Path(file_path).suffix.lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        b64 = _encode_image_base64(file_path)

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Please transcribe all text you can see in this medical document "
                                "image, preserving the structure as much as possible. Include all "
                                "numbers, units, and reference ranges."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000
        )
        text = response.choices[0].message.content or ""
        logger.info(f"Vision OCR extracted {len(text)} chars from {file_path}")
        return text
    except Exception as e:
        logger.error(f"Vision extraction failed: {e}")
        return ""


async def classify_as_medical_report(text: str, openai_service) -> dict:
    """
    Ask GPT-4o whether the extracted text represents a medical report.
    Returns: {is_medical, report_type, confidence, reasoning}
    """
    if not text or len(text.strip()) < 20:
        return {
            "is_medical": False,
            "report_type": "unknown",
            "confidence": 0.0,
            "reasoning": "Too little text to classify"
        }
    snippet = text[:3000]  # keep cost low
    try:
        raw = await openai_service.complete(
            prompt=f"Document text:\n---\n{snippet}\n---",
            system_prompt=MEDICAL_CLASSIFIER_PROMPT,
            temperature=0.1,
            max_tokens=150
        )
        raw_cleaned = raw.strip()
        if raw_cleaned.startswith("```json"):
            raw_cleaned = raw_cleaned[7:]
        elif raw_cleaned.startswith("```"):
            raw_cleaned = raw_cleaned[3:]
        if raw_cleaned.endswith("```"):
            raw_cleaned = raw_cleaned[:-3]
        
        result = json.loads(raw_cleaned.strip())
        return result
    except Exception as e:
        logger.error(f"Medical classification failed: {e}")
        return {
            "is_medical": True,
            "report_type": "unknown",
            "confidence": 0.5,
            "reasoning": "Classification failed, assumed medical"
        }


async def extract_biomarkers(text: str, openai_service) -> list:
    """
    Extract structured biomarker list from report text using GPT-4o.
    Returns list of biomarker dicts.
    """
    if not text or len(text.strip()) < 20:
        return []
    snippet = text[:4000]
    try:
        raw = await openai_service.complete(
            prompt=f"Report text:\n---\n{snippet}\n---",
            system_prompt=BIOMARKER_EXTRACTION_PROMPT,
            temperature=0.1,
            max_tokens=1500
        )
        # Strip markdown fences safely if present
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        biomarkers = json.loads(cleaned)
        if not isinstance(biomarkers, list):
            return []
        logger.info(f"Extracted {len(biomarkers)} biomarkers")
        return biomarkers
    except Exception as e:
        logger.error(f"Biomarker extraction failed: {e}")
        return []


async def generate_report_summary(text: str, openai_service) -> str:
    """
    Generate a patient-friendly narrative summary of the report.
    """
    if not text or len(text.strip()) < 20:
        return "No content could be read from this report."
    snippet = text[:4000]
    try:
        summary = await openai_service.complete(
            prompt=f"Report text:\n---\n{snippet}\n---",
            system_prompt=SUMMARY_PROMPT,
            temperature=0.5,
            max_tokens=400
        )
        return summary.strip()
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return "Summary could not be generated at this time."


async def parse_report(
    file_bytes: bytes,
    filename: str,
    openai_service,
    is_image: bool = False
) -> dict:
    """
    Full parsing pipeline for an uploaded medical report.

    Returns a dict with:
      file_path, file_type, is_medical, is_medical_confidence,
      report_type, raw_text, extracted_data, ai_summary, report_date
    """
    file_path = save_upload(file_bytes, filename)
    file_type = "image" if is_image else "pdf"

    # 1. Extract raw text
    if is_image:
        raw_text = await extract_image_text_via_vision(file_path, openai_service)
    else:
        raw_text = extract_pdf_text(file_path)
        # If PDF text is very sparse (scanned PDF), fall back to Vision
        if len(raw_text.strip()) < 100:
            logger.info("Sparse PDF text — falling back to Vision OCR")
            raw_text = await extract_image_text_via_vision(file_path, openai_service)

    # 2. Classify
    classification = await classify_as_medical_report(raw_text, openai_service)

    # 3. Extract biomarkers only if medical
    biomarkers = []
    if classification.get("is_medical", False):
        biomarkers = await extract_biomarkers(raw_text, openai_service)

    # 4. Summary
    ai_summary = await generate_report_summary(raw_text, openai_service)

    # 5. Try to detect report_date from biomarkers list
    report_date = None
    for b in biomarkers:
        if b.get("report_date"):
            try:
                report_date = datetime.fromisoformat(b["report_date"])
                break
            except Exception:
                pass

    return {
        "file_path": file_path,
        "file_type": file_type,
        "is_medical": classification.get("is_medical", False),
        "is_medical_confidence": classification.get("confidence", 0.0),
        "report_type": classification.get("report_type", "unknown"),
        "raw_text": raw_text,
        "extracted_data": biomarkers,
        "ai_summary": ai_summary,
        "report_date": report_date,
    }
