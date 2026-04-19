"""
Medical Reports API.
Handles upload, parsing, listing, detail, and trend retrieval.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User, MedicalReport, ReportBiomarker
from app.core.security import get_current_user
from app.models.schemas import (
    MedicalReportUploadResponse,
    MedicalReportSummary,
    MedicalReportDetail,
    BiomarkerTrendsResponse,
)
from app.services.report_parser import parse_report
from app.services.trend_service import (
    get_biomarker_trends,
    get_report_list,
    get_report_detail,
    summarize_health_history,
)
from app.services.openai_service import create_openai_service
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_MIME_TYPES = {
    "application/pdf": "pdf",
    "image/jpeg": "image",
    "image/jpg": "image",
    "image/png": "image",
    "image/webp": "image",
}

MAX_FILE_SIZE_MB = 20


@router.post("/upload", response_model=MedicalReportUploadResponse)
async def upload_report(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Upload a medical report (PDF or image).
    Automatically:
    - Extracts text (PyMuPDF for PDF, GPT-4o Vision for images)
    - Classifies whether it's a medical document
    - Extracts structured biomarkers
    - Generates an AI narrative summary
    """
    content_type = file.content_type or ""
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{content_type}'. Upload PDF or image (JPG/PNG/WebP)."
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File too large. Max {MAX_FILE_SIZE_MB}MB.")

    is_image = ALLOWED_MIME_TYPES[content_type] == "image"
    openai_service = create_openai_service()

    try:
        parsed = await parse_report(
            file_bytes=file_bytes,
            filename=file.filename or f"report_{uuid.uuid4().hex[:8]}",
            openai_service=openai_service,
            is_image=is_image,
        )
    except Exception as e:
        logger.error(f"Report parsing failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse report. Please try again.")

    # Store report
    report = MedicalReport(
        user_id=str(current_user.id),
        filename=file.filename or "unnamed",
        file_path=parsed["file_path"],
        file_type=parsed["file_type"],
        report_type=parsed["report_type"],
        is_medical=parsed["is_medical"],
        is_medical_confidence=parsed["is_medical_confidence"],
        raw_text=parsed["raw_text"],
        extracted_data=parsed["extracted_data"],
        ai_summary=parsed["ai_summary"],
        report_date=parsed["report_date"],
    )
    db.add(report)
    db.flush()  # get report.id before committing

    # Store individual biomarkers
    biomarker_count = 0
    for b in (parsed["extracted_data"] or []):
        if b.get("name") is None:
            continue
        report_date = parsed.get("report_date")
        if b.get("report_date") and not report_date:
            try:
                report_date = datetime.fromisoformat(b["report_date"])
            except Exception:
                pass
        biomarker = ReportBiomarker(
            report_id=report.id,
            user_id=str(current_user.id),
            name=b.get("name", "Unknown"),
            value=b.get("value"),
            unit=b.get("unit"),
            reference_low=b.get("reference_low"),
            reference_high=b.get("reference_high"),
            status=b.get("status", "unknown"),
            report_date=report_date,
        )
        db.add(biomarker)
        biomarker_count += 1

    db.commit()

    return MedicalReportUploadResponse(
        success=True,
        report_id=report.id,
        is_medical=parsed["is_medical"],
        report_type=parsed["report_type"],
        biomarker_count=biomarker_count,
        ai_summary=parsed["ai_summary"] or "No summary available.",
        message=(
            "Report processed successfully."
            if parsed["is_medical"]
            else "⚠️ This document doesn't appear to be a medical report. It has been saved but may not contain useful health data."
        ),
    )


@router.get("/", response_model=list)
async def list_reports(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=200),
):
    """List all medical reports for the current user, newest first."""
    return await get_report_list(str(current_user.id), db, limit)


@router.get("/trends")
async def biomarker_trends(
    biomarker: Optional[str] = Query(None, description="Filter by biomarker name (partial match)"),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get time-series biomarker trends for the current user.
    Optionally filter by a specific biomarker name.
    """
    trends = await get_biomarker_trends(str(current_user.id), biomarker, db, limit)
    return {
        "trends": trends,
        "total_biomarkers": len(trends),
    }


@router.get("/summary")
async def health_history_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    GPT-4o narrative summary of the user's full lab history.
    Analyses trends across all uploaded reports.
    """
    openai_service = create_openai_service()
    summary = await summarize_health_history(str(current_user.id), db, openai_service)
    return {"summary": summary}


@router.get("/{report_id}")
async def get_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get full detail of a single medical report, including biomarkers."""
    detail = await get_report_detail(report_id, str(current_user.id), db)
    if not detail:
        raise HTTPException(status_code=404, detail="Report not found.")
    return detail


@router.delete("/{report_id}")
async def delete_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a medical report and all its biomarker records."""
    report = (
        db.query(MedicalReport)
        .filter(MedicalReport.id == report_id, MedicalReport.user_id == str(current_user.id))
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")
    db.delete(report)
    db.commit()
    return {"success": True, "message": "Report deleted."}
