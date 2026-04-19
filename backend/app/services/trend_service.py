"""
Trend Service — biomarker time-series analysis and health history summarisation.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.models import MedicalReport, ReportBiomarker

logger = logging.getLogger(__name__)


HISTORY_SUMMARY_PROMPT = """You are a medical health advisor reviewing a patient's lab history over time.

Based on the structured health data below, write a concise but insightful health summary covering:
1. Key trends (improving / worsening / stable values)
2. Any consistently abnormal values that need attention
3. Positive health indicators
4. A brief recommendation to discuss findings with their doctor

Keep it patient-friendly, warm, and under 250 words.
IMPORTANT: This is for informational guidance only, not a medical diagnosis."""


async def get_biomarker_trends(
    user_id: str,
    biomarker_name: Optional[str],
    db: Session,
    limit: int = 20
) -> Dict[str, List[Dict]]:
    """
    Retrieve time-series values for one or all biomarkers for a user.

    Returns:
        {
          "biomarker_name": [
            {"date": "ISO", "value": 13.5, "unit": "g/dL", "status": "normal", "report_id": "..."},
            ...
          ],
          ...
        }
    """
    query = db.query(ReportBiomarker).filter(
        ReportBiomarker.user_id == user_id,
        ReportBiomarker.value.isnot(None),
        ReportBiomarker.report_date.isnot(None)
    )

    if biomarker_name:
        # Case-insensitive partial match
        query = query.filter(
            func.lower(ReportBiomarker.name).contains(biomarker_name.lower())
        )

    rows = query.order_by(ReportBiomarker.report_date.asc()).limit(limit).all()

    trends: Dict[str, List] = {}
    for row in rows:
        name = row.name
        if name not in trends:
            trends[name] = []
        trends[name].append({
            "date": row.report_date.isoformat() if row.report_date else None,
            "value": row.value,
            "unit": row.unit,
            "status": row.status,
            "reference_low": row.reference_low,
            "reference_high": row.reference_high,
            "report_id": row.report_id,
        })

    logger.info(f"Trends for user {user_id}: {len(trends)} biomarkers found")
    return trends


async def get_report_list(
    user_id: str,
    db: Session,
    limit: int = 50
) -> List[Dict]:
    """Return a list of report metadata for a user, newest first."""
    reports = (
        db.query(MedicalReport)
        .filter(MedicalReport.user_id == user_id)
        .order_by(MedicalReport.uploaded_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "filename": r.filename,
            "report_type": r.report_type,
            "is_medical": r.is_medical,
            "report_date": r.report_date.isoformat() if r.report_date else None,
            "uploaded_at": r.uploaded_at.isoformat(),
            "ai_summary": r.ai_summary,
            "biomarker_count": len(r.biomarkers),
        }
        for r in reports
    ]


async def get_report_detail(
    report_id: str,
    user_id: str,
    db: Session
) -> Optional[Dict]:
    """Return full report detail including all biomarkers."""
    report = (
        db.query(MedicalReport)
        .filter(MedicalReport.id == report_id, MedicalReport.user_id == user_id)
        .first()
    )
    if not report:
        return None

    biomarkers = [
        {
            "id": b.id,
            "name": b.name,
            "value": b.value,
            "unit": b.unit,
            "reference_low": b.reference_low,
            "reference_high": b.reference_high,
            "status": b.status,
            "report_date": b.report_date.isoformat() if b.report_date else None,
        }
        for b in report.biomarkers
    ]

    return {
        "id": report.id,
        "filename": report.filename,
        "file_type": report.file_type,
        "report_type": report.report_type,
        "is_medical": report.is_medical,
        "is_medical_confidence": report.is_medical_confidence,
        "raw_text": report.raw_text,
        "ai_summary": report.ai_summary,
        "report_date": report.report_date.isoformat() if report.report_date else None,
        "uploaded_at": report.uploaded_at.isoformat(),
        "biomarkers": biomarkers,
    }


async def summarize_health_history(
    user_id: str,
    db: Session,
    openai_service
) -> str:
    """
    Use GPT-4o to generate a narrative summary of a user's full lab history.
    Pulls the 10 most recent reports and all their biomarker trends.
    """
    reports = (
        db.query(MedicalReport)
        .filter(MedicalReport.user_id == user_id, MedicalReport.is_medical == True)
        .order_by(MedicalReport.report_date.desc().nullslast())
        .limit(10)
        .all()
    )

    if not reports:
        return "No medical reports found yet. Upload your lab reports to get a personalized health history summary."

    data_lines = []
    for r in reports:
        date_str = r.report_date.strftime("%b %Y") if r.report_date else "Unknown date"
        data_lines.append(f"\n== {r.report_type or 'Report'} ({date_str}) ==")
        for b in r.biomarkers:
            ref = ""
            if b.reference_low is not None and b.reference_high is not None:
                ref = f" [Ref: {b.reference_low}-{b.reference_high} {b.unit or ''}]"
            status_flag = f" ⚠ {b.status.upper()}" if b.status in ("high", "low", "critical") else ""
            data_lines.append(
                f"  {b.name}: {b.value} {b.unit or ''}{ref}{status_flag}"
            )

    data_text = "\n".join(data_lines)

    try:
        summary = await openai_service.complete(
            prompt=f"Patient lab history:\n{data_text}",
            system_prompt=HISTORY_SUMMARY_PROMPT,
            temperature=0.5,
            max_tokens=500
        )
        return summary.strip()
    except Exception as e:
        logger.error(f"History summarisation failed: {e}")
        return "Unable to generate health summary at this time. Please try again later."
