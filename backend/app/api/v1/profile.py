"""
Health Profile API.
Manages the user's persistent medical profile (age, conditions, medications, etc.)
and provides full consultation history.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User, UserHealthProfile, UserSession
from app.core.security import get_current_user
from app.models.schemas import HealthProfileRequest, HealthProfileResponse

router = APIRouter()
logger = logging.getLogger(__name__)


def _bmi(weight_kg: Optional[float], height_cm: Optional[float]) -> Optional[float]:
    """Compute BMI if data is available."""
    if weight_kg and height_cm and height_cm > 0:
        return round(weight_kg / ((height_cm / 100) ** 2), 1)
    return None


def _profile_to_dict(profile: UserHealthProfile) -> dict:
    """Convert ORM profile to a plain dict for response."""
    return {
        "id": profile.id,
        "user_id": profile.user_id,
        "age": profile.age,
        "gender": profile.gender,
        "blood_group": profile.blood_group,
        "height_cm": profile.height_cm,
        "weight_kg": profile.weight_kg,
        "bmi": _bmi(profile.weight_kg, profile.height_cm),
        "allergies": profile.allergies or [],
        "chronic_conditions": profile.chronic_conditions or [],
        "current_medications": profile.current_medications or [],
        "family_history": profile.family_history or [],
        "lifestyle_notes": profile.lifestyle_notes,
        "created_at": profile.created_at.isoformat(),
        "updated_at": profile.updated_at.isoformat(),
    }


@router.get("/health")
async def get_health_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve the current user's health profile."""
    profile = (
        db.query(UserHealthProfile)
        .filter(UserHealthProfile.user_id == str(current_user.id))
        .first()
    )
    if not profile:
        # Return empty profile scaffold rather than 404
        return {
            "id": None,
            "user_id": str(current_user.id),
            "age": None,
            "gender": None,
            "blood_group": None,
            "height_cm": None,
            "weight_kg": None,
            "bmi": None,
            "allergies": [],
            "chronic_conditions": [],
            "current_medications": [],
            "family_history": [],
            "lifestyle_notes": None,
            "created_at": None,
            "updated_at": None,
        }
    return _profile_to_dict(profile)


@router.put("/health")
async def upsert_health_profile(
    payload: HealthProfileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create or update the current user's health profile."""
    profile = (
        db.query(UserHealthProfile)
        .filter(UserHealthProfile.user_id == str(current_user.id))
        .first()
    )

    update_data = payload.model_dump(exclude_none=True)

    if profile:
        for key, value in update_data.items():
            setattr(profile, key, value)
    else:
        profile = UserHealthProfile(
            user_id=str(current_user.id),
            **update_data
        )
        db.add(profile)

    db.commit()
    db.refresh(profile)
    return _profile_to_dict(profile)


@router.get("/history")
async def consultation_history(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Return the user's full medical consultation history (sessions),
    newest first.
    """
    sessions = (
        db.query(UserSession)
        .filter(UserSession.user_id == str(current_user.id))
        .order_by(UserSession.timestamp.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "id": s.id,
            "symptoms": s.symptoms,
            "diagnosis": s.diagnosis,
            "timestamp": s.timestamp.isoformat(),
            "session_duration_seconds": s.session_duration_seconds,
            "feedback": {
                "rating": s.feedback.rating if s.feedback else None,
                "helpful": s.feedback.helpful if s.feedback else None,
                "comments": s.feedback.comments if s.feedback else None,
            } if s.feedback else None,
        }
        for s in sessions
    ]
