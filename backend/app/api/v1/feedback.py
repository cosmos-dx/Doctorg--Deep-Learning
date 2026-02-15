"""
Feedback API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User, Feedback
from app.models.schemas import FeedbackRequest, FeedbackResponse
from app.core.security import get_current_user
from app.core.constants import SuccessMessages, ErrorMessages

router = APIRouter()


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit feedback for a medical consultation session."""
    feedback = Feedback(
        user_id=current_user.id,
        session_id=request.session_id,
        rating=request.rating,
        correct_diagnosis=request.correct_diagnosis,
        helpful=request.helpful,
        comments=request.comments
    )
    
    db.add(feedback)
    db.commit()
    
    return FeedbackResponse(
        status="success",
        message=SuccessMessages.FEEDBACK_RECEIVED
    )
