"""
User API endpoints.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from app.db.database import get_db
from app.db.models import User
from app.models.schemas import UserProfile, SessionResponse
from app.core.security import get_current_user
from app.ml.rag.memory_engine import create_memory_engine

router = APIRouter()
memory_engine = create_memory_engine()


@router.get("/profile", response_model=UserProfile)
async def get_profile(
    current_user: User = Depends(get_current_user)
):
    """Get current user profile."""
    return current_user


@router.get("/sessions", response_model=List[SessionResponse])
async def get_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 10
):
    """Get user's session history."""
    history = await memory_engine.get_user_session_history(
        current_user.id,
        db,
        limit
    )
    
    return history
