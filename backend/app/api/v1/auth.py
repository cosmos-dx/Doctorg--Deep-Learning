"""
Authentication API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User
from app.models.schemas import UserRegisterRequest, UserLoginRequest, TokenResponse
from app.core.security import PasswordService, JWTService
from app.core.constants import ErrorMessages, SuccessMessages

router = APIRouter()


@router.post("/register", response_model=TokenResponse)
async def register(
    request: UserRegisterRequest,
    db: Session = Depends(get_db)
):
    """Register a new user."""
    existing_user = db.query(User).filter(User.email == request.email).first()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    hashed_password = PasswordService.hash_password(request.password)
    
    new_user = User(
        email=request.email,
        hashed_password=hashed_password,
        full_name=request.full_name,
        subscription_tier="free"
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    access_token = JWTService.create_access_token(data={"sub": new_user.id})
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=3600
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    request: UserLoginRequest,
    db: Session = Depends(get_db)
):
    """Login user and return JWT token."""
    user = db.query(User).filter(User.email == request.email).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorMessages.INVALID_CREDENTIALS
        )
    
    if not PasswordService.verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorMessages.INVALID_CREDENTIALS
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )
    
    access_token = JWTService.create_access_token(data={"sub": user.id})
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=3600
    )
