"""
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class SubscriptionTier(str, Enum):
    """Subscription tier enumeration."""
    FREE = "free"
    PREMIUM = "premium"


class ConfidenceLevel(str, Enum):
    """Confidence level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Severity(str, Enum):
    """Severity level enumeration."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class UserRegisterRequest(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


class UserLoginRequest(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserProfile(BaseModel):
    """User profile response."""
    id: str
    email: str
    full_name: Optional[str]
    subscription_tier: SubscriptionTier
    sessions_used: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    """Chat/symptom input request."""
    symptoms: List[str] = Field(..., min_items=1)
    message: Optional[str] = None
    session_id: Optional[str] = None


class MedicalResponse(BaseModel):
    """Structured medical response from LLM."""
    possible_conditions: List[str]
    confidence_level: ConfidenceLevel
    follow_up_questions: List[str]
    risk_factors: List[str]
    suggested_tests: List[str]
    lifestyle_recommendations: List[str]
    severity: Severity
    should_see_doctor: bool
    reasoning: Optional[str] = None


class StreamChunk(BaseModel):
    """SSE stream chunk."""
    content: Optional[str] = None
    done: bool = False
    error: Optional[str] = None
    structured_data: Optional[MedicalResponse] = None


class FeedbackRequest(BaseModel):
    """User feedback request."""
    session_id: str
    rating: int = Field(..., ge=1, le=5)
    correct_diagnosis: Optional[str] = None
    helpful: bool
    comments: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Feedback submission response."""
    status: str
    message: str


class SessionResponse(BaseModel):
    """User session response."""
    id: str
    user_id: str
    symptoms: List[str]
    diagnosis: MedicalResponse
    timestamp: datetime
    feedback_rating: Optional[int] = None
    
    class Config:
        from_attributes = True


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: datetime
    services: dict


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime
