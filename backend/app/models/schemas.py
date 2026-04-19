"""
Pydantic models for request/response validation.
Extended with medical report, health profile, and daily advisor schemas.
"""

from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict, Any
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


class Gender(str, Enum):
    """Gender enumeration."""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"


# ─────────────────────────────── Auth ────────────────────────────────

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


# ─────────────────────────────── Chat ────────────────────────────────

class ChatRequest(BaseModel):
    """Chat/symptom input request."""
    symptoms: List[str] = Field(default_factory=list)
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


# ─────────────────────────────── Feedback ────────────────────────────

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


# ─────────────────────────────── Sessions ────────────────────────────

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


# ─────────────────────────────── Health Profile ──────────────────────

class HealthProfileRequest(BaseModel):
    """Request to create or update a user's health profile."""
    age: Optional[int] = Field(None, ge=1, le=130)
    gender: Optional[Gender] = None
    blood_group: Optional[str] = None  # e.g. "A+", "O-"
    height_cm: Optional[float] = Field(None, ge=50, le=300)
    weight_kg: Optional[float] = Field(None, ge=1, le=700)
    allergies: Optional[List[str]] = None
    chronic_conditions: Optional[List[str]] = None
    current_medications: Optional[List[str]] = None
    family_history: Optional[List[str]] = None
    lifestyle_notes: Optional[str] = None


class HealthProfileResponse(BaseModel):
    """User health profile response."""
    id: str
    user_id: str
    age: Optional[int]
    gender: Optional[str]
    blood_group: Optional[str]
    height_cm: Optional[float]
    weight_kg: Optional[float]
    bmi: Optional[float]
    allergies: Optional[List[str]]
    chronic_conditions: Optional[List[str]]
    current_medications: Optional[List[str]]
    family_history: Optional[List[str]]
    lifestyle_notes: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ─────────────────────────────── Medical Reports ─────────────────────

class BiomarkerOut(BaseModel):
    """Single biomarker reading from a report."""
    id: str
    name: str
    value: Optional[float]
    unit: Optional[str]
    reference_low: Optional[float]
    reference_high: Optional[float]
    status: Optional[str]  # "normal" | "high" | "low" | "critical"
    report_date: Optional[datetime]

    class Config:
        from_attributes = True


class MedicalReportSummary(BaseModel):
    """Lightweight report summary for listing."""
    id: str
    filename: str
    report_type: Optional[str]
    is_medical: bool
    report_date: Optional[datetime]
    uploaded_at: datetime
    ai_summary: Optional[str]
    biomarker_count: int = 0

    class Config:
        from_attributes = True


class MedicalReportDetail(BaseModel):
    """Full report detail including biomarkers."""
    id: str
    filename: str
    file_type: str
    report_type: Optional[str]
    is_medical: bool
    is_medical_confidence: Optional[float]
    ai_summary: Optional[str]
    report_date: Optional[datetime]
    uploaded_at: datetime
    biomarkers: List[BiomarkerOut] = []

    class Config:
        from_attributes = True


class MedicalReportUploadResponse(BaseModel):
    """Response after uploading a medical report."""
    success: bool
    report_id: str
    is_medical: bool
    report_type: Optional[str]
    biomarker_count: int
    ai_summary: str
    message: str


class BiomarkerTrendPoint(BaseModel):
    """Single data point in a biomarker trend."""
    date: str
    value: float
    unit: Optional[str]
    status: Optional[str]
    reference_low: Optional[float]
    reference_high: Optional[float]
    report_id: str


class BiomarkerTrendsResponse(BaseModel):
    """All biomarker trends for a user."""
    trends: Dict[str, List[BiomarkerTrendPoint]]  # biomarker_name → [points]
    total_reports: int


# ─────────────────────────────── Daily Advisor ───────────────────────

class DailyAdviceRequest(BaseModel):
    """Request for daily lifestyle advice."""
    message: str
    symptoms: Optional[List[str]] = []
    session_id: Optional[str] = None


class DailyAdviceResponse(BaseModel):
    """Response from daily advisor agent."""
    advice: str
    session_id: str


# ─────────────────────────────── Misc ────────────────────────────────

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
