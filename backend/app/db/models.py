"""
SQLAlchemy database models.
"""

from sqlalchemy import Column, String, Integer, DateTime, Boolean, JSON, Float, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid


Base = declarative_base()


def generate_uuid():
    """Generate UUID for primary keys."""
    return str(uuid.uuid4())


class User(Base):
    """User model."""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    subscription_tier = Column(String, default="free", nullable=False)
    sessions_used = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    feedback = relationship("Feedback", back_populates="user", cascade="all, delete-orphan")
    reports = relationship("MedicalReport", back_populates="user", cascade="all, delete-orphan")
    health_profile = relationship("UserHealthProfile", back_populates="user", uselist=False)


class UserSession(Base):
    """User session model for storing medical consultations."""
    __tablename__ = "user_sessions"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    symptoms = Column(JSON, nullable=False)
    diagnosis = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    faiss_index = Column(Integer, nullable=True)
    session_duration_seconds = Column(Float, nullable=True)
    
    user = relationship("User", back_populates="sessions")
    feedback = relationship("Feedback", back_populates="session", uselist=False)


class Feedback(Base):
    """Feedback model for continuous learning."""
    __tablename__ = "feedback"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(String, ForeignKey("user_sessions.id"), nullable=False, index=True)
    rating = Column(Integer, nullable=False)
    correct_diagnosis = Column(String, nullable=True)
    helpful = Column(Boolean, default=True)
    comments = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    processed_for_training = Column(Boolean, default=False)
    
    user = relationship("User", back_populates="feedback")
    session = relationship("UserSession", back_populates="feedback")


class Conversation(Base):
    """Conversation history for RAG memory and multi-turn tracking."""
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(String, nullable=False, index=True)
    turn_number = Column(Integer, default=0)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    agent_type = Column(String, nullable=True)
    extra_data = Column(JSON, nullable=True)
    embedding_vector_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class MedicalReport(Base):
    """
    Uploaded medical reports (blood tests, imaging, etc.).
    Stores both the raw file path and GPT-4o extracted structured data.
    """
    __tablename__ = "medical_reports"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)           # "pdf" | "image"
    report_type = Column(String, nullable=True)          # "blood_test" | "imaging" | "urinalysis" | "unknown"
    is_medical = Column(Boolean, default=True)
    is_medical_confidence = Column(Float, nullable=True)
    raw_text = Column(Text, nullable=True)               # extracted OCR / PDF text
    extracted_data = Column(JSON, nullable=True)         # structured biomarkers JSON
    ai_summary = Column(Text, nullable=True)             # GPT-4o narrative summary
    report_date = Column(DateTime, nullable=True)        # date the test was done (from report)
    uploaded_at = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="reports")
    biomarkers = relationship("ReportBiomarker", back_populates="report", cascade="all, delete-orphan")


class ReportBiomarker(Base):
    """
    Individual biomarker values extracted from a medical report.
    Used for time-series trend plotting across multiple reports.
    """
    __tablename__ = "report_biomarkers"

    id = Column(String, primary_key=True, default=generate_uuid)
    report_id = Column(String, ForeignKey("medical_reports.id"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String, nullable=False, index=True)    # e.g., "Hemoglobin", "WBC"
    value = Column(Float, nullable=True)
    unit = Column(String, nullable=True)                 # e.g., "g/dL", "×10³/µL"
    reference_low = Column(Float, nullable=True)
    reference_high = Column(Float, nullable=True)
    status = Column(String, nullable=True)               # "normal" | "high" | "low" | "critical"
    report_date = Column(DateTime, nullable=True, index=True)

    report = relationship("MedicalReport", back_populates="biomarkers")


class UserHealthProfile(Base):
    """
    Persistent medical profile for a user.
    Used to personalize AI consultancy across sessions.
    """
    __tablename__ = "user_health_profiles"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), unique=True, nullable=False, index=True)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)               # "male" | "female" | "other"
    blood_group = Column(String, nullable=True)          # "A+" | "O-" etc.
    height_cm = Column(Float, nullable=True)
    weight_kg = Column(Float, nullable=True)
    allergies = Column(JSON, nullable=True)              # list of strings
    chronic_conditions = Column(JSON, nullable=True)     # list of strings
    current_medications = Column(JSON, nullable=True)    # list of strings
    family_history = Column(JSON, nullable=True)         # list of strings
    lifestyle_notes = Column(Text, nullable=True)        # free text
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="health_profile")



