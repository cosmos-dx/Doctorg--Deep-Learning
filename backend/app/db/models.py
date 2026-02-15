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
    """Conversation history for RAG memory."""
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(String, nullable=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    embedding_vector_id = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
