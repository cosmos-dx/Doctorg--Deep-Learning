"""
Database connection and session management with dependency injection.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator
from app.core.config import settings
from app.db.models import Base
import logging

logger = logging.getLogger(__name__)


class DatabaseFactory:
    """Factory for creating database connections."""
    
    _engine = None
    _session_factory = None
    
    @classmethod
    def create_engine(cls):
        """Create database engine with proper configuration."""
        if cls._engine is None:
            cls._engine = create_engine(
                settings.DATABASE_URL,
                pool_pre_ping=True,
                echo=settings.DEBUG,
                connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
            )
            logger.info("Database engine created")
        return cls._engine
    
    @classmethod
    def create_session_factory(cls):
        """Create session factory."""
        if cls._session_factory is None:
            engine = cls.create_engine()
            cls._session_factory = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=engine
            )
            logger.info("Session factory created")
        return cls._session_factory
    
    @classmethod
    def create_tables(cls):
        """Create all database tables."""
        engine = cls.create_engine()
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created")


def get_db() -> Generator[Session, None, None]:
    """
    Dependency injection for database sessions.
    Use this in FastAPI route dependencies.
    """
    session_factory = DatabaseFactory.create_session_factory()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database - create tables if they don't exist."""
    DatabaseFactory.create_tables()
