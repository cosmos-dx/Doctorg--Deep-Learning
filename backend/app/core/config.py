"""
Configuration management for DoctorG backend.
All settings loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    APP_NAME: str = "DoctorG Medical AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    DATABASE_URL: str
    REDIS_URL: str
    
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    
    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:3001"]
    
    MODEL_PATH: str = "models/"
    DATA_PATH: str = "data/"
    
    MAX_WORKERS: int = 4
    LOG_LEVEL: str = "INFO"
    
    RATE_LIMIT_PER_MINUTE: int = 60
    
    PUBMED_API_KEY: Optional[str] = None
    PUBMED_EMAIL: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


def get_settings() -> Settings:
    """
    Factory function to create Settings instance.
    Returns singleton settings object.
    """
    return Settings()


settings = get_settings()
