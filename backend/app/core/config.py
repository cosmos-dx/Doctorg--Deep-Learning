"""
Configuration management for DoctorG backend.
All settings loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional, List, Union
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    APP_NAME: str = "DoctorG Medical AI"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6379"
    
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_MAX_TOKENS: int = 2048
    OPENAI_TEMPERATURE: float = 0.7
    
    GOOGLE_API_KEY: Optional[str] = None
    
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    CORS_ORIGINS: Union[List[str], str] = "http://localhost:3000,http://localhost:3001"
    
    MODEL_PATH: str = "models/"
    DATA_PATH: str = "data/"
    
    MAX_WORKERS: int = 4
    LOG_LEVEL: str = "INFO"
    
    RATE_LIMIT_PER_MINUTE: int = 60
    
    PUBMED_API_KEY: Optional[str] = None
    PUBMED_EMAIL: Optional[str] = None
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Factory function to create Settings instance.
    Returns singleton settings object.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


settings = get_settings()
