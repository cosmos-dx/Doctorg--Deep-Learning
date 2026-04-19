"""
Main FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import logging

from app.core.config import settings
from app.core.constants import APIEndpoints
from app.db.database import init_db
from app.api.v1 import auth, chat, feedback, user, reports, profile
from app.services.knowledge_base_init import initialize_knowledge_bases

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "Accept", "Origin"],
    max_age=3600
)


@app.middleware("http")
async def add_security_headers(request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    return response


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting DoctorG Medical AI Backend")
    init_db()
    logger.info("Database initialized")
    
    kb_status = await initialize_knowledge_bases()
    logger.info(f"Knowledge bases initialization status: {kb_status}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down DoctorG Medical AI Backend")


@app.get(APIEndpoints.HEALTH)
async def health_check():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "version": settings.APP_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": "connected",
            "llm": "ready",
            "rag": "ready",
            "report_parser": "ready",
            "daily_advisor": "ready",
            "metrics": "ready"
        }
    })


app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["Feedback"])
app.include_router(user.router, prefix="/api/v1/user", tags=["User"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["Medical Reports"])
app.include_router(profile.router, prefix="/api/v1/profile", tags=["Health Profile"])
