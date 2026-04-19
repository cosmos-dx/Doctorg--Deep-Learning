"""
Custom exceptions and error handling for DoctorG backend.
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException, status


class DoctorGException(Exception):
    """Base exception class for DoctorG application."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class AuthenticationError(DoctorGException):
    """Authentication-related errors."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, error_code="AUTH_ERROR", **kwargs)


class AuthorizationError(DoctorGException):
    """Authorization-related errors."""
    
    def __init__(self, message: str = "Insufficient permissions", **kwargs):
        super().__init__(message, error_code="AUTHZ_ERROR", **kwargs)


class DatabaseError(DoctorGException):
    """Database operation errors."""
    
    def __init__(self, message: str = "Database operation failed", **kwargs):
        super().__init__(message, error_code="DB_ERROR", **kwargs)


class AgentError(DoctorGException):
    """Agent execution errors."""
    
    def __init__(self, message: str = "Agent execution failed", agent_type: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if agent_type:
            details["agent_type"] = agent_type
        kwargs["details"] = details
        super().__init__(message, error_code="AGENT_ERROR", **kwargs)


class RAGError(DoctorGException):
    """RAG system errors."""
    
    def __init__(self, message: str = "RAG retrieval failed", **kwargs):
        super().__init__(message, error_code="RAG_ERROR", **kwargs)


class OpenAIError(DoctorGException):
    """OpenAI API errors."""
    
    def __init__(self, message: str = "OpenAI API call failed", **kwargs):
        super().__init__(message, error_code="OPENAI_ERROR", **kwargs)


class ValidationError(DoctorGException):
    """Input validation errors."""
    
    def __init__(self, message: str = "Invalid input", **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)


class EmergencyDetectedError(DoctorGException):
    """Emergency symptoms detected - special handling required."""
    
    def __init__(
        self, 
        message: str = "Emergency symptoms detected",
        symptoms: Optional[list] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if symptoms:
            details["detected_symptoms"] = symptoms
        kwargs["details"] = details
        super().__init__(message, error_code="EMERGENCY_DETECTED", **kwargs)


class HTTPErrorResponse:
    """HTTP error response mappings."""
    
    @staticmethod
    def from_exception(exc: DoctorGException, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR) -> HTTPException:
        """Convert custom exception to HTTPException."""
        return HTTPException(
            status_code=status_code,
            detail={
                "message": exc.message,
                "error_code": exc.error_code,
                "details": exc.details
            }
        )
    
    @staticmethod
    def authentication_error(message: str = "Authentication required") -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": message, "error_code": "AUTH_REQUIRED"}
        )
    
    @staticmethod
    def authorization_error(message: str = "Insufficient permissions") -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"message": message, "error_code": "FORBIDDEN"}
        )
    
    @staticmethod
    def not_found_error(message: str = "Resource not found") -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": message, "error_code": "NOT_FOUND"}
        )
    
    @staticmethod
    def validation_error(message: str = "Invalid input", details: Optional[Dict] = None) -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": message, "error_code": "VALIDATION_ERROR", "details": details or {}}
        )
    
    @staticmethod
    def internal_error(message: str = "Internal server error") -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": message, "error_code": "INTERNAL_ERROR"}
        )


class ErrorCodes:
    """Standard error codes used throughout the application."""
    AUTH_ERROR = "AUTH_ERROR"
    AUTHZ_ERROR = "AUTHZ_ERROR"
    DB_ERROR = "DB_ERROR"
    AGENT_ERROR = "AGENT_ERROR"
    RAG_ERROR = "RAG_ERROR"
    OPENAI_ERROR = "OPENAI_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    EMERGENCY_DETECTED = "EMERGENCY_DETECTED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    NOT_FOUND = "NOT_FOUND"
