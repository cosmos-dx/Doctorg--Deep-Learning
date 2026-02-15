"""
Subscription service for managing user access and session limits.
"""

from sqlalchemy.orm import Session
from typing import Optional
import logging

from app.db.models import User, UserSession
from app.core.constants import SubscriptionLimits, SubscriptionTiers, ErrorMessages
from app.ml.llm.inference import MedicalLLMService
from app.ml.rag.memory_engine import MemoryEngine
from app.models.schemas import MedicalResponse

logger = logging.getLogger(__name__)


class SubscriptionService:
    """Service for managing subscription tiers and access control."""
    
    def __init__(
        self,
        llm_service: MedicalLLMService,
        memory_engine: MemoryEngine
    ):
        self.llm_service = llm_service
        self.memory_engine = memory_engine
    
    async def check_access(self, user: User, feature: str) -> bool:
        """
        Check if user has access to a specific feature based on subscription tier.
        """
        if feature == "memory":
            if user.subscription_tier == SubscriptionTiers.PREMIUM:
                return SubscriptionLimits.PREMIUM_MEMORY_ENABLED
            else:
                return SubscriptionLimits.FREE_MEMORY_ENABLED
        
        if feature == "session":
            if user.subscription_tier == SubscriptionTiers.PREMIUM:
                return True
            else:
                return user.sessions_used < SubscriptionLimits.FREE_SESSION_LIMIT
        
        return False
    
    async def increment_session_count(self, user: User, db: Session):
        """Increment user's session count."""
        user.sessions_used += 1
        db.commit()
        
        logger.info(f"User {user.id} session count: {user.sessions_used}")
    
    async def get_remaining_sessions(self, user: User) -> int:
        """Get remaining sessions for free tier users."""
        if user.subscription_tier == SubscriptionTiers.PREMIUM:
            return -1
        
        remaining = SubscriptionLimits.FREE_SESSION_LIMIT - user.sessions_used
        return max(0, remaining)
    
    async def generate_response(
        self,
        user: User,
        symptoms: list[str],
        db: Session
    ) -> dict:
        """
        Generate medical response based on user's subscription tier.
        """
        if not await self.check_access(user, "session"):
            return {
                "error": ErrorMessages.SESSION_LIMIT_REACHED,
                "sessions_used": user.sessions_used,
                "sessions_limit": SubscriptionLimits.FREE_SESSION_LIMIT
            }
        
        has_memory = await self.check_access(user, "memory")
        
        history = None
        if has_memory:
            logger.info(f"Retrieving memory for premium user {user.id}")
            relevant_history = await self.memory_engine.retrieve_relevant_history(
                user.id,
                " ".join(symptoms),
                db
            )
            
            if relevant_history:
                history = self.memory_engine.format_history_for_context(relevant_history)
        
        medical_response = await self.llm_service.generate_medical_response(
            symptoms=symptoms,
            history=history
        )
        
        session_id = await self.memory_engine.store_session(
            user_id=user.id,
            symptoms=symptoms,
            diagnosis=medical_response.dict(),
            db=db
        )
        
        await self.increment_session_count(user, db)
        
        remaining = await self.get_remaining_sessions(user)
        
        return {
            "response": medical_response,
            "session_id": session_id,
            "memory_used": has_memory,
            "sessions_remaining": remaining,
            "subscription_tier": user.subscription_tier
        }
    
    async def upgrade_to_premium(self, user: User, db: Session):
        """Upgrade user to premium subscription."""
        user.subscription_tier = SubscriptionTiers.PREMIUM
        user.sessions_used = 0
        db.commit()
        
        logger.info(f"User {user.id} upgraded to premium")


def create_subscription_service(
    llm_service: MedicalLLMService,
    memory_engine: MemoryEngine
) -> SubscriptionService:
    """Factory function to create subscription service."""
    return SubscriptionService(llm_service, memory_engine)
