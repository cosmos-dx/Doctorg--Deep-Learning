"""
Triage agent for initial symptom assessment and urgency evaluation.
"""

from typing import AsyncIterator
import logging

from app.agents.base import BaseAgent, AgentContext, AgentResponse
from app.core.constants import AgentTypes, AgentPrompts

logger = logging.getLogger(__name__)


class TriageAgent(BaseAgent):
    """
    Triage agent for initial patient assessment.
    Evaluates urgency and routes to appropriate care level.
    """
    
    def __init__(self, openai_service):
        super().__init__(
            agent_type=AgentTypes.TRIAGE,
            system_prompt=AgentPrompts.TRIAGE_SYSTEM,
            openai_service=openai_service,
            temperature=0.5
        )
    
    async def process(self, context: AgentContext) -> AgentResponse:
        """
        Assess symptoms and determine urgency level.
        """
        prompt = self._build_triage_prompt(context)
        
        response_text = await self._call_openai(prompt)
        
        urgency_level = self._extract_urgency(response_text)
        
        return self._create_response(
            content=response_text,
            metadata={
                "urgency_level": urgency_level,
                "agent_action": "triage_complete"
            },
            confidence=0.85,
            requires_followup=(urgency_level in ["moderate", "low"])
        )
    
    async def process_stream(self, context: AgentContext) -> AsyncIterator[str]:
        """Stream triage assessment."""
        prompt = self._build_triage_prompt(context)
        
        async for chunk in self._call_openai_stream(prompt):
            yield chunk
    
    def _build_triage_prompt(self, context: AgentContext) -> str:
        """Build specialized triage prompt."""
        base_prompt = self._build_prompt(context)
        
        triage_instructions = """
State the urgency level (Emergency/Urgent/Moderate/Low) and a one-sentence recommendation.
Keep it to 2-3 lines max.
"""
        
        return f"{base_prompt}\n\n{triage_instructions}"
    
    def _extract_urgency(self, response: str) -> str:
        """Extract urgency level from response."""
        response_lower = response.lower()
        
        if any(word in response_lower for word in ["emergency", "immediate", "911", "critical"]):
            return "emergency"
        elif any(word in response_lower for word in ["urgent", "today", "soon", "prompt"]):
            return "urgent"
        elif any(word in response_lower for word in ["moderate", "few days", "week"]):
            return "moderate"
        else:
            return "low"
