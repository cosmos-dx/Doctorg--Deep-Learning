"""
Base agent class for all specialized medical agents.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, AsyncIterator
from pydantic import BaseModel
import logging

from app.core.constants import AgentTypes, OpenAIConfig
from app.core.errors import AgentError

logger = logging.getLogger(__name__)


class AgentResponse(BaseModel):
    """Standard response format from agents."""
    agent_type: str
    content: str
    metadata: Dict[str, Any] = {}
    confidence: Optional[float] = None
    requires_followup: bool = False
    guardrail_flags: List[str] = []


class AgentContext(BaseModel):
    """Context passed between agents."""
    user_message: str
    symptoms: List[str] = []
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_history: List[Dict[str, str]] = []
    rag_context: Optional[str] = None
    metadata: Dict[str, Any] = {}


class BaseAgent(ABC):
    """
    Abstract base class for all medical agents.
    Provides common interface and OpenAI integration.
    """
    
    def __init__(
        self,
        agent_type: str,
        system_prompt: str,
        openai_service: Any,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        self.agent_type = agent_type
        self.system_prompt = system_prompt
        self.openai_service = openai_service
        self.temperature = temperature or OpenAIConfig.TEMPERATURE
        self.max_tokens = max_tokens or OpenAIConfig.MAX_TOKENS
        self.logger = logging.getLogger(f"{__name__}.{agent_type}")
    
    @abstractmethod
    async def process(self, context: AgentContext) -> AgentResponse:
        """
        Process the agent's task and return a response.
        Must be implemented by each specialized agent.
        """
        pass
    
    async def process_stream(self, context: AgentContext) -> AsyncIterator[str]:
        """
        Process the agent's task and stream the response.
        Default implementation calls process() and yields the full response.
        Override for true streaming support.
        """
        response = await self.process(context)
        yield response.content
    
    def _build_prompt(self, context: AgentContext) -> str:
        """
        Build the full prompt for the agent including context.
        Can be overridden by subclasses for custom prompt building.
        """
        prompt_parts = []
        
        if context.conversation_history:
            prompt_parts.append("Previous conversation:")
            for msg in context.conversation_history[-3:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"{role.upper()}: {content}")
            prompt_parts.append("")
        
        if context.rag_context:
            prompt_parts.append(f"Relevant medical knowledge:\n{context.rag_context}\n")
        
        if context.symptoms:
            symptoms_str = ", ".join(context.symptoms)
            prompt_parts.append(f"Reported symptoms: {symptoms_str}\n")
        
        if context.metadata.get("health_profile"):
            profile = context.metadata["health_profile"]
            prompt_parts.append(
                f"Patient Health Profile:\n"
                f"- Age: {profile.get('age')}\n"
                f"- Gender: {profile.get('gender')}\n"
                f"- Lifestyle: {profile.get('lifestyle_notes')}\n"
                f"- Conditions: {profile.get('chronic_conditions')}\n"
                f"- Allergies: {profile.get('allergies')}\n"
            )
            
        if context.metadata.get("lab_history"):
            labs = context.metadata["lab_history"]
            prompt_parts.append(f"Recent Lab Biomarkers:\n" + "\n".join([f"  - {lab}" for lab in labs]) + "\n")
        
        prompt_parts.append(f"Current message: {context.user_message}")
        
        return "\n".join(prompt_parts)
    
    async def _call_openai(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Call OpenAI API with the given prompt.
        Handles errors and retries.
        """
        try:
            response = await self.openai_service.complete(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )
            return response
        except Exception as e:
            self.logger.error(f"OpenAI API call failed for {self.agent_type}: {str(e)}")
            raise AgentError(
                message=f"Failed to get response from {self.agent_type} agent",
                agent_type=self.agent_type,
                details={"error": str(e)}
            )
    
    async def _call_openai_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[str]:
        """
        Call OpenAI API with streaming.
        Yields response chunks as they arrive.
        """
        try:
            async for chunk in self.openai_service.complete_stream(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            ):
                yield chunk
        except Exception as e:
            self.logger.error(f"OpenAI streaming failed for {self.agent_type}: {str(e)}")
            raise AgentError(
                message=f"Failed to stream response from {self.agent_type} agent",
                agent_type=self.agent_type,
                details={"error": str(e)}
            )
    
    def _create_response(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
        requires_followup: bool = False,
        guardrail_flags: Optional[List[str]] = None
    ) -> AgentResponse:
        """Create a standardized agent response."""
        return AgentResponse(
            agent_type=self.agent_type,
            content=content,
            metadata=metadata or {},
            confidence=confidence,
            requires_followup=requires_followup,
            guardrail_flags=guardrail_flags or []
        )
