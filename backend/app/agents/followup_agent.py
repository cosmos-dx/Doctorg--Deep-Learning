"""
Follow-up agent for gathering additional medical history and clarifying symptoms.
"""

from typing import AsyncIterator, List
import logging

from app.agents.base import BaseAgent, AgentContext, AgentResponse
from app.core.constants import AgentTypes, AgentPrompts

logger = logging.getLogger(__name__)


class FollowUpAgent(BaseAgent):
    """
    Follow-up agent for asking clarifying questions.
    Gathers additional information for accurate diagnosis.
    """
    
    def __init__(self, openai_service):
        super().__init__(
            agent_type=AgentTypes.FOLLOWUP,
            system_prompt=AgentPrompts.FOLLOWUP_SYSTEM,
            openai_service=openai_service,
            temperature=0.4
        )
    
    async def process(self, context: AgentContext) -> AgentResponse:
        """
        Generate relevant follow-up questions.
        """
        prompt = self._build_followup_prompt(context)
        
        response_text = await self._call_openai(prompt, max_tokens=300)
        
        questions = self._extract_questions(response_text)
        
        return self._create_response(
            content=response_text,
            metadata={
                "questions": questions,
                "question_count": len(questions),
                "agent_action": "followup_generated"
            },
            confidence=0.90
        )
    
    async def process_stream(self, context: AgentContext) -> AsyncIterator[str]:
        """Stream follow-up questions."""
        prompt = self._build_followup_prompt(context)
        
        async for chunk in self._call_openai_stream(prompt, max_tokens=300):
            yield chunk
    
    def _build_followup_prompt(self, context: AgentContext) -> str:
        """Build specialized follow-up prompt."""
        base_prompt = self._build_prompt(context)
        
        followup_instructions = """
Ask 2-3 focused follow-up questions. Number them. One sentence each.
Prioritize: duration, severity, and triggers.
"""
        
        return f"{base_prompt}\n\n{followup_instructions}"
    
    def _extract_questions(self, response: str) -> List[str]:
        """Extract individual questions from response."""
        questions = []
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            
            if line and ('?' in line):
                question = line.lstrip('0123456789.-) ').strip()
                
                if question and question.endswith('?'):
                    questions.append(question)
        
        return questions[:5]
