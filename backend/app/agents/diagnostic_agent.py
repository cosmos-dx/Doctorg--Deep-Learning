"""
Diagnostic agent for differential diagnosis and condition analysis.
"""

from typing import AsyncIterator, List, Dict
import logging
import json
import re

from app.agents.base import BaseAgent, AgentContext, AgentResponse
from app.core.constants import AgentTypes, AgentPrompts

logger = logging.getLogger(__name__)


class DiagnosticAgent(BaseAgent):
    """
    Diagnostic agent for analyzing symptoms and suggesting possible conditions.
    Provides differential diagnosis with explanations.
    """
    
    def __init__(self, openai_service):
        super().__init__(
            agent_type=AgentTypes.DIAGNOSTIC,
            system_prompt=AgentPrompts.DIAGNOSTIC_SYSTEM,
            openai_service=openai_service,
            temperature=0.6
        )
    
    async def process(self, context: AgentContext) -> AgentResponse:
        """
        Generate differential diagnosis based on symptoms.
        """
        prompt = self._build_diagnostic_prompt(context)
        
        response_text = await self._call_openai(prompt, max_tokens=600)
        
        conditions = self._extract_conditions(response_text)
        
        return self._create_response(
            content=response_text,
            metadata={
                "possible_conditions": conditions,
                "diagnostic_approach": "differential",
                "agent_action": "diagnosis_complete"
            },
            confidence=0.75,
            requires_followup=True
        )
    
    async def process_stream(self, context: AgentContext) -> AsyncIterator[str]:
        """Stream diagnostic analysis."""
        prompt = self._build_diagnostic_prompt(context)
        
        async for chunk in self._call_openai_stream(prompt, max_tokens=600):
            yield chunk
    
    def _build_diagnostic_prompt(self, context: AgentContext) -> str:
        """Build specialized diagnostic prompt."""
        base_prompt = self._build_prompt(context)
        
        diagnostic_instructions = """
Provide a concise assessment:

1. **Possible conditions** (2-3 most likely, one line each)
2. **Recommended next steps** (1-2 tests or actions)
3. **Red flags** to watch for (only if relevant, one line)

Use bullet points. Be brief and clear. No filler text.
"""
        
        return f"{base_prompt}\n\n{diagnostic_instructions}"
    
    def _extract_conditions(self, response: str) -> List[str]:
        """Extract condition names from diagnostic response."""
        conditions = []
        
        lines = response.split('\n')
        for line in lines:
            if re.match(r'^\d+\.', line.strip()) or line.strip().startswith('-'):
                condition_match = re.search(r'[:\-\.]?\s*([A-Z][a-zA-Z\s]+?)(?:\s*[\(\-:]|$)', line)
                if condition_match:
                    condition = condition_match.group(1).strip()
                    if len(condition) > 3 and len(condition) < 50:
                        conditions.append(condition)
        
        return conditions[:5]
