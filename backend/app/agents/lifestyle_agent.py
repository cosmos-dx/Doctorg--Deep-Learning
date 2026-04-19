"""
Lifestyle and wellness agent for preventive health recommendations.
"""

from typing import AsyncIterator, List, Dict
import logging

from app.agents.base import BaseAgent, AgentContext, AgentResponse
from app.core.constants import AgentTypes, AgentPrompts

logger = logging.getLogger(__name__)


class LifestyleAgent(BaseAgent):
    """
    Lifestyle agent for wellness and preventive health recommendations.
    Provides evidence-based lifestyle modifications.
    """
    
    def __init__(self, openai_service):
        super().__init__(
            agent_type=AgentTypes.LIFESTYLE,
            system_prompt=AgentPrompts.LIFESTYLE_SYSTEM,
            openai_service=openai_service,
            temperature=0.7
        )
    
    async def process(self, context: AgentContext) -> AgentResponse:
        """
        Generate lifestyle and wellness recommendations.
        """
        prompt = self._build_lifestyle_prompt(context)
        
        response_text = await self._call_openai(prompt, max_tokens=600)
        
        categories = self._extract_categories(response_text)
        
        return self._create_response(
            content=response_text,
            metadata={
                "recommendation_categories": categories,
                "approach": "evidence_based",
                "agent_action": "recommendations_complete"
            },
            confidence=0.80
        )
    
    async def process_stream(self, context: AgentContext) -> AsyncIterator[str]:
        """Stream lifestyle recommendations."""
        prompt = self._build_lifestyle_prompt(context)
        
        async for chunk in self._call_openai_stream(prompt, max_tokens=600):
            yield chunk
    
    def _build_lifestyle_prompt(self, context: AgentContext) -> str:
        """Build specialized lifestyle prompt."""
        base_prompt = self._build_prompt(context)
        
        lifestyle_instructions = """
Based on the patient's symptoms and any details they shared (location, diet, lifestyle, etc.), 
provide 5-6 practical lifestyle recommendations as numbered points.

Cover the most relevant of these:
- **Diet & hydration** changes
- **Sleep & rest** improvements
- **Exercise / activity** adjustments
- **Stress management** techniques
- **Environmental** changes (climate, workspace, etc.)
- **Home remedies** that may help
- **When to see a doctor** (one line)

Keep each point to 1-2 sentences. Be specific and actionable.
"""
        
        return f"{base_prompt}\n\n{lifestyle_instructions}"
    
    def _extract_categories(self, response: str) -> List[str]:
        """Extract recommendation categories from response."""
        categories = []
        
        category_keywords = {
            "diet": ["diet", "food", "nutrition", "eating"],
            "exercise": ["exercise", "physical activity", "movement"],
            "sleep": ["sleep", "rest", "recovery"],
            "stress": ["stress", "mental health", "relaxation"],
            "environment": ["environment", "trigger", "exposure"],
            "prevention": ["prevent", "avoid", "reduce risk"]
        }
        
        response_lower = response.lower()
        
        for category, keywords in category_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                categories.append(category)
        
        return categories
