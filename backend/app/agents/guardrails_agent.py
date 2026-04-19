"""
Guardrails agent for medical safety checks.
Detects emergency symptoms and prevents harmful advice.
"""

from typing import List, Optional
import re
import logging

from app.agents.base import BaseAgent, AgentContext, AgentResponse
from app.core.constants import (
    AgentTypes,
    GuardrailFlags,
    EmergencySymptoms,
    AgentPrompts
)

logger = logging.getLogger(__name__)


class GuardrailsAgent(BaseAgent):
    """
    Safety guardrails agent for medical consultation.
    Flags emergency symptoms and validates response safety.
    """
    
    def __init__(self, openai_service):
        super().__init__(
            agent_type=AgentTypes.GUARDRAILS,
            system_prompt=AgentPrompts.GUARDRAILS_SYSTEM,
            openai_service=openai_service,
            temperature=0.3
        )
        self.emergency_keywords = self._build_emergency_keywords()
    
    def _build_emergency_keywords(self) -> dict:
        """Build comprehensive emergency symptom keyword patterns."""
        return {
            GuardrailFlags.EMERGENCY: [
                r'\b(chest pain|heart attack|cardiac arrest)\b',
                r'\b(stroke|facial droop|arm weakness)\b',
                r'\b(severe bleeding|hemorrhage|blood loss)\b',
                r'\b(difficulty breathing|can\'?t breathe|gasping)\b',
                r'\b(loss of consciousness|unconscious|faint)\b',
                r'\b(severe headache|worst headache|sudden headache)\b',
                r'\b(seizure|convulsion|fitting)\b',
                r'\b(suicide|self harm|end my life)\b',
                r'\b(anaphylaxis|allergic shock|severe allergy)\b',
            ],
            GuardrailFlags.SEEK_IMMEDIATE_CARE: [
                r'\b(high fever|fever over 103|very high temperature)\b',
                r'\b(severe pain|unbearable pain|excruciating)\b',
                r'\b(vomiting blood|blood in stool|rectal bleeding)\b',
                r'\b(sudden confusion|disorientation|altered mental state)\b',
                r'\b(severe dehydration|not urinating)\b',
                r'\b(broken bone|fracture|severe injury)\b',
            ]
        }
    
    async def check_input_safety(self, context: AgentContext) -> AgentResponse:
        """
        Check user input for emergency symptoms and safety concerns.
        Returns flags and recommendations.
        """
        flags = []
        detected_emergencies = []
        
        combined_text = f"{context.user_message} {' '.join(context.symptoms)}".lower()
        
        for flag_type, patterns in self.emergency_keywords.items():
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    flags.append(flag_type)
                    match = re.search(pattern, combined_text, re.IGNORECASE)
                    if match:
                        detected_emergencies.append(match.group(0))
                    break
        
        if GuardrailFlags.EMERGENCY in flags:
            emergency_message = self._create_emergency_message(detected_emergencies)
            return self._create_response(
                content=emergency_message,
                guardrail_flags=[GuardrailFlags.EMERGENCY],
                metadata={
                    "detected_symptoms": detected_emergencies,
                    "action_required": "immediate_care"
                },
                confidence=1.0
            )
        
        elif GuardrailFlags.SEEK_IMMEDIATE_CARE in flags:
            urgent_message = self._create_urgent_message(detected_emergencies)
            return self._create_response(
                content=urgent_message,
                guardrail_flags=[GuardrailFlags.SEEK_IMMEDIATE_CARE],
                metadata={
                    "detected_symptoms": detected_emergencies,
                    "action_required": "urgent_care"
                },
                confidence=0.9
            )
        
        if self._is_non_medical_query(combined_text):
            flags.append(GuardrailFlags.OUT_OF_SCOPE)
            return self._create_response(
                content="I'm a medical consultation AI and can only help with health-related questions. Please ask about medical symptoms or health concerns.",
                guardrail_flags=flags,
                metadata={"action_required": "redirect"}
            )
        
        return self._create_response(
            content="No safety concerns detected. Proceeding with consultation.",
            guardrail_flags=[GuardrailFlags.CONSULT_DOCTOR],
            metadata={"safety_check": "passed"}
        )
    
    async def validate_output_safety(self, agent_response: str, context: AgentContext) -> AgentResponse:
        """
        Validate agent output for safety concerns.
        Checks for medication prescriptions and harmful advice.
        """
        flags = []
        
        medication_patterns = [
            r'\b(take|prescribe|dose|dosage|mg|ml)\s+\w+',
            r'\b(tablet|pill|capsule|injection|medication)\b',
            r'\b\d+\s*(mg|ml|mcg|units)\b'
        ]
        
        for pattern in medication_patterns:
            if re.search(pattern, agent_response, re.IGNORECASE):
                flags.append(GuardrailFlags.MEDICATION_WARNING)
                break
        
        if flags:
            warning_message = (
                "\n\n⚠️ IMPORTANT: This AI cannot prescribe medications or provide specific "
                "dosing instructions. Please consult a licensed healthcare provider for "
                "medication recommendations."
            )
            safe_response = agent_response + warning_message
        else:
            safe_response = agent_response
        
        disclaimer = (
            f"\n\n{AgentPrompts.MEDICAL_DISCLAIMER}"
        )
        
        final_response = safe_response + disclaimer
        
        return self._create_response(
            content=final_response,
            guardrail_flags=flags + [GuardrailFlags.CONSULT_DOCTOR],
            metadata={"validation": "passed", "disclaimer_added": True}
        )
    
    async def process(self, context: AgentContext) -> AgentResponse:
        """Process guardrails check."""
        return await self.check_input_safety(context)
    
    def _create_emergency_message(self, symptoms: List[str]) -> str:
        """Create emergency alert message."""
        symptoms_str = ", ".join(symptoms) if symptoms else "emergency symptoms"
        return (
            f"🚨 EMERGENCY ALERT 🚨\n\n"
            f"You have described symptoms that may indicate a medical emergency: {symptoms_str}\n\n"
            f"⚠️ DO NOT RELY ON THIS AI FOR EMERGENCY CARE\n\n"
            f"IMMEDIATELY:\n"
            f"• Call emergency services (911 in US) or go to the nearest emergency room\n"
            f"• If you're experiencing chest pain, stroke symptoms, or severe bleeding, call 911 NOW\n"
            f"• Do not drive yourself - call an ambulance\n\n"
            f"This AI consultation tool cannot replace emergency medical care. "
            f"Your symptoms require immediate professional evaluation."
        )
    
    def _create_urgent_message(self, symptoms: List[str]) -> str:
        """Create urgent care message."""
        symptoms_str = ", ".join(symptoms) if symptoms else "concerning symptoms"
        return (
            f"⚠️ URGENT CARE RECOMMENDED\n\n"
            f"You have described symptoms that require prompt medical attention: {symptoms_str}\n\n"
            f"Please:\n"
            f"• Visit an urgent care clinic or emergency room today\n"
            f"• Do not delay seeking medical care\n"
            f"• If symptoms worsen, call 911\n\n"
            f"While I can provide general information, your symptoms require "
            f"in-person evaluation by a healthcare professional."
        )
    
    def _is_non_medical_query(self, text: str) -> bool:
        """Check if query is non-medical."""
        non_medical_patterns = [
            r'\b(weather|recipe|sports|news|movie|game|music)\b',
            r'\b(code|programming|software|computer)\b',
            r'\b(joke|story|poem)\b',
            r'\bwho (is|are|was|were)\b',
            r'\bwhat (is|are) (the|a)\s+(capital|president|population)\b'
        ]
        
        for pattern in non_medical_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
