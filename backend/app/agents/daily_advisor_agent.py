"""
Daily Advisor Agent — Provides personalised, everyday lifestyle and wellness guidance.
This agent focuses on actionable day-to-day recommendations calibrated to the
user's health profile (age, weight, chronic conditions, medications).

IMPORTANT: All advice is for general wellness guidance only, not medical treatment.
"""

from typing import AsyncIterator
import logging

from app.agents.base import BaseAgent, AgentContext, AgentResponse
from app.core.constants import AgentTypes


logger = logging.getLogger(__name__)


DAILY_ADVISOR_SYSTEM = """You are DoctorG's Daily Wellness Advisor — a warm, knowledgeable health companion
who gives practical, realistic advice for day-to-day health management.

Your role is to provide personalised LIFESTYLE and WELLNESS guidance, NOT medical diagnosis.

Guidelines:
- Focus on sleep, nutrition, hydration, exercise, stress management, and mental health
- Calibrate advice to the patient's health profile if provided (age, conditions, medications)
- Be specific and actionable — "Drink 2.5L of water today" not "stay hydrated"
- Group advice into clear categories: Morning Routine / Nutrition / Movement / Sleep / Mindset
- Use a friendly, encouraging tone — like a knowledgeable friend, not a clinical report
- Always close with: "Remember to discuss any health concerns with your doctor."
- Keep response to 4-6 bullet points per section, concise

DO NOT:
- Prescribe medications or dosages
- Make definitive diagnoses
- Contradict medications the user is taking
- Be alarmist

ALWAYS include at the end:
"⚕️ This is personalised wellness guidance — not a substitute for professional medical advice."
"""


class DailyAdvisorAgent(BaseAgent):
    """
    Specialist agent for day-to-day lifestyle and wellness consultancy.
    Receives health profile metadata in context.metadata['health_profile'].
    """

    def __init__(self, openai_service):
        super().__init__(
            agent_type=AgentTypes.DAILY_ADVISOR,
            system_prompt=DAILY_ADVISOR_SYSTEM,
            openai_service=openai_service,
            temperature=0.65,
            max_tokens=900
        )
        logger.info("DailyAdvisorAgent initialised")

    def _build_daily_prompt(self, context: AgentContext) -> str:
        """Build a personalised prompt incorporating the user's health profile."""
        parts = []

        health_profile = context.metadata.get("health_profile")
        if health_profile:
            profile_lines = ["Patient Health Profile:"]
            if health_profile.get("age"):
                profile_lines.append(f"  Age: {health_profile['age']}")
            if health_profile.get("gender"):
                profile_lines.append(f"  Gender: {health_profile['gender']}")
            if health_profile.get("weight_kg") and health_profile.get("height_cm"):
                bmi = health_profile["weight_kg"] / ((health_profile["height_cm"] / 100) ** 2)
                profile_lines.append(
                    f"  BMI: {bmi:.1f} "
                    f"(Weight: {health_profile['weight_kg']}kg, Height: {health_profile['height_cm']}cm)"
                )
            if health_profile.get("chronic_conditions"):
                conditions = ", ".join(health_profile["chronic_conditions"])
                profile_lines.append(f"  Chronic conditions: {conditions}")
            if health_profile.get("current_medications"):
                meds = ", ".join(health_profile["current_medications"])
                profile_lines.append(f"  Current medications: {meds}")
            if health_profile.get("allergies"):
                allergies = ", ".join(health_profile["allergies"])
                profile_lines.append(f"  Allergies: {allergies}")
            parts.append("\n".join(profile_lines))

        if context.conversation_history:
            recent = context.conversation_history[-4:]
            history_lines = ["Recent conversation:"]
            for msg in recent:
                history_lines.append(f"  {msg.get('role','user').upper()}: {msg.get('content','')}")
            parts.append("\n".join(history_lines))

        if context.symptoms:
            parts.append(f"Current concerns/symptoms: {', '.join(context.symptoms)}")

        parts.append(f"Patient message: {context.user_message}")

        return "\n\n".join(parts)

    async def process(self, context: AgentContext) -> AgentResponse:
        """Process a daily wellness query and return structured advice."""
        prompt = self._build_daily_prompt(context)
        try:
            content = await self._call_openai(prompt)
            return self._create_response(
                content=content,
                metadata={"agent": AgentTypes.DAILY_ADVISOR},
                confidence=0.8,
                requires_followup=False
            )
        except Exception as e:
            logger.error(f"DailyAdvisorAgent failed: {e}")
            return self._create_response(
                content=(
                    "I'm having trouble generating your daily wellness advice right now. "
                    "Please try again in a moment."
                ),
                metadata={"error": str(e)}
            )

    async def process_stream(self, context: AgentContext) -> AsyncIterator[str]:
        """Stream daily wellness advice chunk by chunk."""
        prompt = self._build_daily_prompt(context)
        try:
            async for chunk in self._call_openai_stream(prompt):
                yield chunk
        except Exception as e:
            logger.error(f"DailyAdvisorAgent stream failed: {e}")
            yield "⚠️ Unable to stream wellness advice right now. Please retry."
