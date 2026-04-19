"""
Agent orchestrator - coordinates multiple specialized agents.
"""

from typing import AsyncIterator, List, Dict, Optional
import logging
from sqlalchemy.orm import Session

from app.agents.base import AgentContext, AgentResponse
from app.agents.triage_agent import TriageAgent
from app.agents.diagnostic_agent import DiagnosticAgent
from app.agents.lifestyle_agent import LifestyleAgent
from app.agents.followup_agent import FollowUpAgent
from app.agents.guardrails_agent import GuardrailsAgent
from app.agents.rag_agent import RAGAgent
from app.agents.daily_advisor_agent import DailyAdvisorAgent
from app.core.constants import GuardrailFlags, AgentTypes, AgentPrompts
from app.core.errors import AgentError, EmergencyDetectedError

logger = logging.getLogger(__name__)

CLARITY_CLEAR = "CLEAR"
CLARITY_VAGUE = "VAGUE"


class AgentOrchestrator:
    """
    Orchestrates multiple agents for comprehensive medical consultation.
    Manages agent execution flow and aggregates results.
    """
    
    def __init__(
        self,
        openai_service,
        rag_agent: RAGAgent,
        memory_engine
    ):
        self.openai_service = openai_service
        self.memory_engine = memory_engine
        
        self.guardrails_agent = GuardrailsAgent(openai_service)
        self.rag_agent = rag_agent
        self.triage_agent = TriageAgent(openai_service)
        self.diagnostic_agent = DiagnosticAgent(openai_service)
        self.lifestyle_agent = LifestyleAgent(openai_service)
        self.followup_agent = FollowUpAgent(openai_service)
        self.daily_advisor_agent = DailyAdvisorAgent(openai_service)
        
        logger.info("AgentOrchestrator initialized with all agents")
    
    def _is_daily_advice_intent(self, message: str) -> bool:
        """
        Detect whether the user's message is a daily-lifestyle / wellness query
        that should be handled by DailyAdvisorAgent rather than the full
        diagnostic pipeline.
        """
        msg_lower = message.lower()
        return any(kw in msg_lower for kw in AgentPrompts.DAILY_INTENT_KEYWORDS)

    async def _classify_clarity(self, context: AgentContext) -> str:
        """
        Determine if the user's message is specific enough for a full consultation
        or if we need to ask clarifying questions first.
        """
        history_hint = ""
        if context.conversation_history:
            history_hint = (
                "\n\nConversation history exists — the patient may be answering "
                "previous follow-up questions. Consider this context."
            )
        
        prompt = f"Patient message: \"{context.user_message}\"{history_hint}"
        
        try:
            result = await self.openai_service.complete(
                prompt=prompt,
                system_prompt=AgentPrompts.CLARITY_CLASSIFIER,
                temperature=0.1,
                max_tokens=10
            )
            classification = result.strip().upper()
            if CLARITY_CLEAR in classification:
                return CLARITY_CLEAR
            return CLARITY_VAGUE
        except Exception as e:
            logger.error(f"Clarity classification failed: {e}")
            return CLARITY_CLEAR

    async def _is_ready_for_diagnosis(self, context: AgentContext) -> bool:
        """
        Determine if the consultation has enough detail to provide a safe differential diagnosis.
        """
        history_block = ""
        if context.conversation_history:
            lines = [f"{m.get('role','user').upper()}: {m.get('content','')}" for m in context.conversation_history]
            history_block = f"Conversation so far:\n" + "\n".join(lines) + "\n\n"
            
        prompt = f"{history_block}Current Patient message: \"{context.user_message}\""
        
        try:
            result = await self.openai_service.complete(
                prompt=prompt,
                system_prompt=AgentPrompts.DIAGNOSIS_READINESS_CLASSIFIER,
                temperature=0.1,
                max_tokens=10
            )
            classification = result.strip().upper()
            if "NOT_READY" in classification:
                return False
            return True
        except Exception as e:
            logger.error(f"Readiness classification failed: {e}")
            # Default to True so it falls back to full flow if this errors
            return True

    async def _generate_clarification(self, context: AgentContext) -> AsyncIterator[str]:
        """Stream a short clarification response for vague queries."""
        history_block = ""
        if context.conversation_history:
            recent = context.conversation_history[-4:]
            lines = [f"{m.get('role','user').upper()}: {m.get('content','')}" for m in recent]
            history_block = f"Conversation so far:\n" + "\n".join(lines) + "\n\n"

        prompt = f"{history_block}Patient says: \"{context.user_message}\""
        
        async for chunk in self.openai_service.complete_stream(
            prompt=prompt,
            system_prompt=AgentPrompts.CLARIFICATION_SYSTEM,
            temperature=0.6,
            max_tokens=500
        ):
            yield chunk

    async def process(
        self,
        user_message: str,
        symptoms: List[str],
        user_id: Optional[str],
        session_id: Optional[str],
        conversation_history: Optional[List[Dict]] = None,
        db: Optional[Session] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, AgentResponse]:
        """
        Process consultation through agent pipeline.
        Returns responses from all relevant agents.
        """
        ctx_metadata = {"db": db}
        if metadata:
            ctx_metadata.update(metadata)
            
        context = AgentContext(
            user_message=user_message,
            symptoms=symptoms,
            user_id=user_id,
            session_id=session_id,
            conversation_history=conversation_history or [],
            metadata=ctx_metadata
        )
        
        guardrails_response = await self.guardrails_agent.check_input_safety(context)
        
        if GuardrailFlags.EMERGENCY in guardrails_response.guardrail_flags:
            logger.warning(f"Emergency detected for user {user_id}")
            raise EmergencyDetectedError(
                message=guardrails_response.content,
                symptoms=guardrails_response.metadata.get("detected_symptoms", [])
            )
        
        if GuardrailFlags.OUT_OF_SCOPE in guardrails_response.guardrail_flags:
            return {
                AgentTypes.GUARDRAILS: guardrails_response
            }
        
        rag_response = await self.rag_agent.process(context)
        context.rag_context = rag_response.content
        
        triage_response = await self.triage_agent.process(context)
        
        urgency_level = triage_response.metadata.get("urgency_level", "moderate")
        
        responses = {
            AgentTypes.GUARDRAILS: guardrails_response,
            AgentTypes.RAG: rag_response,
            AgentTypes.TRIAGE: triage_response
        }
        
        if urgency_level in ["emergency", "urgent"]:
            responses[AgentTypes.DIAGNOSTIC] = await self.diagnostic_agent.process(context)
        else:
            is_ready = await self._is_ready_for_diagnosis(context)
            if not is_ready:
                followup_response = await self.followup_agent.process(context)
                responses.update({AgentTypes.FOLLOWUP: followup_response})
            else:
                diagnostic_response = await self.diagnostic_agent.process(context)
                lifestyle_response = await self.lifestyle_agent.process(context)
                
                responses.update({
                    AgentTypes.DIAGNOSTIC: diagnostic_response,
                    AgentTypes.LIFESTYLE: lifestyle_response
                })
        
        final_response = self._aggregate_responses(responses)
        
        validated_response = await self.guardrails_agent.validate_output_safety(
            agent_response=final_response,
            context=context
        )
        
        responses["final"] = validated_response
        
        if db and user_id and session_id:
            await self._store_conversation(
                user_id=user_id,
                session_id=session_id,
                user_message=user_message,
                agent_response=validated_response.content,
                responses=responses,
                db=db
            )
        
        return responses
    
    async def process_stream(
        self,
        user_message: str,
        symptoms: List[str],
        user_id: Optional[str],
        session_id: Optional[str],
        conversation_history: Optional[List[Dict]] = None,
        db: Optional[Session] = None,
        metadata: Optional[Dict] = None
    ) -> AsyncIterator[Dict[str, str]]:
        """
        Process consultation with streaming responses.
        Yields chunks as they're generated.
        """
        ctx_metadata = {"db": db}
        if metadata:
            ctx_metadata.update(metadata)
            
        context = AgentContext(
            user_message=user_message,
            symptoms=symptoms,
            user_id=user_id,
            session_id=session_id,
            conversation_history=conversation_history or [],
            metadata=ctx_metadata
        )
        
        yield {
            "type": "agent_start",
            "agent": AgentTypes.GUARDRAILS,
            "content": ""
        }
        
        guardrails_response = await self.guardrails_agent.check_input_safety(context)
        
        if GuardrailFlags.EMERGENCY in guardrails_response.guardrail_flags:
            yield {
                "type": "emergency",
                "agent": AgentTypes.GUARDRAILS,
                "content": guardrails_response.content
            }
            return
        
        if GuardrailFlags.OUT_OF_SCOPE in guardrails_response.guardrail_flags:
            yield {
                "type": "out_of_scope",
                "agent": AgentTypes.GUARDRAILS,
                "content": guardrails_response.content
            }
            return
        
        # ── Daily Advisor fast-path ───────────────────────────────────────
        if self._is_daily_advice_intent(user_message):
            yield {
                "type": "agent_start",
                "agent": AgentTypes.DAILY_ADVISOR,
                "content": "### Daily Wellness Advice\n\n"
            }
            async for chunk in self.daily_advisor_agent.process_stream(context):
                yield {
                    "type": "content",
                    "agent": AgentTypes.DAILY_ADVISOR,
                    "content": chunk
                }
            yield {
                "type": "disclaimer",
                "content": f"\n\n---\n{AgentPrompts.MEDICAL_DISCLAIMER}"
            }
            yield {"type": "complete", "content": ""}
            return

        clarity = await self._classify_clarity(context)
        
        if clarity == CLARITY_VAGUE:
            yield {
                "type": "agent_start",
                "agent": AgentTypes.FOLLOWUP,
                "content": ""
            }
            async for chunk in self._generate_clarification(context):
                yield {
                    "type": "content",
                    "agent": AgentTypes.FOLLOWUP,
                    "content": chunk
                }
            yield {
                "type": "complete",
                "content": ""
            }
            return

        
        yield {
            "type": "agent_start",
            "agent": AgentTypes.RAG,
            "content": ""
        }
        
        rag_response = await self.rag_agent.process(context)
        context.rag_context = rag_response.content
        
        yield {
            "type": "agent_complete",
            "agent": AgentTypes.RAG,
            "metadata": rag_response.metadata
        }
        
        is_ready = await self._is_ready_for_diagnosis(context)
        
        if not is_ready:
            yield {
                "type": "agent_start",
                "agent": AgentTypes.FOLLOWUP,
                "content": "### Clarifying Questions\n\n"
            }
            
            async for chunk in self.followup_agent.process_stream(context):
                yield {
                    "type": "content",
                    "agent": AgentTypes.FOLLOWUP,
                    "content": chunk
                }
            
            yield {
                "type": "disclaimer",
                "content": f"\n\n---\n{AgentPrompts.MEDICAL_DISCLAIMER}"
            }
            
            yield {
                "type": "complete",
                "content": ""
            }
            return
        
        yield {
            "type": "agent_start",
            "agent": AgentTypes.DIAGNOSTIC,
            "content": "### Assessment\n\n"
        }
        
        async for chunk in self.diagnostic_agent.process_stream(context):
            yield {
                "type": "content",
                "agent": AgentTypes.DIAGNOSTIC,
                "content": chunk
            }
        
        yield {
            "type": "agent_start",
            "agent": AgentTypes.LIFESTYLE,
            "content": "\n\n### Suggestions\n\n"
        }
        
        async for chunk in self.lifestyle_agent.process_stream(context):
            yield {
                "type": "content",
                "agent": AgentTypes.LIFESTYLE,
                "content": chunk
            }
        
        yield {
            "type": "disclaimer",
            "content": f"\n\n---\n{AgentPrompts.MEDICAL_DISCLAIMER}"
        }
        
        yield {
            "type": "complete",
            "content": ""
        }
    
    def _aggregate_responses(self, responses: Dict[str, AgentResponse]) -> str:
        """Aggregate all agent responses into cohesive output."""
        sections = []
        
        if AgentTypes.DIAGNOSTIC in responses:
            sections.append(f"### Assessment\n\n{responses[AgentTypes.DIAGNOSTIC].content}")
        
        if AgentTypes.LIFESTYLE in responses:
            sections.append(f"### Suggestions\n\n{responses[AgentTypes.LIFESTYLE].content}")
        
        if AgentTypes.FOLLOWUP in responses:
            sections.append(f"---\n\n{responses[AgentTypes.FOLLOWUP].content}")
        
        return "\n\n".join(sections)
    
    async def _store_conversation(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        agent_response: str,
        responses: Dict[str, AgentResponse],
        db: Session
    ):
        """Store conversation in memory for future retrieval."""
        try:
            await self.memory_engine.store_conversation(
                user_id=user_id,
                session_id=session_id,
                conversation={
                    "role": "user",
                    "content": user_message
                },
                db=db
            )
            
            await self.memory_engine.store_conversation(
                user_id=user_id,
                session_id=session_id,
                conversation={
                    "role": "assistant",
                    "content": agent_response,
                    "metadata": {
                        "agents_used": list(responses.keys()),
                        "urgency_level": responses.get(AgentTypes.TRIAGE, AgentResponse(
                            agent_type="", content=""
                        )).metadata.get("urgency_level")
                    }
                },
                db=db
            )
            
            logger.info(f"Stored conversation for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to store conversation: {e}")
