"""
RAG (Retrieval-Augmented Generation) agent.
Retrieves relevant medical knowledge from multiple sources.
"""

from typing import List, Dict, Optional
import logging
from sqlalchemy.orm import Session

from app.agents.base import BaseAgent, AgentContext, AgentResponse
from app.core.constants import AgentTypes, RAGConfig
from app.ml.rag.memory_engine import MemoryEngine

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    """
    RAG agent for knowledge retrieval.
    Queries multiple knowledge sources and aggregates results.
    """
    
    def __init__(
        self,
        openai_service,
        memory_engine: MemoryEngine,
        medical_kb=None,
        pubmed_kb=None
    ):
        super().__init__(
            agent_type=AgentTypes.RAG,
            system_prompt="You are a medical knowledge retrieval system.",
            openai_service=openai_service
        )
        self.memory_engine = memory_engine
        self.medical_kb = medical_kb
        self.pubmed_kb = pubmed_kb
    
    async def process(self, context: AgentContext) -> AgentResponse:
        """
        Retrieve relevant knowledge from all sources.
        """
        retrieved_context = await self.retrieve_all(
            query=context.user_message,
            symptoms=context.symptoms,
            user_id=context.user_id,
            db=context.metadata.get("db")
        )
        
        formatted_context = self._format_context(retrieved_context)
        
        return self._create_response(
            content=formatted_context,
            metadata={
                "sources": {
                    "user_history": len(retrieved_context.get("user_history", [])),
                    "medical_kb": len(retrieved_context.get("medical_kb", [])),
                    "pubmed": len(retrieved_context.get("pubmed", []))
                },
                "total_sources": sum([
                    len(retrieved_context.get("user_history", [])),
                    len(retrieved_context.get("medical_kb", [])),
                    len(retrieved_context.get("pubmed", []))
                ])
            }
        )
    
    async def retrieve_all(
        self,
        query: str,
        symptoms: List[str],
        user_id: Optional[str],
        db: Optional[Session] = None
    ) -> Dict[str, List[Dict]]:
        """
        Retrieve from all knowledge sources in parallel.
        """
        results = {
            "user_history": [],
            "medical_kb": [],
            "pubmed": []
        }
        
        if user_id and db:
            try:
                results["user_history"] = await self.memory_engine.retrieve_relevant_history(
                    user_id=user_id,
                    query=query,
                    db=db,
                    k=RAGConfig.TOP_K_RESULTS
                )
            except Exception as e:
                logger.error(f"Failed to retrieve user history: {e}")
        
        if self.medical_kb:
            try:
                results["medical_kb"] = await self.medical_kb.retrieve(
                    query=query,
                    symptoms=symptoms,
                    k=RAGConfig.TOP_K_RESULTS
                )
            except Exception as e:
                logger.error(f"Failed to retrieve from medical KB: {e}")
        
        if self.pubmed_kb:
            try:
                results["pubmed"] = await self.pubmed_kb.retrieve(
                    query=query,
                    k=RAGConfig.TOP_K_RESULTS
                )
            except Exception as e:
                logger.error(f"Failed to retrieve from PubMed KB: {e}")
        
        return results
    
    def _format_context(self, retrieved: Dict[str, List[Dict]]) -> str:
        """
        Format all retrieved knowledge into a context string.
        """
        context_parts = []
        
        user_history = retrieved.get("user_history", [])
        if user_history:
            context_parts.append("## Patient History")
            for idx, item in enumerate(user_history[:3], 1):
                symptoms = ", ".join(item.get("symptoms", []))
                diagnosis = item.get("diagnosis", {})
                conditions = diagnosis.get("possible_conditions", [])
                
                context_parts.append(
                    f"{idx}. Previous visit ({item.get('timestamp', 'Unknown')})\n"
                    f"   Symptoms: {symptoms}\n"
                    f"   Diagnosed conditions: {', '.join(conditions)}"
                )
            context_parts.append("")
        
        medical_kb = retrieved.get("medical_kb", [])
        if medical_kb:
            context_parts.append("## Medical Knowledge Base")
            for idx, item in enumerate(medical_kb[:5], 1):
                context_parts.append(
                    f"{idx}. {item.get('name', 'Unknown condition')}\n"
                    f"   Description: {item.get('description', '')}\n"
                    f"   Common symptoms: {item.get('symptoms', 'N/A')}"
                )
            context_parts.append("")
        
        pubmed = retrieved.get("pubmed", [])
        if pubmed:
            context_parts.append("## Medical Literature (PubMed)")
            for idx, item in enumerate(pubmed[:3], 1):
                context_parts.append(
                    f"{idx}. {item.get('title', 'Untitled')}\n"
                    f"   Abstract: {item.get('abstract', '')[:300]}...\n"
                    f"   PMID: {item.get('pmid', 'N/A')}"
                )
            context_parts.append("")
        
        if not context_parts:
            return "No relevant medical knowledge found in database."
        
        return "\n".join(context_parts)
    
    async def retrieve_user_history(
        self,
        user_id: str,
        query: str,
        db: Session
    ) -> List[Dict]:
        """Retrieve only user conversation history."""
        try:
            return await self.memory_engine.retrieve_relevant_history(
                user_id=user_id,
                query=query,
                db=db,
                k=RAGConfig.TOP_K_RESULTS
            )
        except Exception as e:
            logger.error(f"Failed to retrieve user history: {e}")
            return []
    
    async def retrieve_medical_knowledge(
        self,
        query: str,
        symptoms: List[str]
    ) -> List[Dict]:
        """Retrieve from medical knowledge base only."""
        if not self.medical_kb:
            return []
        
        try:
            return await self.medical_kb.retrieve(
                query=query,
                symptoms=symptoms,
                k=RAGConfig.TOP_K_RESULTS
            )
        except Exception as e:
            logger.error(f"Failed to retrieve medical knowledge: {e}")
            return []
    
    async def retrieve_pubmed(
        self,
        query: str
    ) -> List[Dict]:
        """Retrieve from PubMed knowledge base only."""
        if not self.pubmed_kb:
            return []
        
        try:
            return await self.pubmed_kb.retrieve(
                query=query,
                k=RAGConfig.TOP_K_RESULTS
            )
        except Exception as e:
            logger.error(f"Failed to retrieve from PubMed: {e}")
            return []
