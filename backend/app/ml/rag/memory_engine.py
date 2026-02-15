"""
RAG Memory Engine using FAISS for vector search and PostgreSQL for metadata.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import pickle
from pathlib import Path
import logging
from sqlalchemy.orm import Session

from app.core.constants import RAGConfig, ModelPaths
from app.db.models import UserSession, Conversation

logger = logging.getLogger(__name__)


class MemoryEngine:
    """Memory engine for storing and retrieving user conversation history."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.embedder = None
        self.index = None
        self.index_to_db_mapping = {}
        self._initialized = True
        
        logger.info("MemoryEngine initialized")
    
    def load_embedder(self):
        """Load sentence transformer for embedding generation."""
        if self.embedder is None:
            logger.info(f"Loading embedder: {ModelPaths.SENTENCE_TRANSFORMER}")
            self.embedder = SentenceTransformer(ModelPaths.SENTENCE_TRANSFORMER)
            logger.info("Embedder loaded successfully")
    
    def initialize_index(self):
        """Initialize FAISS index."""
        if self.index is None:
            self.load_embedder()
            
            self.index = faiss.IndexFlatL2(RAGConfig.EMBEDDING_DIMENSION)
            
            index_path = Path(ModelPaths.FAISS_INDEX)
            if index_path.exists():
                self.load_index(str(index_path))
            
            logger.info("FAISS index initialized")
    
    def save_index(self, path: str):
        """Save FAISS index and mappings to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, path)
        
        with open(f"{path}.mapping", 'wb') as f:
            pickle.dump(self.index_to_db_mapping, f)
        
        logger.info(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """Load FAISS index and mappings from disk."""
        try:
            self.index = faiss.read_index(path)
            
            with open(f"{path}.mapping", 'rb') as f:
                self.index_to_db_mapping = pickle.load(f)
            
            logger.info(f"Index loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
    
    async def store_conversation(
        self,
        user_id: str,
        session_id: str,
        conversation: dict,
        db: Session
    ):
        """
        Store conversation in FAISS and PostgreSQL.
        """
        self.initialize_index()
        
        conversation_text = f"{conversation.get('role', '')}: {conversation.get('content', '')}"
        
        embedding = self.embedder.encode(conversation_text)
        
        faiss_index = self.index.ntotal
        self.index.add(np.array([embedding]))
        
        conversation_record = Conversation(
            user_id=user_id,
            session_id=session_id,
            role=conversation.get('role', 'user'),
            content=conversation.get('content', ''),
            embedding_vector_id=faiss_index
        )
        
        db.add(conversation_record)
        db.commit()
        
        self.index_to_db_mapping[faiss_index] = conversation_record.id
        
        logger.info(f"Stored conversation for user {user_id}, index {faiss_index}")
        
        return faiss_index
    
    async def store_session(
        self,
        user_id: str,
        symptoms: List[str],
        diagnosis: dict,
        db: Session
    ) -> str:
        """
        Store complete user session in database and FAISS.
        """
        self.initialize_index()
        
        session_text = f"Symptoms: {', '.join(symptoms)}"
        embedding = self.embedder.encode(session_text)
        
        faiss_index = self.index.ntotal
        self.index.add(np.array([embedding]))
        
        session_record = UserSession(
            user_id=user_id,
            symptoms=symptoms,
            diagnosis=diagnosis,
            faiss_index=faiss_index
        )
        
        db.add(session_record)
        db.commit()
        db.refresh(session_record)
        
        self.index_to_db_mapping[faiss_index] = session_record.id
        
        logger.info(f"Stored session {session_record.id} for user {user_id}")
        
        return session_record.id
    
    async def retrieve_relevant_history(
        self,
        user_id: str,
        query: str,
        db: Session,
        k: int = RAGConfig.TOP_K_RESULTS
    ) -> List[Dict]:
        """
        Retrieve top-k relevant conversations from user's history.
        """
        self.initialize_index()
        
        if self.index.ntotal == 0:
            logger.info("No history available")
            return []
        
        query_embedding = self.embedder.encode(query)
        
        distances, indices = self.index.search(
            np.array([query_embedding]),
            min(k, self.index.ntotal)
        )
        
        relevant_history = []
        
        for idx, distance in zip(indices[0], distances[0]):
            if distance < RAGConfig.SIMILARITY_THRESHOLD:
                continue
            
            if idx in self.index_to_db_mapping:
                record_id = self.index_to_db_mapping[idx]
                
                session = db.query(UserSession).filter(
                    UserSession.id == record_id,
                    UserSession.user_id == user_id
                ).first()
                
                if session:
                    relevant_history.append({
                        "session_id": session.id,
                        "symptoms": session.symptoms,
                        "diagnosis": session.diagnosis,
                        "timestamp": session.timestamp.isoformat(),
                        "similarity": float(1 / (1 + distance))
                    })
        
        logger.info(f"Retrieved {len(relevant_history)} relevant sessions for user {user_id}")
        
        return relevant_history
    
    async def get_user_session_history(
        self,
        user_id: str,
        db: Session,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get recent session history for a user.
        """
        sessions = db.query(UserSession).filter(
            UserSession.user_id == user_id
        ).order_by(
            UserSession.timestamp.desc()
        ).limit(limit).all()
        
        history = [
            {
                "session_id": session.id,
                "symptoms": session.symptoms,
                "diagnosis": session.diagnosis,
                "timestamp": session.timestamp.isoformat()
            }
            for session in sessions
        ]
        
        return history
    
    def format_history_for_context(self, history: List[Dict]) -> str:
        """Format retrieved history as context for LLM."""
        if not history:
            return ""
        
        context_parts = ["Previous medical history:"]
        
        for idx, item in enumerate(history, 1):
            symptoms = ", ".join(item.get('symptoms', []))
            diagnosis = item.get('diagnosis', {})
            conditions = diagnosis.get('possible_conditions', [])
            
            context_parts.append(
                f"\n{idx}. Date: {item.get('timestamp', 'Unknown')}\n"
                f"   Symptoms: {symptoms}\n"
                f"   Conditions: {', '.join(conditions)}"
            )
        
        return "\n".join(context_parts)


def create_memory_engine() -> MemoryEngine:
    """Factory function to create memory engine instance."""
    return MemoryEngine()
