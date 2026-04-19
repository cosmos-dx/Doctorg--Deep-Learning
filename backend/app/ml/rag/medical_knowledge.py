"""
Medical Knowledge Base from DoctorG dataset.
FAISS-based vector search for medical conditions and symptoms.
"""

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import pickle
from pathlib import Path
import logging

from app.core.constants import RAGConfig, ModelPaths
from app.core.errors import RAGError

logger = logging.getLogger(__name__)


class MedicalKnowledgeBase:
    """
    Medical knowledge base for condition and symptom retrieval.
    Indexes DoctorG dataset with FAISS.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.embedder: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []
        self.condition_map: Dict[int, Dict] = {}
        self._initialized = True
        
        logger.info("MedicalKnowledgeBase initialized")
    
    def load_embedder(self):
        """Load sentence transformer model."""
        if self.embedder is None:
            logger.info(f"Loading medical embedder: {ModelPaths.SENTENCE_TRANSFORMER}")
            self.embedder = SentenceTransformer(ModelPaths.SENTENCE_TRANSFORMER)
            logger.info("Medical embedder loaded")
    
    def load_index(self, index_path: str, metadata_path: str):
        """Load pre-built FAISS index and metadata."""
        try:
            self.load_embedder()
            
            index_file = Path(index_path)
            metadata_file = Path(metadata_path)
            
            if not index_file.exists():
                logger.warning(f"Medical knowledge index not found at {index_path}")
                return False
            
            self.index = faiss.read_index(str(index_file))
            
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata = data.get('metadata', [])
                    self.condition_map = data.get('condition_map', {})
            
            logger.info(f"Medical knowledge loaded: {len(self.metadata)} conditions indexed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load medical knowledge index: {e}")
            return False
    
    async def retrieve(
        self,
        query: str,
        symptoms: List[str],
        k: int = RAGConfig.TOP_K_RESULTS
    ) -> List[Dict]:
        """
        Retrieve relevant medical conditions based on query and symptoms.
        """
        if self.index is None or self.embedder is None:
            logger.warning("Medical knowledge base not initialized")
            return []
        
        try:
            combined_query = f"{query} {' '.join(symptoms)}"
            
            query_embedding = self.embedder.encode(combined_query)
            
            distances, indices = self.index.search(
                np.array([query_embedding]).astype('float32'),
                min(k, self.index.ntotal)
            )
            
            results = []
            
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.metadata):
                    condition = self.metadata[idx].copy()
                    condition['similarity_score'] = float(1 / (1 + distance))
                    
                    if condition['similarity_score'] >= RAGConfig.SIMILARITY_THRESHOLD:
                        results.append(condition)
            
            logger.info(f"Retrieved {len(results)} relevant conditions for query")
            return results
            
        except Exception as e:
            logger.error(f"Medical knowledge retrieval failed: {e}")
            raise RAGError(f"Failed to retrieve medical knowledge: {str(e)}")
    
    def get_condition_by_code(self, code: str) -> Optional[Dict]:
        """Get condition details by disease code."""
        for condition in self.metadata:
            if condition.get('code') == code:
                return condition
        return None
    
    def search_by_symptoms(self, symptoms: List[str], k: int = 5) -> List[Dict]:
        """Search for conditions matching specific symptoms."""
        if not self.metadata:
            return []
        
        symptom_set = set(s.lower().strip() for s in symptoms)
        
        matches = []
        for condition in self.metadata:
            condition_symptoms = condition.get('symptoms', '').lower().split(',')
            condition_symptom_set = set(s.strip() for s in condition_symptoms)
            
            overlap = len(symptom_set & condition_symptom_set)
            if overlap > 0:
                matches.append({
                    **condition,
                    'symptom_overlap': overlap,
                    'match_score': overlap / len(symptom_set)
                })
        
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        return matches[:k]


def create_medical_knowledge_base() -> MedicalKnowledgeBase:
    """Factory function to create MedicalKnowledgeBase instance."""
    return MedicalKnowledgeBase()
