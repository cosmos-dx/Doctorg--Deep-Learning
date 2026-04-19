"""
Knowledge base initialization service.
Loads FAISS indices and prepares RAG agents on startup.
"""

import logging
from pathlib import Path

from app.ml.rag.medical_knowledge import create_medical_knowledge_base
from app.ml.rag.pubmed_knowledge import create_pubmed_knowledge_base
from app.ml.rag.memory_engine import create_memory_engine
from app.core.config import settings

logger = logging.getLogger(__name__)


class KnowledgeBaseInitializer:
    """Initialize all knowledge bases and verify data integrity."""
    
    def __init__(self):
        self.medical_kb = None
        self.pubmed_kb = None
        self.memory_engine = None
        self.initialized = False
    
    async def initialize(self) -> dict:
        """
        Initialize all knowledge bases.
        Returns status dict with initialization results.
        """
        status = {
            "medical_kb": False,
            "pubmed_kb": False,
            "memory_engine": False,
            "errors": []
        }
        
        try:
            logger.info("Initializing medical knowledge base...")
            self.medical_kb = create_medical_knowledge_base()
            
            data_path = Path(settings.DATA_PATH)
            medical_index_path = data_path / 'faiss_indices' / 'medical_knowledge.index'
            medical_metadata_path = data_path / 'faiss_indices' / 'medical_knowledge.metadata'
            
            if medical_index_path.exists():
                success = self.medical_kb.load_index(
                    str(medical_index_path),
                    str(medical_metadata_path)
                )
                status["medical_kb"] = success
                if success:
                    logger.info("✓ Medical knowledge base loaded successfully")
                else:
                    logger.warning("⚠ Medical knowledge base failed to load")
            else:
                logger.warning(f"⚠ Medical knowledge index not found at {medical_index_path}")
                logger.info("  Run: python backend/scripts/ingest_doctorg_data.py")
                status["errors"].append("Medical knowledge index not found")
            
        except Exception as e:
            logger.error(f"✗ Medical knowledge base initialization failed: {e}")
            status["errors"].append(f"Medical KB error: {str(e)}")
        
        try:
            logger.info("Initializing PubMed knowledge base...")
            self.pubmed_kb = create_pubmed_knowledge_base()
            status["pubmed_kb"] = True
            logger.info("✓ PubMed knowledge base initialized (placeholder)")
            
        except Exception as e:
            logger.error(f"✗ PubMed knowledge base initialization failed: {e}")
            status["errors"].append(f"PubMed KB error: {str(e)}")
        
        try:
            logger.info("Initializing memory engine...")
            self.memory_engine = create_memory_engine()
            self.memory_engine.initialize_index()
            status["memory_engine"] = True
            logger.info("✓ Memory engine initialized")
            
        except Exception as e:
            logger.error(f"✗ Memory engine initialization failed: {e}")
            status["errors"].append(f"Memory engine error: {str(e)}")
        
        self.initialized = True
        
        logger.info("=" * 60)
        logger.info("Knowledge Base Initialization Summary:")
        logger.info(f"  Medical KB: {'✓' if status['medical_kb'] else '✗'}")
        logger.info(f"  PubMed KB: {'✓' if status['pubmed_kb'] else '✗'}")
        logger.info(f"  Memory Engine: {'✓' if status['memory_engine'] else '✗'}")
        if status["errors"]:
            logger.warning(f"  Errors: {len(status['errors'])}")
            for error in status["errors"]:
                logger.warning(f"    - {error}")
        logger.info("=" * 60)
        
        return status
    
    def get_medical_kb(self):
        """Get medical knowledge base instance."""
        return self.medical_kb
    
    def get_pubmed_kb(self):
        """Get PubMed knowledge base instance."""
        return self.pubmed_kb
    
    def get_memory_engine(self):
        """Get memory engine instance."""
        return self.memory_engine


_initializer = None


def get_knowledge_base_initializer() -> KnowledgeBaseInitializer:
    """Get singleton knowledge base initializer instance."""
    global _initializer
    if _initializer is None:
        _initializer = KnowledgeBaseInitializer()
    return _initializer


async def initialize_knowledge_bases() -> dict:
    """Initialize all knowledge bases on startup."""
    initializer = get_knowledge_base_initializer()
    return await initializer.initialize()
