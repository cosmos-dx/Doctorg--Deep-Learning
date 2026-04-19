"""
Ingest DoctorG medical dataset into FAISS index.
Processes CSV data and creates embeddings for medical conditions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / 'data'
CSV_PATH = DATA_DIR / 'doctorg_data.csv'
INDEX_PATH = DATA_DIR / 'faiss_indices' / 'medical_knowledge.index'
METADATA_PATH = DATA_DIR / 'faiss_indices' / 'medical_knowledge.metadata'

EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DIMENSION = 384


def load_and_preprocess_data(csv_path: Path) -> pd.DataFrame:
    """Load and clean DoctorG dataset."""
    logger.info(f"Loading data from {csv_path}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    logger.info(f"Loaded {len(df)} records")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    df = df.dropna(subset=['description'])
    
    df['description'] = df['description'].str.strip()
    df['description'] = df['description'].str.replace(r'\s+', ' ', regex=True)
    
    logger.info(f"After cleaning: {len(df)} records")
    
    return df


def create_embeddings(df: pd.DataFrame, model_name: str) -> np.ndarray:
    """Generate embeddings for medical descriptions."""
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    logger.info("Generating embeddings...")
    descriptions = df['description'].tolist()
    
    embeddings = []
    batch_size = 32
    
    for i in tqdm(range(0, len(descriptions), batch_size)):
        batch = descriptions[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings)
    
    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build FAISS index from embeddings."""
    logger.info("Building FAISS index...")
    
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    
    embeddings = embeddings.astype('float32')
    index.add(embeddings)
    
    logger.info(f"FAISS index built with {index.ntotal} vectors")
    
    return index


def prepare_metadata(df: pd.DataFrame) -> tuple:
    """Prepare metadata for storage."""
    logger.info("Preparing metadata...")
    
    metadata = []
    condition_map = {}
    
    for idx, row in df.iterrows():
        condition = {
            'code': str(row.get('code', '')),
            'name': str(row.get('name', '')),
            'symptoms': str(row.get('symptom', '')),
            'description': str(row.get('description', '')),
            'weight': float(row.get('weight', 0.0)) if pd.notna(row.get('weight')) else 0.0
        }
        
        metadata.append(condition)
        condition_map[idx] = condition
    
    logger.info(f"Prepared metadata for {len(metadata)} conditions")
    
    return metadata, condition_map


def save_index_and_metadata(
    index: faiss.Index,
    metadata: list,
    condition_map: dict,
    index_path: Path,
    metadata_path: Path
):
    """Save FAISS index and metadata to disk."""
    logger.info("Saving FAISS index and metadata...")
    
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS index saved to {index_path}")
    
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'metadata': metadata,
            'condition_map': condition_map
        }, f)
    
    logger.info(f"Metadata saved to {metadata_path}")


def main():
    """Main ingestion pipeline."""
    try:
        logger.info("Starting DoctorG data ingestion...")
        
        df = load_and_preprocess_data(CSV_PATH)
        
        embeddings = create_embeddings(df, EMBEDDING_MODEL)
        
        index = build_faiss_index(embeddings)
        
        metadata, condition_map = prepare_metadata(df)
        
        save_index_and_metadata(
            index,
            metadata,
            condition_map,
            INDEX_PATH,
            METADATA_PATH
        )
        
        logger.info("✓ DoctorG data ingestion completed successfully!")
        logger.info(f"  - Indexed: {len(metadata)} medical conditions")
        logger.info(f"  - Index: {INDEX_PATH}")
        logger.info(f"  - Metadata: {METADATA_PATH}")
        
    except Exception as e:
        logger.error(f"✗ Ingestion failed: {e}")
        raise


if __name__ == '__main__':
    main()
