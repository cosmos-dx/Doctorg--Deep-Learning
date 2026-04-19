"""
Web agent for fetching medical datasets from PubMed and HuggingFace.
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
import pandas as pd
from datasets import load_dataset
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from app.core.constants import PubMedConfig, DatasetSources

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalDataAgent:
    """Agent for fetching and processing medical datasets."""
    
    def __init__(self, pubmed_api_key: Optional[str] = None, pubmed_email: Optional[str] = None):
        self.pubmed_api_key = pubmed_api_key
        self.pubmed_email = pubmed_email
        self.base_url = PubMedConfig.BASE_URL
        
    async def fetch_pubmed_abstracts(
        self,
        query: str,
        max_results: int = 1000
    ) -> List[Dict]:
        """
        Fetch medical abstracts from PubMed using Entrez API.
        """
        logger.info(f"Fetching PubMed abstracts for query: {query}")
        
        search_url = f"{self.base_url}{PubMedConfig.SEARCH_ENDPOINT}"
        fetch_url = f"{self.base_url}{PubMedConfig.FETCH_ENDPOINT}"
        
        search_params = {
            "db": PubMedConfig.DATABASE,
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        }
        
        if self.pubmed_email:
            search_params["email"] = self.pubmed_email
        if self.pubmed_api_key:
            search_params["api_key"] = self.pubmed_api_key
        
        abstracts = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=search_params) as response:
                    if response.status == 200:
                        data = await response.json()
                        id_list = data.get("esearchresult", {}).get("idlist", [])
                        
                        logger.info(f"Found {len(id_list)} PubMed articles")
                        
                        for i in range(0, len(id_list), 100):
                            batch_ids = id_list[i:i+100]
                            
                            fetch_params = {
                                "db": PubMedConfig.DATABASE,
                                "id": ",".join(batch_ids),
                                "retmode": "xml",
                                "rettype": "abstract"
                            }
                            
                            if self.pubmed_email:
                                fetch_params["email"] = self.pubmed_email
                            if self.pubmed_api_key:
                                fetch_params["api_key"] = self.pubmed_api_key
                            
                            async with session.get(fetch_url, params=fetch_params) as fetch_response:
                                if fetch_response.status == 200:
                                    xml_content = await fetch_response.text()
                                    batch_abstracts = self._parse_pubmed_xml(xml_content)
                                    abstracts.extend(batch_abstracts)
                                    
                                    await asyncio.sleep(0.5)
                    
                    else:
                        logger.error(f"PubMed API error: {response.status}")
        
        except Exception as e:
            logger.error(f"Error fetching PubMed abstracts: {e}")
        
        logger.info(f"Successfully fetched {len(abstracts)} abstracts")
        return abstracts
    
    def _parse_pubmed_xml(self, xml_content: str) -> List[Dict]:
        """Parse PubMed XML response."""
        abstracts = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for article in root.findall(".//PubmedArticle"):
                title_elem = article.find(".//ArticleTitle")
                abstract_elem = article.find(".//AbstractText")
                keywords_elem = article.findall(".//Keyword")
                
                if title_elem is not None and abstract_elem is not None:
                    abstracts.append({
                        "title": title_elem.text or "",
                        "abstract": abstract_elem.text or "",
                        "keywords": [k.text for k in keywords_elem if k.text] if keywords_elem else []
                    })
        
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
        
        return abstracts
    
    async def fetch_clinical_qa_datasets(self) -> List[Dict]:
        """
        Download clinical QA datasets from HuggingFace.
        """
        logger.info("Fetching clinical QA datasets from HuggingFace")
        
        all_data = []
        
        try:
            logger.info("Loading MedQA dataset...")
            medqa = load_dataset(DatasetSources.MEDQA, "med_qa_en", split="train")
            
            for item in medqa:
                all_data.append({
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "options": item.get("options", {}),
                    "source": "medqa"
                })
            
            logger.info(f"Loaded {len(all_data)} MedQA examples")
        
        except Exception as e:
            logger.error(f"Error loading MedQA: {e}")
        
        try:
            logger.info("Loading PubMedQA dataset...")
            pubmedqa = load_dataset(DatasetSources.PUBMEDQA, "pqa_labeled", split="train")
            
            for item in pubmedqa:
                all_data.append({
                    "question": item.get("question", ""),
                    "context": item.get("context", {}).get("contexts", []),
                    "answer": item.get("final_decision", ""),
                    "source": "pubmedqa"
                })
            
            logger.info(f"Total examples: {len(all_data)}")
        
        except Exception as e:
            logger.error(f"Error loading PubMedQA: {e}")
        
        return all_data
    
    async def augment_existing_dataset(self, csv_path: str) -> pd.DataFrame:
        """
        Combine existing CSV dataset with fetched medical data.
        """
        logger.info(f"Loading existing dataset from {csv_path}")
        existing_df = pd.read_csv(csv_path)
        
        logger.info("Fetching PubMed abstracts...")
        pubmed_data = await self.fetch_pubmed_abstracts(
            "medical symptoms diagnosis treatment",
            max_results=1000
        )
        
        logger.info("Fetching clinical QA datasets...")
        clinical_qa = await self.fetch_clinical_qa_datasets()
        
        pubmed_df = pd.DataFrame(pubmed_data)
        clinical_df = pd.DataFrame(clinical_qa)
        
        logger.info(f"Dataset sizes - Existing: {len(existing_df)}, PubMed: {len(pubmed_df)}, Clinical QA: {len(clinical_df)}")
        
        augmented_data = {
            "existing": existing_df,
            "pubmed": pubmed_df,
            "clinical_qa": clinical_df
        }
        
        return augmented_data
    
    def save_augmented_data(self, data: Dict, output_dir: str):
        """Save augmented datasets to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if "existing" in data:
            data["existing"].to_csv(output_path / "existing_data.csv", index=False)
        
        if "pubmed" in data and len(data["pubmed"]) > 0:
            data["pubmed"].to_csv(output_path / "pubmed_data.csv", index=False)
        
        if "clinical_qa" in data and len(data["clinical_qa"]) > 0:
            data["clinical_qa"].to_csv(output_path / "clinical_qa_data.csv", index=False)
        
        logger.info(f"Saved augmented datasets to {output_dir}")


async def main():
    """Main execution function."""
    agent = MedicalDataAgent()
    
    csv_path = "backend/data/doctorg_data.csv"
    output_dir = "backend/data/augmented"
    
    augmented_data = await agent.augment_existing_dataset(csv_path)
    agent.save_augmented_data(augmented_data, output_dir)
    
    logger.info("Dataset augmentation completed!")


if __name__ == "__main__":
    asyncio.run(main())
