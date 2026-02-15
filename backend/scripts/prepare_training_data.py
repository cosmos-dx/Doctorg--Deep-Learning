"""
Prepare training data for LLM fine-tuning.
Convert datasets to instruction-tuning format with structured JSON outputs.
"""

import pandas as pd
import json
from pathlib import Path
import logging
from typing import List, Dict
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingDataPreparator:
    """Prepare and format training data for medical LLM fine-tuning."""
    
    def __init__(self):
        self.instruction_template = """You are a medical AI assistant. Analyze the symptoms and provide a structured medical assessment.

Symptoms: {symptoms}

Provide your response in the following JSON format:
{{
  "possible_conditions": ["condition1", "condition2"],
  "confidence_level": "low|medium|high",
  "follow_up_questions": ["question1", "question2"],
  "risk_factors": ["factor1", "factor2"],
  "suggested_tests": ["test1", "test2"],
  "lifestyle_recommendations": ["recommendation1", "recommendation2"],
  "severity": "mild|moderate|severe",
  "should_see_doctor": true|false,
  "reasoning": "brief explanation"
}}"""
    
    def convert_csv_to_instruction_format(self, csv_path: str) -> List[Dict]:
        """Convert existing CSV data to instruction-tuning format."""
        logger.info(f"Converting CSV data from {csv_path}")
        
        df = pd.read_csv(csv_path)
        training_data = []
        
        grouped = df.groupby(['code', 'name'])
        
        for (code, disease_name), group in grouped:
            symptoms = group['symptom'].tolist()
            description = group['description'].iloc[0] if 'description' in group.columns else ""
            
            symptom_text = ", ".join(symptoms[:5])
            
            instruction = self.instruction_template.format(symptoms=symptom_text)
            
            response = {
                "possible_conditions": [disease_name],
                "confidence_level": "medium",
                "follow_up_questions": [
                    "How long have you experienced these symptoms?",
                    "Have the symptoms worsened over time?",
                    "Any other health conditions?"
                ],
                "risk_factors": symptoms[:3],
                "suggested_tests": ["Physical examination", "Blood test"],
                "lifestyle_recommendations": [
                    "Monitor symptoms",
                    "Maintain healthy diet",
                    "Stay hydrated"
                ],
                "severity": "moderate",
                "should_see_doctor": True,
                "reasoning": description[:200] if description else f"Based on symptoms, possible {disease_name}"
            }
            
            training_data.append({
                "instruction": instruction,
                "output": json.dumps(response, indent=2)
            })
        
        logger.info(f"Converted {len(training_data)} examples from CSV")
        return training_data
    
    def convert_pubmed_to_instruction_format(self, pubmed_data: List[Dict]) -> List[Dict]:
        """Convert PubMed abstracts to instruction-tuning format."""
        logger.info("Converting PubMed data")
        
        training_data = []
        
        for item in pubmed_data[:500]:
            title = item.get('title', '')
            abstract = item.get('abstract', '')
            
            if not title or not abstract or len(abstract) < 100:
                continue
            
            instruction = f"""Based on the following medical information, provide a structured analysis:

Title: {title}
Abstract: {abstract[:500]}

Provide insights about potential conditions, risk factors, and recommendations."""
            
            response = {
                "possible_conditions": item.get('keywords', [])[:3] if item.get('keywords') else ["General medical condition"],
                "confidence_level": "medium",
                "follow_up_questions": [
                    "What are your current symptoms?",
                    "How long have you had these symptoms?"
                ],
                "risk_factors": [],
                "suggested_tests": ["Consult medical literature", "Professional evaluation"],
                "lifestyle_recommendations": [
                    "Follow evidence-based practices",
                    "Consult healthcare provider"
                ],
                "severity": "mild",
                "should_see_doctor": True,
                "reasoning": abstract[:200]
            }
            
            training_data.append({
                "instruction": instruction,
                "output": json.dumps(response, indent=2)
            })
        
        logger.info(f"Converted {len(training_data)} PubMed examples")
        return training_data
    
    def convert_clinical_qa_to_instruction_format(self, clinical_qa: List[Dict]) -> List[Dict]:
        """Convert clinical QA data to instruction-tuning format."""
        logger.info("Converting Clinical QA data")
        
        training_data = []
        
        for item in clinical_qa[:500]:
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            if not question or not answer:
                continue
            
            instruction = f"""Medical Question: {question}

Provide a structured medical response based on this clinical question."""
            
            response = {
                "possible_conditions": ["Based on clinical question"],
                "confidence_level": "high",
                "follow_up_questions": [
                    "Any additional symptoms?",
                    "Medical history?"
                ],
                "risk_factors": [],
                "suggested_tests": ["Clinical evaluation"],
                "lifestyle_recommendations": [
                    "Follow medical advice",
                    "Regular check-ups"
                ],
                "severity": "moderate",
                "should_see_doctor": True,
                "reasoning": str(answer)[:200]
            }
            
            training_data.append({
                "instruction": instruction,
                "output": json.dumps(response, indent=2)
            })
        
        logger.info(f"Converted {len(training_data)} Clinical QA examples")
        return training_data
    
    def merge_and_shuffle(self, *datasets: List[Dict]) -> List[Dict]:
        """Merge multiple datasets and shuffle."""
        merged = []
        for dataset in datasets:
            merged.extend(dataset)
        
        random.shuffle(merged)
        logger.info(f"Merged and shuffled {len(merged)} total examples")
        
        return merged
    
    def save_training_data(self, data: List[Dict], output_path: str):
        """Save training data in JSONL format."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Saved {len(data)} training examples to {output_path}")
    
    def create_train_val_split(self, data: List[Dict], val_ratio: float = 0.1):
        """Split data into training and validation sets."""
        random.shuffle(data)
        
        split_idx = int(len(data) * (1 - val_ratio))
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        logger.info(f"Split: {len(train_data)} train, {len(val_data)} validation")
        
        return train_data, val_data


def main():
    """Main execution function."""
    preparator = TrainingDataPreparator()
    
    csv_data = preparator.convert_csv_to_instruction_format(
        "backend/data/doctorg_data.csv"
    )
    
    try:
        pubmed_df = pd.read_csv("backend/data/augmented/pubmed_data.csv")
        pubmed_data = preparator.convert_pubmed_to_instruction_format(
            pubmed_df.to_dict('records')
        )
    except FileNotFoundError:
        logger.warning("PubMed data not found, skipping")
        pubmed_data = []
    
    try:
        clinical_df = pd.read_csv("backend/data/augmented/clinical_qa_data.csv")
        clinical_data = preparator.convert_clinical_qa_to_instruction_format(
            clinical_df.to_dict('records')
        )
    except FileNotFoundError:
        logger.warning("Clinical QA data not found, skipping")
        clinical_data = []
    
    all_data = preparator.merge_and_shuffle(csv_data, pubmed_data, clinical_data)
    
    train_data, val_data = preparator.create_train_val_split(all_data, val_ratio=0.1)
    
    preparator.save_training_data(train_data, "backend/data/training/train.jsonl")
    preparator.save_training_data(val_data, "backend/data/training/val.jsonl")
    
    logger.info("Training data preparation completed!")


if __name__ == "__main__":
    main()
