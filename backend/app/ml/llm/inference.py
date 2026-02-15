"""
LLM inference service for medical response generation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import logging
from typing import List, Optional, AsyncGenerator
from app.core.constants import LLMConfig, ModelPaths
from app.models.schemas import MedicalResponse, ConfidenceLevel, Severity

logger = logging.getLogger(__name__)


class MedicalLLMService:
    """Service for medical LLM inference with structured output generation."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self._initialized = True
        
        logger.info(f"MedicalLLMService initialized on {self.device}")
    
    def load_model(self):
        """Load the fine-tuned medical LLM model."""
        if self.model is not None:
            return
        
        try:
            logger.info(f"Loading model from {ModelPaths.LLM_BASE_MODEL}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                ModelPaths.LLM_BASE_MODEL,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                ModelPaths.LLM_BASE_MODEL,
                load_in_8bit=LLMConfig.LOAD_IN_8BIT,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def create_prompt(self, symptoms: List[str], history: Optional[str] = None) -> str:
        """Create structured prompt for the model."""
        symptom_text = ", ".join(symptoms)
        
        prompt = f"""You are a medical AI assistant. Analyze the symptoms and provide a structured medical assessment.

Symptoms: {symptom_text}"""
        
        if history:
            prompt += f"\n\nPatient History:\n{history}"
        
        prompt += """

Provide your response in the following JSON format:
{
  "possible_conditions": ["condition1", "condition2"],
  "confidence_level": "low|medium|high",
  "follow_up_questions": ["question1", "question2"],
  "risk_factors": ["factor1", "factor2"],
  "suggested_tests": ["test1", "test2"],
  "lifestyle_recommendations": ["recommendation1", "recommendation2"],
  "severity": "mild|moderate|severe",
  "should_see_doctor": true|false,
  "reasoning": "brief explanation"
}

JSON Response:"""
        
        return prompt
    
    async def generate_medical_response(
        self,
        symptoms: List[str],
        history: Optional[str] = None
    ) -> MedicalResponse:
        """
        Generate structured medical response for given symptoms.
        """
        self.load_model()
        
        prompt = self.create_prompt(symptoms, history)
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=LLMConfig.MAX_LENGTH
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=LLMConfig.MAX_LENGTH,
                    temperature=LLMConfig.TEMPERATURE,
                    top_p=LLMConfig.TOP_P,
                    top_k=LLMConfig.TOP_K,
                    num_return_sequences=LLMConfig.NUM_RETURN_SEQUENCES,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            json_response = self._extract_json_from_response(generated_text)
            
            medical_response = self._parse_medical_response(json_response)
            
            return medical_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._get_fallback_response(symptoms)
    
    async def generate_stream(
        self,
        symptoms: List[str],
        history: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response for real-time updates.
        """
        self.load_model()
        
        prompt = self.create_prompt(symptoms, history)
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=LLMConfig.MAX_LENGTH
            ).to(self.device)
            
            generation_config = {
                "max_length": LLMConfig.MAX_LENGTH,
                "temperature": LLMConfig.TEMPERATURE,
                "top_p": LLMConfig.TOP_P,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            with torch.no_grad():
                for output in self.model.generate(
                    **inputs,
                    **generation_config,
                    return_dict_in_generate=True,
                    output_scores=True
                ):
                    token = self.tokenizer.decode(output, skip_special_tokens=True)
                    yield token
                    
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            yield json.dumps({"error": str(e)})
    
    def _extract_json_from_response(self, text: str) -> dict:
        """Extract JSON from model response."""
        try:
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_str = text[json_start:json_end]
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in response")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return {}
    
    def _parse_medical_response(self, json_data: dict) -> MedicalResponse:
        """Parse JSON into MedicalResponse model."""
        try:
            return MedicalResponse(
                possible_conditions=json_data.get('possible_conditions', []),
                confidence_level=ConfidenceLevel(json_data.get('confidence_level', 'medium')),
                follow_up_questions=json_data.get('follow_up_questions', []),
                risk_factors=json_data.get('risk_factors', []),
                suggested_tests=json_data.get('suggested_tests', []),
                lifestyle_recommendations=json_data.get('lifestyle_recommendations', []),
                severity=Severity(json_data.get('severity', 'moderate')),
                should_see_doctor=json_data.get('should_see_doctor', True),
                reasoning=json_data.get('reasoning')
            )
        except Exception as e:
            logger.error(f"Error parsing medical response: {e}")
            return self._get_fallback_response([])
    
    def _get_fallback_response(self, symptoms: List[str]) -> MedicalResponse:
        """Fallback response when model generation fails."""
        return MedicalResponse(
            possible_conditions=["Unable to determine"],
            confidence_level=ConfidenceLevel.LOW,
            follow_up_questions=[
                "Could you provide more details about your symptoms?",
                "How long have you been experiencing this?"
            ],
            risk_factors=symptoms[:3] if symptoms else [],
            suggested_tests=["Consult a healthcare professional"],
            lifestyle_recommendations=[
                "Monitor your symptoms",
                "Seek medical attention if symptoms worsen"
            ],
            severity=Severity.MODERATE,
            should_see_doctor=True,
            reasoning="Please consult a healthcare professional for proper diagnosis."
        )


def create_llm_service() -> MedicalLLMService:
    """Factory function to create LLM service instance."""
    return MedicalLLMService()
