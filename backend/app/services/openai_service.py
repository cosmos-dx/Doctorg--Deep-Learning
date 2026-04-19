"""
OpenAI API integration service.
Handles all communication with OpenAI GPT models.
"""

import openai
from typing import Optional, AsyncIterator, List, Dict, Any
import logging
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from app.core.config import settings
from app.core.constants import OpenAIConfig
from app.core.errors import OpenAIError

logger = logging.getLogger(__name__)


class OpenAIService:
    """
    Service for interacting with OpenAI API.
    Provides text completion and streaming capabilities.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = OpenAIConfig.TIMEOUT
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.model = model or settings.OPENAI_MODEL
        self.timeout = timeout
        
        openai.api_key = self.api_key
        self.client = openai.AsyncOpenAI(api_key=self.api_key, timeout=self.timeout)
        
        logger.info(f"OpenAIService initialized with model: {self.model}")
    
    @retry(
        retry=retry_if_exception_type((openai.APITimeoutError, openai.APIConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = OpenAIConfig.TEMPERATURE,
        max_tokens: int = OpenAIConfig.MAX_TOKENS,
        model: Optional[str] = None
    ) -> str:
        """
        Get a completion from OpenAI API.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt to set context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Model to use (overrides default)
        
        Returns:
            Generated text response
        """
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            response = await self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            
            logger.info(
                f"OpenAI completion successful - "
                f"tokens: {response.usage.total_tokens}, "
                f"model: {response.model}"
            )
            
            return content
            
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {str(e)}")
            raise OpenAIError("Rate limit exceeded. Please try again later.", details={"error": str(e)})
        
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise OpenAIError("OpenAI API error occurred", details={"error": str(e)})
        
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI completion: {str(e)}")
            raise OpenAIError("Failed to complete request", details={"error": str(e)})
    
    @retry(
        retry=retry_if_exception_type((openai.APITimeoutError, openai.APIConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def complete_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = OpenAIConfig.TEMPERATURE,
        max_tokens: int = OpenAIConfig.MAX_TOKENS,
        model: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Get a streaming completion from OpenAI API.
        
        Yields response chunks as they arrive.
        """
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            stream = await self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            
            logger.info(f"OpenAI streaming completion successful - model: {self.model}")
            
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded in streaming: {str(e)}")
            raise OpenAIError("Rate limit exceeded. Please try again later.", details={"error": str(e)})
        
        except openai.APIError as e:
            logger.error(f"OpenAI API error in streaming: {str(e)}")
            raise OpenAIError("OpenAI API error occurred", details={"error": str(e)})
        
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI streaming: {str(e)}")
            raise OpenAIError("Failed to stream response", details={"error": str(e)})
    
    async def complete_with_context(
        self,
        messages: List[Dict[str, str]],
        temperature: float = OpenAIConfig.TEMPERATURE,
        max_tokens: int = OpenAIConfig.MAX_TOKENS,
        model: Optional[str] = None
    ) -> str:
        """
        Complete with full conversation context.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Model to use
        
        Returns:
            Generated text response
        """
        try:
            response = await self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in contextual completion: {str(e)}")
            raise OpenAIError("Failed to complete with context", details={"error": str(e)})
    
    async def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Get embedding vector for text.
        
        Args:
            text: Text to embed
            model: Embedding model to use
        
        Returns:
            Embedding vector
        """
        try:
            response = await self.client.embeddings.create(
                model=model,
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise OpenAIError("Failed to get embedding", details={"error": str(e)})


def create_openai_service() -> OpenAIService:
    """Factory function to create OpenAIService instance."""
    return OpenAIService()
