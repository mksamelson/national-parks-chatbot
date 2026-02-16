"""
Groq API integration for LLM inference
"""
import os
from typing import List, Dict
from groq import Groq
import logging

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_MODEL = "llama-3.1-70b-versatile"  # Groq's Llama 3.1 70B
TEMPERATURE = 0.7
MAX_TOKENS = 1024


class LLMClient:
    """Wrapper for Groq API client"""

    def __init__(self):
        self.client = None
        self.model = DEFAULT_MODEL

    def connect(self):
        """Initialize Groq client"""
        if self.client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY must be set")

            # Strip whitespace/newlines from API key
            api_key = api_key.strip()

            logger.info("Connecting to Groq API")
            self.client = Groq(api_key=api_key)
            logger.info(f"✓ Connected to Groq (model: {self.model})")

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS
    ) -> str:
        """
        Generate text using Groq API

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        self.connect()

        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise

    def generate_with_context(
        self,
        question: str,
        context_chunks: List[Dict],
        system_prompt: str = None
    ) -> Dict:
        """
        Generate answer with retrieved context (RAG)

        Args:
            question: User question
            context_chunks: Retrieved context chunks
            system_prompt: Optional system prompt

        Returns:
            Dict with answer and sources
        """
        # Format context
        context_text = "\n\n".join([
            f"[Source {i+1} - {chunk['park_name']}]\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])

        # Build prompt
        user_prompt = f"""Context from National Parks Service:

{context_text}

User Question: {question}

Please answer the question based on the context provided. Include relevant source citations (e.g., "According to Source 1..."). If the context doesn't contain enough information to answer the question, say so."""

        # Generate answer
        answer = self.generate(
            prompt=user_prompt,
            system_prompt=system_prompt
        )

        # Extract sources
        sources = [
            {
                "park_name": chunk["park_name"],
                "park_code": chunk["park_code"],
                "url": chunk["source_url"],
                "score": chunk["score"]
            }
            for chunk in context_chunks
        ]

        return {
            "answer": answer,
            "sources": sources
        }


# Global instance
llm_client = LLMClient()
