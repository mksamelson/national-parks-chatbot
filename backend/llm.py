"""
LLM Generation - Groq API Integration

This module handles text generation using Groq's API for fast LLM inference.

Why Groq API instead of local models (e.g., Ollama, llama.cpp):
- Inference Speed: Groq provides blazing fast inference (~300 tokens/sec)
  - Local LLMs on CPU: ~10-20 tokens/sec (slow for user experience)
  - Groq API: ~300 tokens/sec (near-instant responses)
- Memory Efficiency: No local model loading saves RAM
  - Local Llama 70B quantized: ~40GB+ RAM (impossible on free tier)
  - Groq API calls: ~50MB overhead
- This allows the app to run on Render's 512MB free tier
- Quality: Llama 3.3 70B produces high-quality, contextual answers

Model Details:
- Model: llama-3.3-70b-versatile (Groq's hosted Llama 3.3 70B)
- Free Tier: 30 requests/minute, 14,400 tokens/minute
- Temperature: 0.7 (balanced creativity/accuracy)
- Max Tokens: 1024 (sufficient for detailed answers)

RAG Integration:
- generate_with_context(): Formats retrieved context chunks into prompt
- Includes source citations in prompts (e.g., "According to Source 1...")
- System prompts guide the model to cite sources and admit knowledge gaps

Trade-offs:
+ Pros: Ultra-fast inference, no local model management, free tier generous
- Cons: API dependency, rate limits (30 req/min), requires internet

Note: Llama 3.1 70B was deprecated; updated to Llama 3.3 70B in January 2026

Author: Built with Claude Code
Date: February 2026
"""
import os
from typing import List, Dict
from groq import Groq
import logging

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_MODEL = "llama-3.3-70b-versatile"  # Groq's Llama 3.3 70B (3.1 deprecated)
TEMPERATURE = 0
MAX_TOKENS = 1024


class LLMClient:
    """
    Wrapper for Groq API client providing LLM text generation

    Lazy-loads the connection to Groq API on first use.
    Handles prompt formatting for RAG (Retrieval Augmented Generation).
    """

    def __init__(self):
        """Initialize with no client (lazy loading)"""
        self.client = None
        self.model = DEFAULT_MODEL

    def connect(self):
        """
        Connect to Groq API (lazy initialization)

        Reads API key from environment variable and establishes connection.
        Strips whitespace from API key to handle copy/paste errors.

        Raises:
            ValueError: If GROQ_API_KEY not set in environment
        """
        if self.client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY must be set")

            # Strip whitespace/newlines from API key
            api_key = api_key.strip()

            self.client = Groq(api_key=api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS
    ) -> str:
        """
        Generate text using Groq API

        Sends a prompt to Groq's Llama 3.3 70B model and returns generated text.
        Automatically connects to Groq API if not already connected.

        Args:
            prompt: User prompt (the main question/instruction)
            system_prompt: Optional system prompt to set behavior/context
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
                        Default 0.7 balances accuracy and creativity
            max_tokens: Maximum tokens to generate (default: 1024)
                       Limits response length to control costs/latency

        Returns:
            str: Generated text response from the model

        Example:
            >>> llm = LLMClient()
            >>> answer = llm.generate("What is Yellowstone?")
            >>> print(answer)
            'Yellowstone is America's first national park...'
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
        system_prompt: str = None,
        conversation_history: List[Dict] = None,
        park_code: str = None
    ) -> Dict:
        """
        Generate answer with retrieved context (RAG - Retrieval Augmented Generation)

        This is the core RAG method that combines retrieved context with LLM generation.
        It formats the context chunks into a prompt, generates an answer using the LLM,
        and returns both the answer and source metadata.

        RAG Flow:
        1. Format context chunks into numbered sources with park names
        2. Build prompt with context + user question
        3. Include conversation history for multi-turn conversations
        4. Instruct model to cite sources (e.g., "According to Source 1...")
        5. Generate answer using Groq API
        6. Extract and return source metadata

        Args:
            question: User question about national parks
            context_chunks: List of retrieved context chunks from vector search
                           Each chunk is a Dict with keys:
                           - text: Document content
                           - park_name: Park name (e.g., "Yellowstone National Park")
                           - park_code: Park code (e.g., "yell")
                           - source_url: Original NPS document URL
                           - score: Similarity score (0-1)
            system_prompt: Optional system prompt to guide model behavior
            conversation_history: Optional list of previous messages for multi-turn conversations
                                Each message is a dict with 'role' and 'content' keys
            park_code: Optional park code indicating which park is being discussed
                      If provided, explicitly tells LLM the conversation is about this park

        Returns:
            Dict containing:
            - answer (str): Generated answer with source citations
            - sources (List[Dict]): List of source metadata, each with:
                - park_name: Park name
                - park_code: Park code
                - url: Source URL
                - score: Similarity score

        Example:
            >>> llm = LLMClient()
            >>> chunks = [
            ...     {"text": "Yellowstone has wolves...", "park_name": "Yellowstone",
            ...      "park_code": "yell", "source_url": "https://...", "score": 0.92}
            ... ]
            >>> result = llm.generate_with_context("What wildlife is in Yellowstone?", chunks)
            >>> print(result['answer'])
            'According to Source 1, Yellowstone has wolves...'
        """
        # Format context chunks into numbered sources
        # Each source includes park name for attribution
        # Example: "[Source 1 - Yellowstone National Park]\n<chunk text>"
        context_text = "\n\n".join([
            f"[Source {i+1} - {chunk['park_name']}]\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])

        # Build conversation context statement if we know which park is being discussed
        park_context_statement = ""
        if park_code and context_chunks:
            # Get park name from first context chunk (all should be from same park)
            park_name = context_chunks[0].get('park_name', 'National Park')
            park_context_statement = f"""IMPORTANT CONTEXT: The user is currently asking about {park_name}. All context provided below is specifically about {park_name}. When the user uses words like "there", "it", or "the park", they are referring to {park_name}.

"""

        # Build RAG prompt with context + question
        # Instructs model to use context and admit if context is insufficient
        user_prompt = f"""{park_context_statement}Context from National Parks Service:

{context_text}

User Question: {question}

Please answer the question based on the context provided above. Remember that all context is about {context_chunks[0].get('park_name', 'the park being discussed') if context_chunks else 'national parks'}."""

        # Build messages array for Groq API with conversation history
        self.connect()

        messages = []

        # 1. System prompt
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        # 2. Conversation history (if provided)
        if conversation_history:
            for msg in conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # 3. Current user prompt with context
        messages.append({
            "role": "user",
            "content": user_prompt
        })

        # Call Groq API with full conversation context
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )

            answer = response.choices[0].message.content

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise

        # Extract source metadata for frontend display
        # Provides attribution links and similarity scores
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
