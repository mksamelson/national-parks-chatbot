"""
Embedding Model - Cohere API Integration

This module handles text-to-vector embedding generation using Cohere's API.

Why Cohere API instead of local models (e.g., sentence-transformers):
- Memory Efficiency: No local model loading saves ~300MB RAM
  - sentence-transformers + PyTorch = ~400-500MB
  - Cohere API calls = ~50MB overhead
- This allows the app to run on Render's 512MB free tier
- Fast startup: App binds to port in <2 seconds (critical for Render health checks)
- Quality: Cohere's embed-english-v3.0 produces high-quality 1024-dim embeddings

Model Details:
- Model: embed-english-v3.0
- Dimension: 1024
- Free Tier: 100 API calls/minute, unlimited total calls
- Input Types: search_query (for questions), search_document (for corpus)

Trade-offs:
+ Pros: Low memory, fast startup, no model management
- Cons: API dependency, slight latency (~100-200ms per call)

Author: Built with Claude Code
Date: February 2026
"""
import os
import cohere
from typing import List
import logging

logger = logging.getLogger(__name__)

# Model configuration
EMBEDDING_DIM = 1024  # Cohere embed-english-v3.0 dimension
COHERE_MODEL = "embed-english-v3.0"  # Full model (1024-dim, not lite 384-dim)


class EmbeddingModel:
    """
    Wrapper for Cohere API embeddings

    Provides a simple interface for generating embeddings using Cohere's API.
    Lazy-loads the API client on first use to save resources.
    """

    def __init__(self):
        """Initialize with no client (lazy loading)"""
        self.client = None

    def _get_client(self):
        """
        Get or create Cohere API client (lazy initialization)

        Returns:
            cohere.ClientV2: Initialized Cohere client

        Raises:
            ValueError: If COHERE_API_KEY not set in environment
        """
        if self.client is None:
            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                raise ValueError("COHERE_API_KEY environment variable not set")

            # Strip whitespace/newlines from API key to handle copy/paste errors
            api_key = api_key.strip()

            # Use Cohere v5 ClientV2 API (not deprecated Client)
            self.client = cohere.ClientV2(api_key=api_key)
            logger.info(f"✓ Cohere client initialized (model: {COHERE_MODEL})")

        return self.client

    def encode(self, text: str) -> List[float]:
        """
        Generate embedding for a single text (used for user queries)

        Args:
            text: The text to embed (typically a user question)

        Returns:
            List[float]: 1024-dimension embedding vector

        Example:
            >>> model = EmbeddingModel()
            >>> embedding = model.encode("What wildlife is in Yellowstone?")
            >>> len(embedding)
            1024
        """
        client = self._get_client()

        # Use input_type="search_query" for questions
        response = client.embed(
            texts=[text],
            model=COHERE_MODEL,
            input_type="search_query",  # Optimized for queries
            embedding_types=["float"]   # Get float embeddings
        )

        return response.embeddings.float_[0]

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (used for document corpus)

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of 1024-dimension embedding vectors

        Note:
            Uses input_type="search_document" for corpus embeddings
        """
        client = self._get_client()

        # Use input_type="search_document" for documents
        response = client.embed(
            texts=texts,
            model=COHERE_MODEL,
            input_type="search_document",  # Optimized for documents
            embedding_types=["float"]       # Get float embeddings
        )

        return response.embeddings.float_


# Global instance (loaded on first use)
embedding_model = EmbeddingModel()
