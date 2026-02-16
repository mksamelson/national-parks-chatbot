"""
Embedding model using Cohere API (no local models = low memory!)
"""
import os
import cohere
from typing import List
import logging

logger = logging.getLogger(__name__)

# Model configuration
EMBEDDING_DIM = 1024  # Cohere embed-english-light-v3.0
COHERE_MODEL = "embed-english-light-v3.0"  # Free tier model


class EmbeddingModel:
    """Wrapper for Cohere API embeddings (memory-efficient!)"""

    def __init__(self):
        self.client = None

    def _get_client(self):
        """Get or create Cohere client"""
        if self.client is None:
            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                raise ValueError("COHERE_API_KEY environment variable not set")
            self.client = cohere.Client(api_key)
            logger.info(f"✓ Cohere client initialized (model: {COHERE_MODEL})")
        return self.client

    def encode(self, text: str) -> List[float]:
        """Generate embedding for a single text using Cohere API"""
        client = self._get_client()

        response = client.embed(
            texts=[text],
            model=COHERE_MODEL,
            input_type="search_query"
        )

        return response.embeddings[0]

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using Cohere API"""
        client = self._get_client()

        response = client.embed(
            texts=texts,
            model=COHERE_MODEL,
            input_type="search_document"
        )

        return response.embeddings


# Global instance (loaded on first use)
embedding_model = EmbeddingModel()
