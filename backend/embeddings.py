"""
Embedding model loader and inference
"""
from sentence_transformers import SentenceTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model"""

    def __init__(self):
        self.model = None
        self.model_name = MODEL_NAME

    def load_model(self):
        """Load the embedding model (lazy loading)"""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✓ Model loaded (dimension: {EMBEDDING_DIM})")

    def encode(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        self.load_model()
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        self.load_model()
        embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
        return embeddings.tolist()


# Global instance (loaded on first use)
embedding_model = EmbeddingModel()
