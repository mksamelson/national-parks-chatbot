"""
Qdrant vector database client
"""
import os
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
import logging

logger = logging.getLogger(__name__)

# Configuration
COLLECTION_NAME = "national_parks"


class VectorDB:
    """Wrapper for Qdrant client"""

    def __init__(self):
        self.client = None
        self.collection_name = COLLECTION_NAME

    def connect(self):
        """Connect to Qdrant Cloud"""
        if self.client is None:
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")

            if not qdrant_url or not qdrant_api_key:
                raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set")

            # Strip whitespace/newlines from credentials
            qdrant_url = qdrant_url.strip()
            qdrant_api_key = qdrant_api_key.strip()

            logger.info(f"Connecting to Qdrant at {qdrant_url}")
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
            )
            logger.info("✓ Connected to Qdrant")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        park_code: str = None
    ) -> List[Dict]:
        """
        Search for similar documents

        Args:
            query_vector: Query embedding
            top_k: Number of results to return
            park_code: Optional filter by park code

        Returns:
            List of search results with text and metadata
        """
        self.connect()

        # Build filter if park_code provided
        query_filter = None
        if park_code:
            query_filter = {
                "must": [
                    {"key": "park_code", "match": {"value": park_code}}
                ]
            }

        # Search
        results: List[ScoredPoint] = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "text": result.payload.get("text", ""),
                "park_code": result.payload.get("park_code", ""),
                "park_name": result.payload.get("park_name", ""),
                "source_url": result.payload.get("source_url", ""),
                "chunk_id": result.payload.get("chunk_id", "")
            })

        return formatted_results


# Global instance
vector_db = VectorDB()
