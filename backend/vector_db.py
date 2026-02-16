"""
Vector Database - Qdrant Cloud Integration

This module handles vector similarity search using Qdrant Cloud.

Vector Database Details:
- Service: Qdrant Cloud (managed vector database)
- Collection: national_parks
- Vector Dimension: 1024 (matches Cohere embed-english-v3.0)
- Distance Metric: Cosine similarity
- Free Tier: 1GB storage (~1M vectors)

Data Structure:
Each point in the collection contains:
- vector: 1024-dim embedding (from Cohere)
- payload: {
    text: Document chunk text
    park_code: 4-letter park code (e.g., 'yell' for Yellowstone)
    park_name: Full park name
    source_url: Original document URL
    chunk_id: Unique identifier
  }

Search Flow:
1. Receive query embedding (1024-dim vector from Cohere)
2. Perform cosine similarity search against collection
3. Optionally filter by park_code
4. Return top_k most similar chunks with metadata

Why Qdrant:
+ Pros: Fast vector search, managed service, generous free tier
+ Cons: External dependency (but free tier is reliable)

API Version: qdrant-client 1.7.3 (uses .search() method)

Author: Built with Claude Code
Date: February 2026
"""
import os
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
import logging

logger = logging.getLogger(__name__)

# Configuration
COLLECTION_NAME = "national_parks"  # Qdrant collection name


class VectorDB:
    """
    Wrapper for Qdrant client providing vector similarity search

    Lazy-loads the connection to Qdrant Cloud on first search.
    """

    def __init__(self):
        """Initialize with no client (lazy loading)"""
        self.client = None
        self.collection_name = COLLECTION_NAME

    def connect(self):
        """
        Connect to Qdrant Cloud (lazy initialization)

        Reads credentials from environment variables and establishes connection.
        Strips whitespace from credentials to handle copy/paste errors.

        Raises:
            ValueError: If QDRANT_URL or QDRANT_API_KEY not set in environment
        """
        if self.client is None:
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")

            if not qdrant_url or not qdrant_api_key:
                raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set")

            # Strip whitespace/newlines from credentials to handle copy/paste errors
            # (Common issue when copying from Qdrant Cloud dashboard)
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
        Search for similar documents using vector similarity

        Performs cosine similarity search against the Qdrant collection.
        Optionally filters results by park code.

        Args:
            query_vector: 1024-dim embedding vector (from Cohere)
            top_k: Number of results to return (default: 5)
            park_code: Optional 4-letter park code to filter results
                      (e.g., 'yell' for Yellowstone, 'yose' for Yosemite)

        Returns:
            List[Dict]: List of search results, each containing:
                - id: Qdrant point ID
                - score: Similarity score (0-1, higher = more similar)
                - text: Document chunk text
                - park_code: Park identifier
                - park_name: Full park name
                - source_url: Original document URL
                - chunk_id: Unique chunk identifier

        Example:
            >>> vector_db = VectorDB()
            >>> query_vec = [0.1, 0.2, ...]  # 1024-dim vector
            >>> results = vector_db.search(query_vec, top_k=3)
            >>> print(results[0]['park_name'])
            'Yellowstone National Park'
        """
        # Ensure connection is established
        self.connect()

        # Build filter if park_code provided
        # This restricts search to only documents from specified park
        query_filter = None
        if park_code:
            query_filter = {
                "must": [
                    {"key": "park_code", "match": {"value": park_code}}
                ]
            }

        # Perform vector similarity search using Qdrant
        # Uses cosine similarity distance metric
        results: List[ScoredPoint] = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter
        )

        # Format results into consistent dictionary structure
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,  # Cosine similarity (0-1)
                "text": result.payload.get("text", ""),
                "park_code": result.payload.get("park_code", ""),
                "park_name": result.payload.get("park_name", ""),
                "source_url": result.payload.get("source_url", ""),
                "chunk_id": result.payload.get("chunk_id", "")
            })

        return formatted_results


# Global instance (shared across application)
# Lazy loads connection on first use
vector_db = VectorDB()
