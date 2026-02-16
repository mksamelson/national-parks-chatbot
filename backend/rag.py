"""
RAG (Retrieval Augmented Generation) Pipeline

This module orchestrates the complete RAG pipeline for the National Parks chatbot.

What is RAG:
RAG combines information retrieval with LLM generation to produce accurate,
grounded answers. Instead of relying solely on the LLM's training data, RAG:
1. Retrieves relevant documents from a knowledge base (vector database)
2. Provides this context to the LLM as part of the prompt
3. Generates an answer based on the retrieved context

Why RAG for National Parks:
- Accuracy: Answers grounded in official NPS documentation
- Up-to-date: Vector DB can be updated without retraining LLM
- Attribution: Can cite specific sources for transparency
- Reduced hallucination: LLM answers from provided context, not memory

Architecture Flow:
1. User asks question → FastAPI endpoint (main.py)
2. Question → Cohere API → 1024-dim embedding vector (embeddings.py)
3. Vector → Qdrant vector search → top_k similar document chunks (vector_db.py)
4. Question + context chunks → Groq LLM → generated answer (llm.py)
5. Answer + source metadata → FastAPI → frontend

This module connects all components and handles the orchestration logic.

Components:
- embedding_model: Cohere API client for text→vector conversion
- vector_db: Qdrant client for similarity search
- llm_client: Groq API client for text generation

Author: Built with Claude Code
Date: February 2026
"""
from typing import Dict, List
from embeddings import embedding_model
from vector_db import vector_db
from llm import llm_client
import logging

logger = logging.getLogger(__name__)

# System prompt that guides the LLM's behavior
# Sets the chatbot's personality, expertise domain, and response guidelines
SYSTEM_PROMPT = """You are a helpful and knowledgeable National Parks expert assistant. Your role is to help visitors learn about U.S. National Parks, including their features, activities, wildlife, history, and visitor information.

Guidelines:
- Provide accurate, helpful information based on the context provided
- Include specific details when available (trail names, distances, seasonal info, etc.)
- Cite your sources when answering (e.g., "According to Source 1...")
- If you don't have enough information to answer, say so and suggest where users can find more info
- Be friendly and encouraging about visiting national parks
- Always prioritize visitor safety when relevant"""


class RAGPipeline:
    """
    RAG Pipeline orchestrator for question answering

    Coordinates the three-step RAG process:
    1. Embedding: Convert question to vector (Cohere)
    2. Retrieval: Find similar documents (Qdrant)
    3. Generation: Produce answer with context (Groq)
    """

    def __init__(self):
        """Initialize pipeline with global API clients"""
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.llm = llm_client

    async def answer_question(
        self,
        question: str,
        top_k: int = 5,
        park_code: str = None
    ) -> Dict:
        """
        Answer a question using the complete RAG pipeline

        This is the main entry point for question answering. It performs:
        1. Question embedding (Cohere API)
        2. Vector similarity search (Qdrant)
        3. Context-aware answer generation (Groq API)

        Args:
            question: User question about national parks
                     (e.g., "What wildlife can I see in Yellowstone?")
            top_k: Number of context chunks to retrieve (default: 5)
                  More chunks = more context but slower/more expensive
            park_code: Optional 4-letter park code to filter results
                      (e.g., 'yell' for Yellowstone, 'yose' for Yosemite)
                      If None, searches across all parks

        Returns:
            Dict containing:
            - answer (str): Generated answer with source citations
            - sources (List[Dict]): List of source metadata for attribution
            - question (str): Original question (echo back)
            - num_sources (int): Number of sources used

        Example:
            >>> pipeline = RAGPipeline()
            >>> result = await pipeline.answer_question(
            ...     "What are the best trails in Yellowstone?",
            ...     top_k=5
            ... )
            >>> print(result['answer'])
            'According to Source 1, popular trails include...'
        """
        try:
            # Step 1: Generate question embedding using Cohere API
            # Converts natural language question → 1024-dim vector
            logger.info(f"Generating embedding for question: {question[:50]}...")
            query_vector = self.embedding_model.encode(question)

            # Step 2: Retrieve relevant context using Qdrant vector search
            # Finds top_k most similar document chunks via cosine similarity
            logger.info(f"Retrieving top {top_k} relevant chunks...")
            context_chunks = self.vector_db.search(
                query_vector=query_vector,
                top_k=top_k,
                park_code=park_code
            )

            # Handle case where no relevant documents found
            if not context_chunks:
                return {
                    "answer": "I couldn't find relevant information to answer your question. Please try rephrasing or ask about specific national parks.",
                    "sources": [],
                    "question": question,
                    "num_sources": 0
                }

            # Step 3: Generate answer with LLM using retrieved context
            # Groq API generates answer grounded in NPS documentation
            logger.info("Generating answer with LLM...")
            result = self.llm.generate_with_context(
                question=question,
                context_chunks=context_chunks,
                system_prompt=SYSTEM_PROMPT
            )

            # Add metadata for frontend display
            result["question"] = question
            result["num_sources"] = len(context_chunks)

            logger.info("✓ Answer generated successfully")
            return result

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            raise

    async def search(
        self,
        query: str,
        top_k: int = 10,
        park_code: str = None
    ) -> List[Dict]:
        """
        Direct vector search without LLM generation

        This endpoint performs only the retrieval step (no generation).
        Useful for debugging, exploring the knowledge base, or building
        custom UIs that display raw search results.

        Args:
            query: Search query (e.g., "hiking trails")
            top_k: Number of results to return (default: 10)
            park_code: Optional park code filter

        Returns:
            List[Dict]: Search results with metadata, each containing:
            - id: Qdrant point ID
            - score: Similarity score (0-1)
            - text: Document chunk text
            - park_code: Park identifier
            - park_name: Full park name
            - source_url: Original document URL
            - chunk_id: Unique chunk identifier

        Example:
            >>> pipeline = RAGPipeline()
            >>> results = await pipeline.search("geysers", top_k=5)
            >>> print(results[0]['park_name'])
            'Yellowstone National Park'
        """
        try:
            # Generate query embedding using Cohere API
            query_vector = self.embedding_model.encode(query)

            # Retrieve results from Qdrant (no LLM generation)
            results = self.vector_db.search(
                query_vector=query_vector,
                top_k=top_k,
                park_code=park_code
            )

            return results

        except Exception as e:
            logger.error(f"Error in search: {e}")
            raise


# Global instance
rag_pipeline = RAGPipeline()
