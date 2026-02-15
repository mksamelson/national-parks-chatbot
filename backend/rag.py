"""
RAG (Retrieval Augmented Generation) Pipeline
"""
from typing import Dict, List
from embeddings import embedding_model
from vector_db import vector_db
from llm import llm_client
import logging

logger = logging.getLogger(__name__)

# System prompt for the chatbot
SYSTEM_PROMPT = """You are a helpful and knowledgeable National Parks expert assistant. Your role is to help visitors learn about U.S. National Parks, including their features, activities, wildlife, history, and visitor information.

Guidelines:
- Provide accurate, helpful information based on the context provided
- Include specific details when available (trail names, distances, seasonal info, etc.)
- Cite your sources when answering (e.g., "According to Source 1...")
- If you don't have enough information to answer, say so and suggest where users can find more info
- Be friendly and encouraging about visiting national parks
- Always prioritize visitor safety when relevant"""


class RAGPipeline:
    """RAG pipeline for question answering"""

    def __init__(self):
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
        Answer a question using RAG

        Args:
            question: User question
            top_k: Number of context chunks to retrieve
            park_code: Optional filter by specific park

        Returns:
            Dict with answer, sources, and metadata
        """
        try:
            # Step 1: Generate question embedding
            logger.info(f"Generating embedding for question: {question[:50]}...")
            query_vector = self.embedding_model.encode(question)

            # Step 2: Retrieve relevant context
            logger.info(f"Retrieving top {top_k} relevant chunks...")
            context_chunks = self.vector_db.search(
                query_vector=query_vector,
                top_k=top_k,
                park_code=park_code
            )

            if not context_chunks:
                return {
                    "answer": "I couldn't find relevant information to answer your question. Please try rephrasing or ask about specific national parks.",
                    "sources": [],
                    "question": question
                }

            # Step 3: Generate answer with LLM
            logger.info("Generating answer with LLM...")
            result = self.llm.generate_with_context(
                question=question,
                context_chunks=context_chunks,
                system_prompt=SYSTEM_PROMPT
            )

            # Add metadata
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

        Args:
            query: Search query
            top_k: Number of results
            park_code: Optional filter by park

        Returns:
            List of search results
        """
        try:
            # Generate query embedding
            query_vector = self.embedding_model.encode(query)

            # Retrieve results
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
