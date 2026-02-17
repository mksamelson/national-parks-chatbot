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
- If you don't have enough information to answer, say so and suggest where users can find more info
- Be friendly and encouraging about visiting national parks
- Always prioritize visitor safety when relevant
- When answering follow-up questions, reference previous parts of the conversation naturally
- If a user's question refers to "it" or "there", use conversation context to understand what they mean"""


class RAGPipeline:
    """
    RAG Pipeline orchestrator for question answering

    Coordinates the RAG process with conversational query rewriting:
    1. Query Rewriting: Rewrite question with conversation context (if history exists)
    2. Embedding: Convert question to vector (Cohere)
    3. Retrieval: Find similar documents (Qdrant)
    4. Generation: Produce answer with context (Groq)
    """

    def __init__(self):
        """Initialize pipeline with global API clients"""
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.llm = llm_client

    def _rewrite_query_with_context(
        self,
        question: str,
        conversation_history: List[Dict]
    ) -> str:
        """
        Rewrite a question to include context from conversation history

        This is crucial for conversational RAG: pronouns and references like
        "there", "it", "them" need to be resolved BEFORE vector search,
        otherwise the search won't retrieve relevant documents.

        Example:
            Q1: "What are the best trails at Zion?"
            Q2: "What wildlife will I see there?"
            Rewritten Q2: "What wildlife can I see at Zion National Park?"

        Args:
            question: Current user question (may contain pronouns/references)
            conversation_history: Previous conversation messages

        Returns:
            Rewritten question with context resolved, suitable for vector search
        """
        # Build conversation context for the rewriting prompt
        conversation_text = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in conversation_history[-4:]  # Use last 4 messages (2 exchanges)
        ])

        rewrite_prompt = f"""Given the conversation history below, rewrite the user's latest question to be self-contained and specific. Replace pronouns and references (like "it", "there", "that", "them") with the actual entities they refer to.

Conversation history:
{conversation_text}

Latest question: {question}

Rewrite this question to be clear and specific, suitable for searching a database. Include the park name or specific topic being discussed. Keep it concise (under 20 words).

Rewritten question:"""

        try:
            rewritten = self.llm.generate(
                prompt=rewrite_prompt,
                system_prompt="You are a helpful assistant that rewrites questions to be clear and specific for database search. Output only the rewritten question, nothing else.",
                temperature=0.3,  # Lower temperature for more focused rewriting
                max_tokens=100
            )
            # Clean up the response (remove quotes, extra whitespace)
            rewritten = rewritten.strip().strip('"').strip("'").strip()

            logger.info(f"Query rewriting: '{question}' -> '{rewritten}'")
            return rewritten

        except Exception as e:
            logger.error(f"Query rewriting failed: {e}, using original question")
            return question  # Fallback to original question if rewriting fails

    def _extract_park_context(
        self,
        question: str,
        conversation_history: List[Dict]
    ) -> str:
        """
        Extract the park being discussed from conversation history

        When users are having a conversation about a specific park, we should
        filter vector search to ONLY that park to prevent mixing information
        from other parks.

        Example:
            User: "Tell me about Glacier National Park"
            User: "What wildlife is there?"
            -> Extract: "glac" (Glacier's park code)
            -> Filter search to only Glacier documents

        Args:
            question: Current user question
            conversation_history: Previous conversation messages

        Returns:
            Park code (4-letter) if a park is being discussed, None otherwise
        """
        # Common park names and their codes
        PARK_MAPPINGS = {
            'yellowstone': 'yell',
            'yosemite': 'yose',
            'zion': 'zion',
            'glacier': 'glac',
            'grand canyon': 'grca',
            'rocky mountain': 'romo',
            'great smoky': 'grsm',
            'great smoky mountains': 'grsm',
            'acadia': 'acad',
            'olympic': 'olym',
            'grand teton': 'grte',
            'bryce canyon': 'brca',
            'arches': 'arch',
            'canyonlands': 'cany',
            'sequoia': 'seki',
            'kings canyon': 'seki',
            'death valley': 'deva',
            'joshua tree': 'jotr',
            'shenandoah': 'shen',
            'mount rainier': 'mora',
            'crater lake': 'crla'
        }

        # Check last 6 messages (3 exchanges) for park mentions
        recent_messages = conversation_history[-6:] if conversation_history else []

        # Combine recent conversation text
        conversation_text = " ".join([
            msg['content'].lower()
            for msg in recent_messages
        ])

        # Also check current question
        conversation_text += " " + question.lower()

        # Find park mentions (most recent takes precedence)
        for park_name, park_code in PARK_MAPPINGS.items():
            if park_name in conversation_text:
                logger.info(f"Detected park context: {park_name} ({park_code})")
                return park_code

        return None

    async def answer_question(
        self,
        question: str,
        top_k: int = 5,
        park_code: str = None,
        conversation_history: List[Dict] = None
    ) -> Dict:
        """
        Answer a question using the complete RAG pipeline

        This is the main entry point for question answering. It performs:
        1. Query rewriting with conversation context (if history provided)
        2. Question embedding (Cohere API)
        3. Vector similarity search (Qdrant)
        4. Context-aware answer generation (Groq API)

        Args:
            question: User question about national parks
                     (e.g., "What wildlife can I see in Yellowstone?")
            top_k: Number of context chunks to retrieve (default: 5)
                  More chunks = more context but slower/more expensive
            park_code: Optional 4-letter park code to filter results
                      (e.g., 'yell' for Yellowstone, 'yose' for Yosemite)
                      If None, searches across all parks
            conversation_history: Optional list of previous messages for multi-turn conversations
                                Each message is a dict with 'role' and 'content' keys
                                (e.g., [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}])

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
            # Step 0a: Extract park context from conversation (if history exists)
            # This ensures we only search documents from the park being discussed
            # Example: User talks about "Glacier" → filter search to only Glacier docs
            active_park_code = park_code  # Use explicit park_code if provided
            if not active_park_code and conversation_history and len(conversation_history) > 0:
                active_park_code = self._extract_park_context(question, conversation_history)
                if active_park_code:
                    logger.info(f"Auto-filtering to park: {active_park_code}")

            # Step 0b: Rewrite query with conversation context (if history exists)
            # This resolves pronouns like "there", "it" before vector search
            # Example: "what wildlife is there?" → "what wildlife is at Zion National Park?"
            search_query = question
            if conversation_history and len(conversation_history) > 0:
                search_query = self._rewrite_query_with_context(question, conversation_history)

            # Step 1: Generate question embedding using Cohere API
            # Converts natural language question → 1024-dim vector
            # Uses the rewritten query for better context-aware search
            query_vector = self.embedding_model.encode(search_query)

            # Step 2: Retrieve relevant context using Qdrant vector search
            # Finds top_k most similar document chunks via cosine similarity
            # Uses active_park_code to filter to specific park if conversation detected one
            context_chunks = self.vector_db.search(
                query_vector=query_vector,
                top_k=top_k,
                park_code=active_park_code  # Automatically filters to detected park
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
            # Includes conversation history and detected park context
            result = self.llm.generate_with_context(
                question=question,
                context_chunks=context_chunks,
                system_prompt=SYSTEM_PROMPT,
                conversation_history=conversation_history,
                park_code=active_park_code  # Tell LLM which park is being discussed
            )

            # Add metadata for frontend display
            result["question"] = question
            result["num_sources"] = len(context_chunks)

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
