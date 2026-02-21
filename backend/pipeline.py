"""
RAG Pipeline — National Parks Chatbot

Implements the full retrieval-augmented generation pipeline using native
LangChain integrations: CohereEmbeddings, QdrantVectorStore, and ChatGroq.
LangGraph orchestrates the pipeline as a directed graph of nodes.

Graph flow:
    START
      └─> extract_park          detect park from question / conversation
            ├─> rewrite_query   (only when conversation history is present)
            │     └─> retrieve
            └─> retrieve        (direct, when no history)
                  ├─> generate    (context found)
                  └─> no_results  (no context found)
                        └─> END

Author: Built with Claude Code
Date: February 2026
"""
import logging
import os
from typing import Dict, List, Optional

from langchain_cohere import CohereEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END, START, StateGraph
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# ─────────────────────────── Constants ────────────────────────────────────────

MODEL = "llama-3.3-70b-versatile"
COLLECTION = "national_parks"

SYSTEM_PROMPT = """You are a helpful and knowledgeable National Parks expert assistant. Your role is to help visitors learn about U.S. National Parks, including their features, activities, wildlife, history, and visitor information.

Guidelines:
- Provide accurate, helpful information based on the context provided
- Include specific details when available (trail names, distances, seasonal info, etc.)
- If you don't have enough information to answer, say so and suggest where users can find more info
- Be friendly and encouraging about visiting national parks
- Always prioritize visitor safety when relevant
- When answering follow-up questions, reference previous parts of the conversation naturally
- If a user's question refers to "it" or "there", use conversation context to understand what they mean"""

# Park name → 4-letter code (used for detection)
PARK_MAPPINGS: Dict[str, str] = {
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
    'crater lake': 'crla',
}

# 4-letter code → full park name (used for display and prompts)
CODE_TO_NAME: Dict[str, str] = {
    'yell': 'Yellowstone National Park',
    'yose': 'Yosemite National Park',
    'zion': 'Zion National Park',
    'glac': 'Glacier National Park',
    'grca': 'Grand Canyon National Park',
    'romo': 'Rocky Mountain National Park',
    'grsm': 'Great Smoky Mountains National Park',
    'acad': 'Acadia National Park',
    'olym': 'Olympic National Park',
    'grte': 'Grand Teton National Park',
    'brca': 'Bryce Canyon National Park',
    'arch': 'Arches National Park',
    'cany': 'Canyonlands National Park',
    'seki': 'Sequoia and Kings Canyon National Parks',
    'deva': 'Death Valley National Park',
    'jotr': 'Joshua Tree National Park',
    'shen': 'Shenandoah National Park',
    'mora': 'Mount Rainier National Park',
    'crla': 'Crater Lake National Park',
}

# ─────────────────────────── LangChain prompt template ────────────────────────

# Used in rewrite_query_node via LCEL: REWRITE_PROMPT | ChatGroq(...)
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that rewrites questions to be clear and "
        "specific for database search. Output only the rewritten question, nothing else.",
    ),
    (
        "human",
        "Given the conversation history below, rewrite the user's latest question "
        "to be self-contained and specific. Replace pronouns and references (like "
        "'it', 'there', 'that', 'them') with the actual entities they refer to.\n\n"
        "Conversation history:\n{conversation_text}\n\n"
        "Latest question: {question}{park_context}\n\n"
        "Rewrite this question to be clear and specific, suitable for searching a "
        "database. Include the park name or specific topic being discussed. Keep it "
        "concise (under 20 words).\n\nRewritten question:",
    ),
])

# ─────────────────────────── Lazy client initializers ─────────────────────────

_embeddings: Optional[CohereEmbeddings] = None
_qdrant_client: Optional[QdrantClient] = None
_vectorstore: Optional[QdrantVectorStore] = None


def _get_embeddings() -> CohereEmbeddings:
    global _embeddings
    if _embeddings is None:
        api_key = os.getenv("COHERE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable not set")
        _embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=api_key,
        )
    return _embeddings


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        url = os.getenv("QDRANT_URL", "").strip()
        api_key = os.getenv("QDRANT_API_KEY", "").strip()
        if not url or not api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set")
        _qdrant_client = QdrantClient(url=url, api_key=api_key)
    return _qdrant_client


def _get_vectorstore() -> QdrantVectorStore:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = QdrantVectorStore(
            client=_get_qdrant_client(),
            collection_name=COLLECTION,
            embedding=_get_embeddings(),
            content_payload_key="text",  # matches payload key used when building the index
        )
    return _vectorstore


# ─────────────────────────── LangGraph state schema ───────────────────────────

class RAGState(TypedDict):
    question: str               # Original user question (never modified)
    top_k: int                  # Number of chunks to retrieve
    conversation_history: List[Dict]
    park_code: Optional[str]    # Explicitly provided by the caller
    active_park_code: Optional[str]  # Detected from context, or same as park_code
    search_query: str           # Original or rewritten query sent to the retriever
    context_chunks: List[Dict]  # Retrieved documents from Qdrant
    answer: str                 # Final LLM-generated answer
    sources: List[Dict]         # Source metadata for frontend attribution
    num_sources: int


# ─────────────────────────── Helper ───────────────────────────────────────────

def find_park_in_text(text: str) -> Optional[str]:
    """
    Scan a single text string for a known park name and return its 4-letter code.

    This is the canonical park-detection function.  It is called on every
    Human Message before the query is sent to Qdrant, and its result is stored
    in the API response (active_park_code) so that follow-up questions that use
    pronouns ("there", "it", "the park") can be resolved without the user
    having to repeat the park name.

    Returns the first matching park code, or None if no park name is found.
    """
    text_lower = text.lower()
    for park_name, code in PARK_MAPPINGS.items():
        if park_name in text_lower:
            return code
    return None


def _detect_park(question: str, conversation_history: List[Dict]) -> Optional[str]:
    """
    Detect which park is being discussed using text-only signals.

    Priority order (all text-based — the dropdown park_code is intentionally
    excluded here and treated as a last-resort fallback in extract_park_node):
      1. Current Human Message  — user explicitly names the park right now
      2. Recent Human Messages in history — user named the park in a prior turn
      3. Last Assistant message — used only when it unambiguously contains a
         single park name, which happens when the prior response was filtered
         to one park.  Skipped if the assistant mentioned multiple parks.
    """
    # 1. Current question
    code = find_park_in_text(question)
    if code:
        logger.info(f"Park detected in current question: {code}")
        return code

    # 2. Human Messages in recent history (most recent first)
    user_messages = [m for m in conversation_history[-6:] if m.get("role") == "user"]
    for msg in reversed(user_messages):
        code = find_park_in_text(msg["content"])
        if code:
            logger.info(f"Park detected from user history: {code}")
            return code

    # 3. Last Assistant message (single-park match only — avoids false positives)
    assistant_messages = [m for m in conversation_history[-6:] if m.get("role") == "assistant"]
    if assistant_messages:
        last_lower = assistant_messages[-1]["content"].lower()
        matched_codes = {
            code
            for park_name, code in PARK_MAPPINGS.items()
            if park_name in last_lower
        }
        if len(matched_codes) == 1:
            code = next(iter(matched_codes))
            logger.info(f"Park detected from last assistant message (single match): {code}")
            return code

    return None


# ─────────────────────────── Graph nodes ──────────────────────────────────────

def extract_park_node(state: RAGState) -> dict:
    """
    Node 1 — Detect the active park from the Human Message text.

    Text detection (find_park_in_text / _detect_park) always runs first and
    covers three cases in priority order:
      1. Park name in the current question
      2. Park name in a prior Human Message
      3. Unambiguous park name in the last Assistant message

    The explicit park_code (forwarded by the frontend from the previous turn,
    or set via the park dropdown) is used only as a last-resort fallback when
    no park name appears anywhere in the conversation text.  This ensures that
    what the user *types* always overrides any UI selection.
    """
    active_park_code = _detect_park(
        state["question"],
        state.get("conversation_history") or [],
    )

    if not active_park_code and state.get("park_code"):
        active_park_code = state["park_code"]
        logger.info(f"Park from explicit park_code fallback: {active_park_code}")

    logger.info(f"Active park: {active_park_code or 'none (searching all parks)'}")
    return {"active_park_code": active_park_code}


def rewrite_query_node(state: RAGState) -> dict:
    """
    Node 2 (conditional) — Rewrite the question using LangChain LCEL.

    Resolves pronouns and references (e.g. "there", "it") so that the
    retrieval step receives a self-contained query suitable for vector search.
    Falls back to the original question if the LLM call fails.
    """
    question = state["question"]
    history = state.get("conversation_history") or []
    active_park_code = state.get("active_park_code")

    conversation_text = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-4:]
    ])

    park_context = ""
    if active_park_code:
        park_name = CODE_TO_NAME.get(active_park_code, active_park_code.upper())
        park_context = (
            f"\n\nIMPORTANT: The conversation is about {park_name}. "
            "Ensure the rewritten question includes this park name if relevant."
        )

    try:
        llm = ChatGroq(model=MODEL, temperature=0.3, max_tokens=100)
        chain = REWRITE_PROMPT | llm
        response = chain.invoke({
            "conversation_text": conversation_text,
            "question": question,
            "park_context": park_context,
        })
        rewritten = response.content.strip().strip('"').strip("'").strip()
        logger.info(f"Query rewrite: '{question}' -> '{rewritten}'")
        return {"search_query": rewritten}

    except Exception as e:
        logger.error(f"Query rewriting failed: {e}, using original question")
        return {"search_query": question}


def retrieve_node(state: RAGState) -> dict:
    """
    Node 3 — Retrieve the top-k most relevant document chunks from Qdrant.

    QdrantVectorStore handles embedding internally, so no separate embed_query
    node is needed.  Filters by active_park_code when a park has been detected.
    """
    search_query = state["search_query"]
    top_k = state["top_k"]
    active_park_code = state.get("active_park_code")

    park_filter = None
    if active_park_code:
        park_filter = Filter(
            must=[
                FieldCondition(
                    key="park_code",
                    match=MatchValue(value=active_park_code),
                )
            ]
        )

    vectorstore = _get_vectorstore()
    try:
        docs_with_scores = vectorstore.similarity_search_with_score(
            query=search_query,
            k=top_k,
            filter=park_filter,
        )
    except Exception as e:
        # Qdrant requires a keyword index on park_code for filtered searches.
        # If the index is missing, fall back to an unfiltered search and filter
        # the results manually in Python.  Run data_ingestion/create_index.py
        # to create the index and make this fallback unnecessary.
        if active_park_code and "Index required" in str(e):
            logger.warning(
                "Qdrant park_code index missing — falling back to unfiltered search "
                "with manual filtering. Run data_ingestion/create_index.py to fix permanently."
            )
            docs_all = vectorstore.similarity_search_with_score(
                query=search_query,
                k=top_k * 3,
                filter=None,
            )
            docs_with_scores = [
                (doc, score) for doc, score in docs_all
                if doc.metadata.get("park_code") == active_park_code
            ][:top_k]
        else:
            raise

    context_chunks = [
        {
            "id": i,
            "score": score,
            "text": doc.page_content,
            "park_code": doc.metadata.get("park_code", ""),
            "park_name": doc.metadata.get("park_name", ""),
            "source_url": doc.metadata.get("source_url", ""),
            "chunk_id": doc.metadata.get("chunk_id", ""),
        }
        for i, (doc, score) in enumerate(docs_with_scores)
    ]
    logger.info(f"Retrieved {len(context_chunks)} chunks")

    if active_park_code and context_chunks:
        parks_found = {c.get("park_code") for c in context_chunks}
        if active_park_code not in parks_found or len(parks_found) > 1:
            logger.warning(f"Park mismatch: expected {active_park_code}, got {parks_found}")
        else:
            logger.info(f"All results from expected park: {active_park_code}")

    return {"context_chunks": context_chunks}


def generate_node(state: RAGState) -> dict:
    """
    Node 4 — Generate an answer using LangChain ChatGroq with retrieved context.

    Builds a message list of [SystemMessage, *history, HumanMessage] and
    invokes the Groq LLM.  The user message contains the numbered context
    sources followed by the question.
    """
    question = state["question"]
    context_chunks = state["context_chunks"]
    active_park_code = state.get("active_park_code")
    history = state.get("conversation_history") or []

    # Format retrieved chunks as numbered sources
    context_text = "\n\n".join([
        f"[Source {i+1} - {chunk['park_name']}]\n{chunk['text']}"
        for i, chunk in enumerate(context_chunks)
    ])

    park_name = (
        context_chunks[0].get("park_name", "the park being discussed")
        if context_chunks else "national parks"
    )

    # Explicit park context statement keeps the LLM focused on the right park
    park_context_statement = ""
    if active_park_code and context_chunks:
        park_context_statement = (
            f"IMPORTANT CONTEXT: The user is currently asking about {park_name}. "
            f"All context provided below is specifically about {park_name}. "
            f"When the user uses words like 'there', 'it', or 'the park', "
            f"they are referring to {park_name}.\n\n"
        )

    park_restriction = (
        f"Answer ONLY about {park_name} using ONLY the context above. "
        f"Do not mention, compare, or reference any other national parks."
        if active_park_code else
        "Answer using only the context provided above."
    )

    user_content = (
        f"{park_context_statement}"
        f"Context from National Parks Service:\n\n{context_text}\n\n"
        f"User Question: {question}\n\n"
        f"{park_restriction}"
    )

    # Assemble messages: system + conversation history + final user prompt
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_content))

    try:
        # streaming=True enables token-level events via astream_events;
        # invoke() behaviour is unchanged — still returns a complete response.
        llm = ChatGroq(model=MODEL, temperature=0, streaming=True)
        response = llm.invoke(messages)
        answer = response.content
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise

    sources = [
        {
            "park_name": chunk["park_name"],
            "park_code": chunk["park_code"],
            "url": chunk["source_url"],
            "score": chunk["score"],
        }
        for chunk in context_chunks
    ]

    return {
        "answer": answer,
        "sources": sources,
        "num_sources": len(context_chunks),
    }


def no_results_node(state: RAGState) -> dict:
    """
    Node 5 (alternate) — Return a graceful message when retrieval finds nothing.
    """
    return {
        "answer": (
            "I couldn't find relevant information to answer your question. "
            "Please try rephrasing or ask about specific national parks."
        ),
        "sources": [],
        "num_sources": 0,
    }


# ─────────────────────────── Routing functions ────────────────────────────────

def _route_after_park_extraction(state: RAGState) -> str:
    """Rewrite the query only when there is conversation history to draw from."""
    if state.get("conversation_history"):
        return "rewrite_query"
    return "retrieve"


def _route_after_retrieval(state: RAGState) -> str:
    """Generate an answer only when context chunks were actually found."""
    if state.get("context_chunks"):
        return "generate"
    return "no_results"


# ─────────────────────────── Graph construction ───────────────────────────────

def _build_graph():
    """Compile the LangGraph StateGraph for the RAG pipeline."""
    graph = StateGraph(RAGState)

    graph.add_node("extract_park", extract_park_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("no_results", no_results_node)

    graph.add_edge(START, "extract_park")
    graph.add_conditional_edges("extract_park", _route_after_park_extraction)
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_conditional_edges("retrieve", _route_after_retrieval)
    graph.add_edge("generate", END)
    graph.add_edge("no_results", END)

    return graph.compile()


# ─────────────────────────── Public interface ─────────────────────────────────

class RAGPipeline:
    """
    RAG Pipeline using LangGraph for orchestration and native LangChain integrations.

    Graph flow:
        extract_park → (rewrite_query →) retrieve → generate

    Public interface is identical to the previous RAGPipeline so main.py
    requires only a one-line import change.
    """

    def __init__(self):
        self._graph = _build_graph()

    async def answer_question(
        self,
        question: str,
        top_k: int = 5,
        park_code: str = None,
        conversation_history: List[Dict] = None,
    ) -> Dict:
        """
        Answer a question using the complete RAG pipeline.

        Args:
            question: User question about national parks
            top_k: Number of context chunks to retrieve (default: 5)
            park_code: Optional 4-letter park code to restrict search (e.g. 'yell')
            conversation_history: Previous messages as list of
                                  {'role': 'user'|'assistant', 'content': str} dicts

        Returns:
            Dict with keys: answer, sources, question, num_sources
        """
        logger.info(
            f"Question: '{question}' | history: {len(conversation_history or [])} msgs"
        )

        initial_state: RAGState = {
            "question": question,
            "top_k": top_k,
            "conversation_history": conversation_history or [],
            "park_code": park_code,
            "active_park_code": None,
            "search_query": question,
            "context_chunks": [],
            "answer": "",
            "sources": [],
            "num_sources": 0,
        }

        result = await self._graph.ainvoke(initial_state)

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "question": question,
            "num_sources": result["num_sources"],
            "active_park_code": result.get("active_park_code"),
        }

    async def astream_answer(
        self,
        question: str,
        top_k: int = 5,
        park_code: str = None,
        conversation_history: List[Dict] = None,
    ):
        """
        Stream answer tokens using LangGraph's astream_events.

        Runs the full RAG graph and yields dicts as they become available:
          {"type": "token",  "content": str}          — one per generated token
          {"type": "done",   "sources": list,
                             "num_sources": int}       — final metadata event
          {"type": "error",  "message": str}           — if an exception occurs

        The generate node uses streaming=True on ChatGroq so that LangGraph's
        callback system emits on_chat_model_stream events for each token.
        The no_results path (empty retrieval) emits the fallback text as a
        single token followed immediately by the done event.
        """
        initial_state: RAGState = {
            "question": question,
            "top_k": top_k,
            "conversation_history": conversation_history or [],
            "park_code": park_code,
            "active_park_code": None,
            "search_query": question,
            "context_chunks": [],
            "answer": "",
            "sources": [],
            "num_sources": 0,
        }

        try:
            answer_started = False

            async for event in self._graph.astream_events(initial_state, version="v2"):
                event_kind = event.get("event", "")
                node = event.get("metadata", {}).get("langgraph_node", "")

                # Stream individual tokens from the generate node's LLM call
                if event_kind == "on_chat_model_stream" and node == "generate":
                    content = event["data"]["chunk"].content
                    if content:
                        answer_started = True
                        yield {"type": "token", "content": content}

                # Final graph completion — capture sources and handle no_results path
                elif event_kind == "on_chain_end" and event.get("name") == "LangGraph":
                    final_state = event["data"].get("output", {})
                    sources = final_state.get("sources", [])
                    num_sources = final_state.get("num_sources", 0)

                    # no_results path: no LLM call was made, emit the fallback text
                    if not answer_started and final_state.get("answer"):
                        yield {"type": "token", "content": final_state["answer"]}

                    yield {"type": "done", "sources": sources, "num_sources": num_sources}

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {"type": "error", "message": str(e)}

    async def search(
        self,
        query: str,
        top_k: int = 10,
        park_code: str = None,
    ) -> List[Dict]:
        """
        Direct vector search without LLM generation.

        Args:
            query: Search query text
            top_k: Number of results to return (default: 10)
            park_code: Optional park code filter

        Returns:
            List of matching document chunks with metadata
        """
        park_filter = None
        if park_code:
            park_filter = Filter(
                must=[
                    FieldCondition(
                        key="park_code",
                        match=MatchValue(value=park_code),
                    )
                ]
            )

        vectorstore = _get_vectorstore()
        docs_with_scores = vectorstore.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=park_filter,
        )

        return [
            {
                "id": i,
                "score": score,
                "text": doc.page_content,
                "park_code": doc.metadata.get("park_code", ""),
                "park_name": doc.metadata.get("park_name", ""),
                "source_url": doc.metadata.get("source_url", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
            }
            for i, (doc, score) in enumerate(docs_with_scores)
        ]


# Global instance
rag_pipeline = RAGPipeline()
