"""
National Parks Chatbot - FastAPI Backend

RAG-powered API server for the National Parks chatbot.

Architecture:
- LangGraph StateGraph orchestrates the RAG pipeline (rag.py)
- Cohere API for embeddings, Qdrant Cloud for vector search, Groq for LLM
- Two chat modes: standard (full response) and streaming (Server-Sent Events)

Key design decisions:
- Lazy loading: RAG pipeline loads on first request for fast startup (<2 sec),
  which is critical for Render's free-tier port-binding health checks.
- lifespan context manager (replaces deprecated @app.on_event).
- Streaming via LangGraph astream_events + FastAPI StreamingResponse (SSE).

Endpoints:
- GET  /                  - Root health check
- GET  /health            - Detailed health check
- POST /api/chat          - Standard RAG chat (complete response)
- POST /api/chat/stream   - Streaming RAG chat (Server-Sent Events)
- POST /api/search        - Direct vector search (no LLM)
- GET  /api/parks         - List available parks (placeholder)

Date: February 2026
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

# Load environment variables before any LangChain/LangGraph imports
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────── Lazy pipeline loader ─────────────────────────────

_rag_pipeline = None


def get_rag_pipeline():
    """Load the RAG pipeline on first use to keep startup time under 2 seconds."""
    global _rag_pipeline
    if _rag_pipeline is None:
        from rag import rag_pipeline as rp
        _rag_pipeline = rp
    return _rag_pipeline


# ─────────────────────────── App lifecycle ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: log startup / shutdown."""
    logger.info("National Parks Chatbot API started")
    yield
    logger.info("National Parks Chatbot API shutting down")


# ─────────────────────────── FastAPI app ──────────────────────────────────────

app = FastAPI(
    title="National Parks Chatbot API",
    description="RAG-based chatbot for U.S. National Parks information",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────── Request / Response models ────────────────────────

class Message(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    question: str = Field(..., description="User question about national parks")
    park_code: Optional[str] = Field(None, description="Optional park code to filter results")
    top_k: Optional[int] = Field(5, description="Number of context chunks to retrieve", ge=1, le=10)
    conversation_history: Optional[List[Message]] = Field(
        default=None,
        description="Previous conversation messages (optional, for multi-turn conversations)",
    )

    @field_validator("conversation_history")
    @classmethod
    def validate_history_length(cls, v):
        if v is not None and len(v) > 20:
            raise ValueError("conversation_history cannot exceed 20 messages (10 exchanges)")
        return v


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    park_code: Optional[str] = Field(None, description="Optional park code to filter results")
    top_k: Optional[int] = Field(10, description="Number of results to return", ge=1, le=20)


class Source(BaseModel):
    park_name: str
    park_code: str
    url: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    question: str
    num_sources: int


class SearchResult(BaseModel):
    id: int
    score: float
    text: str
    park_code: str
    park_name: str
    source_url: str
    chunk_id: str


class HealthResponse(BaseModel):
    status: str
    message: str
    version: str


# ─────────────────────────── Helpers ──────────────────────────────────────────

def _history_to_dicts(messages: Optional[List[Message]]) -> Optional[List[Dict]]:
    """Convert Pydantic Message models to plain dicts for the RAG pipeline."""
    if not messages:
        return None
    return [{"role": m.role, "content": m.content} for m in messages]


# ─────────────────────────── Endpoints ────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
async def root():
    """Root health check."""
    return {"status": "healthy", "message": "National Parks Chatbot API is running", "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check — kept intentionally lightweight for Render."""
    return {"status": "healthy", "message": "All systems operational", "version": "1.0.0"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Standard RAG chat endpoint — returns a complete response.

    Example request:
    ```json
    {
        "question": "What wildlife can I see in Yellowstone?",
        "top_k": 5,
        "conversation_history": [
            {"role": "user",      "content": "Tell me about Yellowstone"},
            {"role": "assistant", "content": "Yellowstone is..."}
        ]
    }
    ```
    """
    try:
        pipeline = get_rag_pipeline()
        result = await pipeline.answer_question(
            question=request.question,
            top_k=request.top_k,
            park_code=request.park_code,
            conversation_history=_history_to_dicts(request.conversation_history),
        )
        return result
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming RAG chat endpoint — returns tokens as Server-Sent Events.

    Uses LangGraph's astream_events to emit each generated token as it is
    produced by the Groq LLM, so the frontend can render the answer
    progressively without waiting for the full response.

    SSE event format:
      data: {"type": "token",  "content": "<text>"}
      data: {"type": "done",   "sources": [...], "num_sources": N}
      data: {"type": "error",  "message": "<msg>"}   (on exception)
      data: [DONE]                                    (stream terminator)

    Example request body is identical to /api/chat.
    """
    pipeline = get_rag_pipeline()
    conversation_history = _history_to_dicts(request.conversation_history)

    async def event_generator():
        try:
            async for event in pipeline.astream_answer(
                question=request.question,
                top_k=request.top_k,
                park_code=request.park_code,
                conversation_history=conversation_history,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    """
    Direct vector search endpoint — no LLM generation.

    Example request:
    ```json
    {"query": "hiking trails", "top_k": 10}
    ```
    """
    try:
        pipeline = get_rag_pipeline()
        results = await pipeline.search(
            query=request.query,
            top_k=request.top_k,
            park_code=request.park_code,
        )
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/parks")
async def list_parks():
    """List available parks (placeholder — would query Qdrant for unique codes)."""
    return {"message": "Parks listing endpoint - to be implemented", "parks": []}


# ─────────────────────────── Entry point ──────────────────────────────────────

# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
