"""
National Parks Chatbot - FastAPI Backend
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import logging
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="National Parks Chatbot API",
    description="RAG-based chatbot for U.S. National Parks information",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy import of RAG pipeline to speed up startup
rag_pipeline = None


def get_rag_pipeline():
    """Lazy load RAG pipeline on first use"""
    global rag_pipeline
    if rag_pipeline is None:
        logger.info("Loading RAG pipeline (first request)...")
        from rag import rag_pipeline as rp
        rag_pipeline = rp
        logger.info("✓ RAG pipeline loaded")
    return rag_pipeline


@app.on_event("startup")
async def startup_event():
    """Log when app is ready"""
    logger.info("🚀 FastAPI app started successfully - ready to accept requests")
    logger.info(f"Health check available at /health")


# Request/Response models
class ChatRequest(BaseModel):
    question: str = Field(..., description="User question about national parks")
    park_code: Optional[str] = Field(None, description="Optional park code to filter results")
    top_k: Optional[int] = Field(5, description="Number of context chunks to retrieve", ge=1, le=10)


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


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "National Parks Chatbot API is running",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check - simple and fast for Render"""
    # Just return healthy - don't check env vars to keep it fast
    # Env vars will be checked when actually used
    return {
        "status": "healthy",
        "message": "All systems operational",
        "version": "1.0.0"
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - answer questions using RAG

    Example request:
    ```json
    {
        "question": "What wildlife can I see in Yellowstone?",
        "top_k": 5
    }
    ```
    """
    try:
        logger.info(f"Chat request: {request.question}")

        # Get RAG pipeline (loads on first use)
        pipeline = get_rag_pipeline()

        result = await pipeline.answer_question(
            question=request.question,
            top_k=request.top_k,
            park_code=request.park_code
        )

        return result

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    """
    Direct vector search endpoint (no LLM generation)

    Example request:
    ```json
    {
        "query": "hiking trails",
        "top_k": 10
    }
    ```
    """
    try:
        logger.info(f"Search request: {request.query}")

        # Get RAG pipeline (loads on first use)
        pipeline = get_rag_pipeline()

        results = await pipeline.search(
            query=request.query,
            top_k=request.top_k,
            park_code=request.park_code
        )

        return results

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/parks")
async def list_parks():
    """
    List all available parks in the database
    (To be implemented - would query Qdrant for unique park codes)
    """
    # TODO: Implement if needed
    return {
        "message": "Parks listing endpoint - to be implemented",
        "parks": []
    }


# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
