# National Parks Chatbot

An intelligent chatbot that helps users explore and learn about U.S. National Parks through natural conversation. Built using RAG (Retrieval Augmented Generation) with 100% free-tier services.

## Features

- 🏞️ Information on 20+ major U.S. National Parks
- 💬 Natural language Q&A interface
- 🧠 **Multi-turn conversation memory** - Ask follow-up questions naturally
- 📚 Sourced from official NPS data and documents
- 🔗 Citations to authoritative sources
- ⚡ Fast responses powered by Groq API
- 🆓 Completely free to run (free tier services)

## Architecture

```
User → Lovable.ai Frontend → Render Backend API (FastAPI)
                                     ↓
                    ┌────────────────┼────────────────┐
                    ↓                ↓                ↓
              Cohere API        Qdrant Cloud    Groq API
           (Embeddings)       (Vector Search)  (Llama 3.3 70B)
          embed-english-v3.0
             1024-dim
```

### RAG Pipeline Flow (with Conversational Understanding):
1. **User Query** → FastAPI endpoint
2. **Query Rewriting** → If conversation history exists, LLM rewrites query to resolve pronouns/references
   - Example: "what wildlife is there?" → "what wildlife is at Zion National Park?"
3. **Embedding** → Cohere API converts (rewritten) question to 1024-dim vector
4. **Retrieval** → Qdrant finds top-k similar documents (cosine similarity)
5. **Generation** → Groq LLM generates answer with retrieved context + conversation history
6. **Response** → Answer + sources returned to frontend

### Key Design Decisions:
- **Smart park context detection** → Automatically filters search to the park being discussed by analyzing USER messages only (ignores assistant responses to prevent context pollution)
- **Conversational query rewriting** → Resolves pronouns before vector search for accurate context retrieval
- **API-based embeddings** (Cohere) instead of local models → saves 300MB RAM
- **Lazy loading** of RAG pipeline → <2 second startup for Render port binding
- **Free tier services** → entire stack runs on $0/month
- **Memory footprint** → ~150MB (fits in Render's 512MB free tier)

### Recent Updates:
- **February 2026**: Fixed conversation context detection algorithm
  - Now correctly prioritizes current question over history
  - Filters to USER messages only (assistant responses no longer pollute context)
  - Processes messages in reverse order (newest first) for accurate context
  - Added graceful fallback for Qdrant filtering when indexes are unavailable
  - All 7 comprehensive test cases pass ✅

## Tech Stack

- **Frontend**: Lovable.ai (React)
- **Backend**: Python FastAPI on Render (512MB RAM free tier)
- **Vector Database**: Qdrant Cloud (1GB free tier)
- **LLM**: Groq API (Llama 3.3 70B, 30 req/min free)
- **Embeddings**: Cohere API (embed-english-v3.0, 1024-dim, 100 calls/min free)
- **Data Sources**: NPS.gov, NPS API, park brochures

## Project Structure

```
national-parks-chatbot/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── embeddings.py        # Embedding model loader
│   ├── vector_db.py         # Qdrant client
│   ├── llm.py               # Groq API integration
│   ├── rag.py               # RAG pipeline
│   ├── requirements.txt     # Backend dependencies
│   └── render.yaml          # Render deployment config
├── data_ingestion/
│   ├── scrape_nps.py        # NPS website scraper
│   ├── process_pdfs.py      # PDF text extraction
│   ├── chunk_documents.py   # Document chunking
│   ├── create_embeddings.py # Generate & upload embeddings
│   └── requirements.txt     # Data processing dependencies
├── data/
│   ├── raw/                 # Scraped and downloaded data
│   ├── processed/           # Cleaned and chunked data
│   └── metadata/            # Park metadata
├── .gitignore
├── PLAN.md                  # Detailed implementation plan
└── README.md                # This file
```

## Setup Instructions

### Prerequisites

- Python 3.9+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/mksamelson/national-parks-chatbot.git
cd national-parks-chatbot
```

### 2. Set Up Free Accounts

Create free accounts for these services:

1. **Cohere API**: https://dashboard.cohere.com
   - Get your API key (for embeddings)
   - Free tier: 100 API calls/minute

2. **Groq API**: https://console.groq.com
   - Get your API key (for LLM)
   - Free tier: 30 requests/minute

3. **Qdrant Cloud**: https://cloud.qdrant.io
   - Create a free cluster
   - Get your cluster URL and API key
   - Free tier: 1GB storage

4. **Render**: https://render.com
   - Sign up (for backend deployment)
   - Free tier: 512MB RAM, 750 hours/month

5. **Lovable.ai**: https://lovable.ai
   - Sign up (for frontend)

### 3. Data Collection

```bash
cd data_ingestion
pip install -r requirements.txt

# Set NPS API key (optional but recommended)
export NPS_API_KEY=your_key_here

# Run data collection
python scrape_nps.py
python process_pdfs.py
python chunk_documents.py
```

### 4. Create Vector Database

```bash
# Set API credentials
export COHERE_API_KEY=your_cohere_key
export QDRANT_URL=your_cluster_url
export QDRANT_API_KEY=your_api_key

# Generate embeddings (using Cohere API) and upload to Qdrant
# Note: This takes ~10 minutes due to rate limiting (stays under free tier)
python create_embeddings.py
```

### 5. Run Backend Locally

```bash
cd ../backend
pip install -r requirements.txt

# Set environment variables (or use .env file)
export COHERE_API_KEY=your_cohere_key
export GROQ_API_KEY=your_groq_key
export QDRANT_URL=your_cluster_url
export QDRANT_API_KEY=your_api_key

# Run FastAPI server
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

Note: The RAG pipeline uses lazy loading - it loads on first request (not startup) to enable fast port binding on Render.

### 6. Deploy Backend to Render

1. Push code to GitHub
2. Connect GitHub repo to Render
3. Set environment variables in Render dashboard
4. Deploy!

### 7. Build Frontend in Lovable.ai

1. Use the frontend specifications from `PLAN.md`
2. Connect to your Render backend URL
3. Deploy frontend

## Environment Variables

Create a `.env` file in the root directory (not committed to git):

```bash
# Cohere API (for embeddings)
COHERE_API_KEY=your_cohere_api_key

# Groq API (for LLM generation)
GROQ_API_KEY=your_groq_api_key

# Qdrant Cloud (for vector database)
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

# NPS API (optional, for data collection)
NPS_API_KEY=your_nps_api_key
```

**Important:** Strip any whitespace/newlines when copying API keys. All keys are automatically `.strip()`'ed in the code to handle copy/paste errors.

## API Endpoints

### Chat Endpoint (with Conversation Memory)

`POST /api/chat` - Main chat endpoint with optional conversation history

**Basic request (single question):**
```json
{
  "question": "What wildlife can I see in Yellowstone?",
  "top_k": 5
}
```

**Multi-turn conversation (with history):**
```json
{
  "question": "What about grizzly bears?",
  "top_k": 5,
  "conversation_history": [
    {"role": "user", "content": "What wildlife can I see in Yellowstone?"},
    {"role": "assistant", "content": "Yellowstone is home to diverse wildlife including grizzly bears, wolves, bison, elk..."}
  ]
}
```

**Parameters:**
- `question` (required): User question about national parks
- `top_k` (optional): Number of context chunks to retrieve (1-10, default: 5)
- `park_code` (optional): Filter results to specific park (e.g., "yell" for Yellowstone)
- `conversation_history` (optional): Array of previous messages (max 20 messages)

**Conversation History Format:**
- Each message: `{"role": "user" | "assistant", "content": "string"}`
- Maximum 20 messages (10 exchanges) to stay within token limits
- Optional - leave empty or omit for single-turn questions
- Backend is stateless - client manages conversation state

**How Conversational Context Works:**

The system uses two techniques to maintain conversation context:

1. **Park Context Detection** - Automatically detects which park you're discussing and filters search results to ONLY that park
2. **Query Rewriting** - Rewrites ambiguous questions to resolve pronouns and references

**Example conversation:**
1. User: "Tell me about Glacier National Park"
   - System detects: "Glacier" → filters all future searches to Glacier only
2. User: "What wildlife will I see there?"
   - **Park filter:** Glacier (auto-detected)
   - **Query rewritten:** "What wildlife can I see at Glacier National Park?"
   - **Search:** Only Glacier documents retrieved
   - **Result:** Accurate Glacier-specific wildlife information
3. User: "Are they dangerous?"
   - **Park filter:** Still Glacier
   - **Query rewritten:** "Are the animals at Glacier National Park dangerous?"
   - **Search:** Only Glacier safety documents
   - **Result:** Glacier-specific safety information

**Key benefit:** Once you mention a park, all follow-up questions automatically focus on that park until you mention a different park. This prevents mixing information from multiple parks.

**Example: Multi-turn conversation flow**
```python
# First question (no history)
response1 = post("/api/chat", {"question": "Tell me about Yellowstone"})

# Follow-up (with history)
response2 = post("/api/chat", {
  "question": "What hiking trails are there?",  # "there" = Yellowstone (from context)
  "conversation_history": [
    {"role": "user", "content": "Tell me about Yellowstone"},
    {"role": "assistant", "content": response1['answer']}
  ]
})

# Another follow-up (understands pronouns)
response3 = post("/api/chat", {
  "question": "Which one is best for beginners?",  # "one" = hiking trail (from context)
  "conversation_history": [
    {"role": "user", "content": "Tell me about Yellowstone"},
    {"role": "assistant", "content": response1['answer']},
    {"role": "user", "content": "What hiking trails are there?"},
    {"role": "assistant", "content": response2['answer']}
  ]
})
```

### Search Endpoint

`POST /api/search` - Direct vector search (no LLM generation)
```json
{
  "query": "hiking trails",
  "top_k": 10,
  "park_code": "yose"
}
```

### Health Check

`GET /` or `GET /health` - Server health status

## Troubleshooting

### Render Deployment Issues

**Port scan timeout:**
- Cause: App takes too long to bind to port
- Solution: Already implemented - lazy loading ensures <2 second startup

**Memory exceeded on Render:**
- Cause: Using local embedding models
- Solution: Already implemented - using Cohere API instead

**Vector dimension mismatch:**
- Error: "expected dim: 1024, got 384"
- Solution: Ensure using `embed-english-v3.0` (1024-dim), not `embed-english-light-v3.0` (384-dim)
- Re-run `create_embeddings.py` if needed

**API key errors:**
- Error: "Illegal header value"
- Solution: API keys auto-stripped of whitespace, but verify .env file has no extra newlines

**Cohere rate limit:**
- Error: "trial token rate limit exceeded"
- Solution: `create_embeddings.py` includes rate limiting (50 chunks/batch, 15 sec delays)

### Qdrant Issues

**Check database status:**
```bash
cd data_ingestion
python check_qdrant.py
```

**Verify embeddings:**
- Expected: ~2,000+ vectors in collection
- Dimension: 1024
- If dimension wrong, must recreate collection

## Development

### Running Tests

**Backend conversation memory tests:**
```bash
# Start backend server in one terminal
cd backend
uvicorn main:app --reload

# Run tests in another terminal
python test_conversation_backend.py
```

The test suite includes 7 comprehensive tests:
1. Basic question without history
2. Follow-up question with conversation history
3. Extended follow-up maintaining context
4. **Assistant messages don't change context** (critical test)
5. Current question overrides history
6. Most recent user mention takes precedence
7. Extended conversation (5+ turns) maintains context

**Unit tests:**
```bash
pytest tests/
```

### Code Formatting

```bash
black backend/ data_ingestion/
```

## Cost Breakdown

All services are on free tiers:

- **Cohere API**: 100 calls/min free, unlimited total calls
- **Groq API**: 30 req/min, 14,400 tokens/min free
- **Qdrant Cloud**: 1GB free (stores ~2,000 chunks @ 1024-dim)
- **Render**: 512MB RAM, 750 hours/month free (our app uses ~150MB)
- **Lovable.ai**: Free hosting included
- **Total Monthly Cost**: $0

### Why Free Tier Works:
- Cohere API eliminates need for local embeddings models (saves ~300MB RAM)
- Lazy loading enables <2 second startup (critical for Render port binding)
- Memory-optimized architecture fits in 512MB (vs 400-500MB for local models)
- Rate limits are sufficient for typical usage patterns

## Data Sources

- [National Park Service Official Website](https://www.nps.gov)
- [NPS Data API](https://www.nps.gov/subjects/developer/api-documentation.htm)
- [NPS Publications](https://www.nps.gov/subjects/publications/)
- Wikipedia (supplementary)

## License

MIT License

## Acknowledgments

- National Park Service for providing open data
- Groq for free LLM API access
- Qdrant for free vector database hosting
- Anthropic Claude for assisting with development

---

Built with ❤️ for national park enthusiasts
