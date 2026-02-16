# National Parks Chatbot

An intelligent chatbot that helps users explore and learn about U.S. National Parks through natural conversation. Built using RAG (Retrieval Augmented Generation) with 100% free-tier services.

## Features

- 🏞️ Information on 20+ major U.S. National Parks
- 💬 Natural language Q&A interface
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

### RAG Pipeline Flow:
1. **User Query** → FastAPI endpoint
2. **Embedding** → Cohere API converts question to 1024-dim vector
3. **Retrieval** → Qdrant finds top-k similar documents (cosine similarity)
4. **Generation** → Groq LLM generates answer with retrieved context
5. **Response** → Answer + sources returned to frontend

### Key Design Decisions:
- **API-based embeddings** (Cohere) instead of local models → saves 300MB RAM
- **Lazy loading** of RAG pipeline → <2 second startup for Render port binding
- **Free tier services** → entire stack runs on $0/month
- **Memory footprint** → ~150MB (fits in Render's 512MB free tier)

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

- `GET /` - Health check
- `POST /api/chat` - Main chat endpoint
  ```json
  {
    "question": "What wildlife can I see in Yellowstone?"
  }
  ```
- `POST /api/search` - Direct vector search
  ```json
  {
    "query": "hiking trails",
    "top_k": 5
  }
  ```

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

## Contributing

Contributions welcome! Please open an issue or submit a PR.

## License

MIT License

## Acknowledgments

- National Park Service for providing open data
- Groq for free LLM API access
- Qdrant for free vector database hosting
- Anthropic Claude for assisting with development

---

Built with ❤️ for national park enthusiasts
