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
User → Lovable.ai Frontend → Render Backend API → Qdrant Vector DB
                                     ↓
                            Groq API (Llama 3.1 70B)
                                     ↓
                            sentence-transformers embeddings
```

## Tech Stack

- **Frontend**: Lovable.ai (React)
- **Backend**: Python FastAPI on Render
- **Vector Database**: Qdrant Cloud (free tier)
- **LLM**: Groq API (Llama 3.1 70B, free tier)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
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

1. **Groq API**: https://console.groq.com
   - Get your API key

2. **Qdrant Cloud**: https://cloud.qdrant.io
   - Create a free cluster
   - Get your cluster URL and API key

3. **Render**: https://render.com
   - Sign up (for backend deployment)

4. **Lovable.ai**: https://lovable.ai
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
# Set Qdrant credentials
export QDRANT_URL=your_cluster_url
export QDRANT_API_KEY=your_api_key

# Generate embeddings and upload to Qdrant
python create_embeddings.py
```

### 5. Run Backend Locally

```bash
cd ../backend
pip install -r requirements.txt

# Set environment variables
export GROQ_API_KEY=your_groq_key
export QDRANT_URL=your_cluster_url
export QDRANT_API_KEY=your_api_key

# Run FastAPI server
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

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

Create a `.env` file (not committed to git):

```bash
# Groq API
GROQ_API_KEY=your_groq_api_key

# Qdrant Cloud
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

# NPS API (optional)
NPS_API_KEY=your_nps_api_key
```

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

- **Groq API**: Free tier (30 req/min)
- **Qdrant Cloud**: 1GB free (enough for ~1M embeddings)
- **Render**: 512MB RAM, 750 hours/month free
- **Lovable.ai**: Free hosting included
- **Total Monthly Cost**: $0

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
