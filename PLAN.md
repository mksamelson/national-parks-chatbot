# National Parks Chatbot Implementation Plan

## Context
Building a chatbot to help users explore and learn about national parks through natural conversation. The chatbot will use RAG (Retrieval Augmented Generation) to provide accurate, sourced information from authoritative documents and websites. All components must use free tiers to minimize costs.

## Architecture Overview

```
User → Lovable.ai Frontend → Render Backend API → Vector DB (Qdrant Cloud)
                                     ↓
                            Free LLM API (Groq/OpenRouter)
                                     ↓
                            Embedding Model (sentence-transformers)
```

## Recommended Free Tools

### 1. **LLM Model (Free Tier)**
**Recommendation: Groq API**
- **Model**: Llama 3.1 70B or Mixtral 8x7B
- **Why**:
  - Completely free tier with generous rate limits
  - Extremely fast inference (fastest LLM API available)
  - High-quality responses suitable for chatbot use
  - Simple REST API
- **Alternative**: OpenRouter (free tier) or Together AI

### 2. **Embedding Model (Open Source, No Cost)**
**Recommendation: sentence-transformers/all-MiniLM-L6-v2**
- **Why**:
  - Small, efficient (80MB model)
  - Good quality embeddings (384 dimensions)
  - Runs easily on Render free tier
  - Fast inference
- **Alternative**: sentence-transformers/all-mpnet-base-v2 (better quality, larger size)

### 3. **Vector Database (Free Tier)**
**Recommendation: Qdrant Cloud**
- **Why**:
  - Free tier: 1GB storage (enough for ~1M embeddings)
  - Fully managed cloud service
  - REST API and Python client
  - No credit card required for free tier
- **Alternative**: Pinecone (1M vectors free) or Weaviate Cloud

### 4. **Backend Hosting**
**Render Free Tier**
- 512MB RAM
- Spins down after inactivity (cold starts ~30s)
- 750 hours/month free
- Python/Node.js support

### 5. **Frontend**
**Lovable.ai**
- AI-powered frontend builder
- React-based
- Free hosting included

## Data Sources for National Parks

### Primary Sources (Authoritative)
1. **National Park Service (NPS) Official Website**
   - Main site: https://www.nps.gov
   - Individual park pages (e.g., https://www.nps.gov/yose/index.htm)
   - Contains: visitor info, history, maps, activities, safety

2. **NPS Data API**
   - API: https://www.nps.gov/subjects/developer/api-documentation.htm
   - Free API key
   - Structured data on parks, campgrounds, events, alerts

3. **NPS Publications**
   - Park brochures (PDFs)
   - Management plans
   - Scientific reports
   - Available at: https://www.nps.gov/subjects/publications/

### Secondary Sources
4. **Wikipedia National Parks Articles**
   - Comprehensive overviews
   - Historical context
   - Example: https://en.wikipedia.org/wiki/Yellowstone_National_Park

5. **Recreation.gov**
   - Camping and reservation info
   - https://www.recreation.gov

## Implementation Steps

### Phase 1: Data Collection & Preparation (Week 1)
1. **Scrape NPS website data**
   - Create Python scraper using BeautifulSoup/Scrapy
   - Target: top 20-30 most visited parks
   - Extract: descriptions, activities, visitor info, safety alerts

2. **Download NPS PDFs**
   - Park brochures for major parks
   - Use PyPDF2 or pdfplumber for text extraction

3. **Fetch from NPS API**
   - Get structured data (alerts, events, campgrounds)
   - Store as JSON

4. **Data preprocessing**
   - Clean HTML/PDF text
   - Chunk documents (500-1000 tokens per chunk)
   - Add metadata (park name, source URL, date)

### Phase 2: Vector Database Setup (Week 1)
1. **Set up Qdrant Cloud**
   - Create free account at https://cloud.qdrant.io
   - Create collection with 384 dimensions (for MiniLM-L6-v2)
   - Configure distance metric (cosine similarity)

2. **Generate embeddings**
   - Install sentence-transformers locally
   - Embed all document chunks
   - Upload to Qdrant with metadata

3. **Test retrieval**
   - Query examples: "hiking in Yosemite", "Yellowstone wildlife"
   - Verify relevant chunks returned

### Phase 3: Backend API Development (Week 2)
1. **Set up Python FastAPI backend**
   ```
   /api/chat - Main chat endpoint
   /api/search - Direct search endpoint
   /api/health - Health check
   ```

2. **Core components**
   - Qdrant client for vector search
   - Groq API client for LLM
   - RAG pipeline: retrieve → format context → generate answer

3. **API Logic**
   - User sends question
   - Embed question with sentence-transformers
   - Search Qdrant for top 5 relevant chunks
   - Format prompt with context
   - Call Groq API with context + question
   - Return response with sources

4. **Dependencies**
   ```
   fastapi
   uvicorn
   qdrant-client
   sentence-transformers
   groq
   torch (CPU version)
   ```

### Phase 4: Frontend Development (Week 2)
1. **Design in Lovable.ai**
   - Chat interface
   - Message history
   - Source citations display
   - Park suggestions/examples

2. **API Integration**
   - Connect to Render backend
   - Handle loading states
   - Display streaming responses (if supported)

3. **Features**
   - Example questions
   - Park name autocomplete
   - Link to official NPS pages

### Phase 5: Deployment (Week 3)
1. **Deploy backend to Render**
   - Create `requirements.txt`
   - Add `render.yaml` configuration
   - Set environment variables (Groq API key, Qdrant URL)
   - Deploy from GitHub repo

2. **Deploy frontend to Lovable**
   - Configure backend API URL
   - Test end-to-end

3. **Testing**
   - Test various queries
   - Verify source citations
   - Check response quality
   - Monitor cold start behavior

### Phase 6: Optimization (Ongoing)
1. **Improve retrieval**
   - Experiment with chunk sizes
   - Add reranking (if needed)
   - Tune top-k parameter

2. **Prompt engineering**
   - System prompt refinement
   - Citation format
   - Handling edge cases

3. **Data expansion**
   - Add more parks
   - Include seasonal info
   - Add trail data

## Technical Implementation Details

### Backend Structure
```
backend/
├── main.py              # FastAPI app
├── embeddings.py        # Embedding model loader
├── vector_db.py         # Qdrant client
├── llm.py               # Groq API integration
├── rag.py               # RAG pipeline
├── requirements.txt
└── render.yaml
```

### Data Ingestion Script
```
data_ingestion/
├── scrape_nps.py        # Web scraper
├── process_pdfs.py      # PDF extractor
├── chunk_documents.py   # Text chunking
├── create_embeddings.py # Embed & upload to Qdrant
└── requirements.txt
```

### Frontend Structure (Lovable.ai)
```
- Chat component
- Message list
- Input field
- Source citations panel
```

## RAG Prompt Template

```
You are a helpful National Parks expert. Use the following information to answer the user's question.
If you don't know the answer based on the context, say so.

Context from National Parks Service:
{retrieved_chunks}

User Question: {user_question}

Answer (include relevant source citations):
```

## Free Tier Limitations & Mitigations

1. **Render Cold Starts**
   - Mitigation: Add health check ping service (cron-job.org)
   - Keep-alive every 10 minutes

2. **Groq Rate Limits**
   - Free tier: ~30 requests/minute
   - Mitigation: Implement request queue, show "high demand" message

3. **Qdrant Storage**
   - 1GB limit (~1M vectors)
   - Mitigation: Start with top 30 parks, expand if needed

4. **Render RAM (512MB)**
   - Mitigation: Use smaller embedding model (MiniLM-L6-v2)
   - Lazy load model on first request

## Verification & Testing

### Data Collection
- [ ] Scraped data from at least 20 parks
- [ ] Extracted text from park brochures
- [ ] Data properly chunked and formatted

### Vector Database
- [ ] Qdrant collection created
- [ ] All chunks embedded and uploaded
- [ ] Test queries return relevant results

### Backend API
- [ ] `/api/chat` endpoint works
- [ ] Embedding model loads successfully
- [ ] Groq API integration working
- [ ] RAG pipeline returns accurate answers with sources

### Frontend
- [ ] Chat interface functional
- [ ] Messages display correctly
- [ ] Source citations shown
- [ ] Mobile responsive

### End-to-End
- [ ] Ask question about Yellowstone → Get accurate answer
- [ ] Ask about park activities → Get relevant suggestions
- [ ] Ask about camping → Get specific campground info
- [ ] Verify citations link to original sources

### Performance
- [ ] Backend responds in <5s (warm)
- [ ] Cold start completes in <30s
- [ ] No errors under normal usage

## Success Metrics
- Chatbot answers 90%+ of park-related questions accurately
- Responses include proper source citations
- System handles 100+ daily users on free tier
- Stays within all free tier limits

## Timeline (AI-Assisted Development)

### Day 1: Setup & Data Collection (2-3 hours)
- **Your tasks**: Sign up for free accounts (Groq, Qdrant, Render, Lovable) - 30 min
- **My tasks**:
  - Write data scraping scripts - 30 min
  - Write data processing & chunking - 30 min
  - Run scripts to collect park data - 1-2 hours (mostly wait time)

### Day 2: Vector DB & Backend (2-3 hours)
- **Your tasks**: Provide API keys, create Qdrant collection - 15 min
- **My tasks**:
  - Generate embeddings & upload to Qdrant - 1 hour
  - Build complete FastAPI backend - 1-2 hours
  - Local testing - 30 min

### Day 3: Deployment & Frontend (2-3 hours)
- **Your tasks**:
  - Deploy backend to Render - 30 min (with my guidance)
  - Work in Lovable.ai to build frontend - 1-2 hours (with my specifications)
- **My tasks**:
  - Provide frontend specifications & code snippets
  - Debug any deployment issues - 30 min

### Day 4: Testing & Refinement (1-2 hours)
- End-to-end testing
- Fix bugs and improve responses
- Tune RAG parameters

**Total**: 3-4 days, 7-11 hours total (including your setup time)
**Active coding time for me**: ~4-5 hours
**Your involvement needed**: ~2-3 hours (mostly account setup and deployment clicks)

## Next Steps After Approval
1. Create GitHub repository
2. Set up development environment
3. Begin data scraping script
4. Sign up for free accounts (Groq, Qdrant, Render)
