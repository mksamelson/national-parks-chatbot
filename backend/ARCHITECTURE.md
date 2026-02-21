# Backend Architecture: How a Query Flows Through the System

This document traces a user question from the moment it leaves the browser
through `main.py` and `pipeline.py` until the answer arrives back on screen.

---

## 1. Big Picture

```
Browser (React)
    │
    │  POST /api/chat
    │  { question, park_code?, conversation_history? }
    ▼
main.py  ──────────────────────────────────────────────────────
    │  Validates the request (Pydantic)
    │  Calls pipeline.answer_question()
    ▼
pipeline.py  ──────────────────────────────────────────────────
    │
    │   LangGraph StateGraph
    │
    │   ┌──────────────┐
    │   │ extract_park │  Which park are we talking about?
    │   └──────┬───────┘
    │          │
    │    has history?
    │    ┌─────┴──────────────────────────────┐
    │   YES                                   NO
    │    ▼                                    ▼
    │   ┌───────────────┐           go straight to retrieve
    │   │ rewrite_query │  Resolve "there", "it", pronouns
    │   └──────┬────────┘
    │          ▼
    │   ┌──────────────┐
    │   │   retrieve   │  Search Qdrant for relevant text chunks
    │   └──────┬───────┘
    │          │
    │   chunks found?
    │   ┌──────┴──────────────┐
    │  YES                   NO
    │   ▼                    ▼
    │  ┌──────────┐    ┌────────────┐
    │  │ generate │    │ no_results │
    │  └──────────┘    └────────────┘
    │      LLM answer    Fallback msg
    │          │              │
    └──────────┴──────────────┘
               │
    main.py returns JSON response
               │
    Browser receives answer + sources
```

---

## 2. The Request: Frontend → main.py

The React frontend sends an HTTP POST to `/api/chat`:

```json
{
  "question": "What are the best hiking trails?",
  "park_code": "zion",
  "conversation_history": [
    { "role": "user",      "content": "Tell me about Zion National Park" },
    { "role": "assistant", "content": "Zion is known for its towering..." }
  ]
}
```

| Field                  | Required? | Purpose                                              |
|------------------------|-----------|------------------------------------------------------|
| `question`             | Yes       | The user's current question                          |
| `park_code`            | No        | 4-letter code (e.g. `"zion"`) to limit results to one park |
| `conversation_history` | No        | Previous turns so the chatbot understands follow-ups |

---

## 3. main.py — The Gatekeeper

`main.py` is a **FastAPI** application. Its job is to:

1. **Receive the HTTP request** and parse the JSON body
2. **Validate the data** using a Pydantic model (`ChatRequest`)
3. **Load the pipeline** (lazy — only on the first request ever)
4. **Call the pipeline** and wait for the answer
5. **Return the JSON response** to the browser

### 3a. Request Validation (`ChatRequest`)

```python
class ChatRequest(BaseModel):
    question: str                              # required
    park_code: Optional[str] = None            # optional park filter
    top_k: Optional[int] = Field(5, ge=1, le=10)  # how many text chunks to retrieve
    conversation_history: Optional[List[Message]] = None  # chat history
```

Pydantic automatically rejects bad requests before they reach the pipeline
(e.g. missing question, history longer than 20 messages).

### 3b. Lazy Pipeline Loading

```python
_rag_pipeline = None          # starts as None

def get_rag_pipeline():
    global _rag_pipeline
    if _rag_pipeline is None:
        from pipeline import rag_pipeline as rp   # imported on first call only
        _rag_pipeline = rp
    return _rag_pipeline
```

The pipeline is **not loaded at startup** — it's loaded the first time a
request comes in. This keeps the server startup time under 2 seconds, which
is critical for Render's free-tier health checks.

### 3c. Calling the Pipeline

```python
@app.post("/api/chat")
async def chat(request: ChatRequest):
    pipeline = get_rag_pipeline()
    result = await pipeline.answer_question(
        question=request.question,
        top_k=request.top_k,
        park_code=request.park_code,
        conversation_history=_history_to_dicts(request.conversation_history),
    )
    return result
```

`_history_to_dicts()` converts the Pydantic `Message` objects into plain
Python dicts before passing them to the pipeline.

---

## 4. pipeline.py — The Brain

`pipeline.py` contains a **LangGraph StateGraph** — a directed graph where
each node is a step in answering the question. The graph runs the nodes in
order, passing a shared `state` dict between them.

### 4a. The State Object

Every node reads from and writes to a shared `RAGState` dict:

```python
class RAGState(TypedDict):
    question: str               # Original question (never changed)
    top_k: int                  # How many chunks to retrieve (default 5)
    conversation_history: list  # Prior messages
    park_code: str | None       # Explicit park from the frontend dropdown
    active_park_code: str | None  # Park detected from context (may differ)
    search_query: str           # Original or rewritten query for Qdrant
    context_chunks: list        # Text chunks retrieved from Qdrant
    answer: str                 # Final answer from the LLM
    sources: list               # Source URLs sent back to the browser
    num_sources: int
```

### 4b. Lazy Client Initialization

The three external services are initialized **once** and reused:

| Function               | Service          | What it does                              |
|------------------------|------------------|-------------------------------------------|
| `_get_embeddings()`    | Cohere API       | Converts text → a list of 1024 numbers   |
| `_get_qdrant_client()` | Qdrant Cloud     | Raw connection to the vector database     |
| `_get_vectorstore()`   | Qdrant via LangChain | Wraps the client; handles embedding + search |

---

## 5. The Five Graph Nodes

### Node 1 — `extract_park`

**Question:** *Which park is this conversation about?*

```
Input:  question + conversation_history + park_code (optional fallback)
Output: active_park_code  (e.g. "zion", "yell", or None)
```

#### `find_park_in_text(text)` — the core detection function

All park detection goes through a single named function:

```python
def find_park_in_text(text: str) -> Optional[str]:
    text_lower = text.lower()
    for park_name, code in PARK_MAPPINGS.items():
        if park_name in text_lower:
            return code
    return None
```

It scans any string for a known park name (e.g. "yellowstone", "grand canyon")
and returns the corresponding 4-letter code.  It is called on the current
question and on every Human Message in history.

#### Detection priority (all text-based)

1. **Park name in the current Human Message** — user says "Tell me about Zion"
2. **Park name in a recent Human Message in history** — user named the park in a prior turn
3. **Last Assistant message** — used only if it mentions exactly one park
   (avoids false positives when the assistant compared multiple parks)
4. **`park_code` fallback** — used *last*, only when text detection finds
   nothing.  This may come from the frontend's stored `detectedPark` (the
   `active_park_code` returned by the previous response) or from the dropdown.
   What the user *types* always overrides any UI selection.

The result is stored in `active_park_code` and also returned in the API
response so the frontend can forward it on the next request as a `park_code`
fallback, enabling pronoun resolution ("What's the weather *there* in June?")
across turns.

If no park is found by any method, `active_park_code = None` and retrieval
searches all parks.

---

### Routing after `extract_park`

```python
def _route_after_park_extraction(state):
    if state.get("conversation_history"):
        return "rewrite_query"   # there is prior conversation → rewrite first
    return "retrieve"            # fresh question → go straight to retrieval
```

---

### Node 2 — `rewrite_query` *(conditional — only with conversation history)*

**Question:** *What does the user actually mean by "there" or "it"?*

```
Input:  question + conversation_history + active_park_code
Output: search_query  (a self-contained, specific question)
```

**Why this is needed:** Follow-up questions often use pronouns that a vector
database cannot understand. "What are the trails like?" is too vague. The
rewrite resolves this to something like:
"What are the best hiking trails in Zion National Park?"

**How it works:** A prompt is sent to the Groq LLM (via LangChain LCEL):

```
REWRITE_PROMPT | ChatGroq(model="llama-3.3-70b-versatile")
```

The prompt instructs the LLM to output only the rewritten question (no
explanation), including the park name if one is active.

If the LLM call fails for any reason, the original question is used as-is.

---

### Node 3 — `retrieve`

**Question:** *What does the NPS database say about this topic?*

```
Input:  search_query + active_park_code + top_k
Output: context_chunks  (list of matching text passages with metadata)
```

**How it works:**

1. `QdrantVectorStore` converts `search_query` to a 1024-dimension embedding
   vector using the Cohere API (this happens internally — no separate node
   needed)
2. Qdrant performs a **cosine similarity search** across all stored vectors
   to find the `top_k` most relevant text chunks
3. If `active_park_code` is set, a **keyword filter** is applied so only
   chunks from that park are returned (`park_code == "zion"`)

**Fallback:** If the Qdrant keyword index is missing (rare), the node
fetches 3× more results from all parks and filters in Python.

Each retrieved chunk looks like:
```python
{
    "text": "Zion's famous Angels Landing trail...",
    "park_name": "Zion National Park",
    "park_code": "zion",
    "source_url": "https://www.nps.gov/zion/...",
    "score": 0.87   # similarity score 0–1
}
```

---

### Routing after `retrieve`

```python
def _route_after_retrieval(state):
    if state.get("context_chunks"):
        return "generate"     # we have context → answer the question
    return "no_results"       # nothing found → graceful fallback
```

---

### Node 4a — `generate`

**Question:** *Given the retrieved text, what is the best answer?*

```
Input:  question + context_chunks + active_park_code + conversation_history
Output: answer + sources
```

**How it works:**

The node builds a message list and sends it to the Groq LLM:

```
[SystemMessage]          ← the chatbot's personality and rules
[HumanMessage]           ← turn 1 from history
[AIMessage]              ← turn 1 assistant response
[HumanMessage]           ← turn 2 from history
  ...
[HumanMessage]           ← the FINAL user prompt (contains the context)
```

The final user prompt contains three things:

1. **Park context statement** — reminds the LLM which park is active:
   > "IMPORTANT CONTEXT: The user is currently asking about Zion National Park.
   > All context provided below is specifically about Zion National Park..."

2. **Retrieved context** — the numbered text chunks from Qdrant:
   > "[Source 1 - Zion National Park] Angels Landing is a 5.4-mile out-and-back..."
   > "[Source 2 - Zion National Park] The Narrows follows the Virgin River..."

3. **Park restriction** — prevents the LLM from going off-script:
   > "Answer ONLY about Zion National Park using ONLY the context above.
   > Do not mention, compare, or reference any other national parks."

The LLM is `llama-3.3-70b-versatile` running on Groq's inference servers.

---

### Node 4b — `no_results`

If retrieval found nothing, this node returns a friendly fallback message
instead of calling the LLM:

```
"I couldn't find relevant information to answer your question.
 Please try rephrasing or ask about specific national parks."
```

---

## 6. The Response: pipeline.py → main.py → Browser

After the graph finishes, `answer_question()` returns:

```python
{
    "answer": "Angels Landing is one of Zion's most iconic hikes...",
    "sources": [
        { "park_name": "Zion National Park", "park_code": "zion",
          "url": "https://www.nps.gov/zion/...", "score": 0.87 },
        ...
    ],
    "question": "What are the best hiking trails?",
    "num_sources": 5,
    "active_park_code": "zion"
}
```

`main.py` returns this directly as JSON. The browser receives it and:
- Displays `answer` as the bot's message bubble
- Shows `sources` as collapsible citation cards
- Stores `active_park_code` so it can be forwarded on the next follow-up

---

## 7. Streaming Mode (`/api/chat/stream`)

The streaming endpoint works identically except it sends the answer
**token by token** as Server-Sent Events (SSE) instead of waiting for
the full response:

```
data: {"type": "token",  "content": "Angels"}
data: {"type": "token",  "content": " Landing"}
data: {"type": "token",  "content": " is"}
...
data: {"type": "done",   "sources": [...], "num_sources": 5}
data: [DONE]
```

The frontend can display each token as it arrives, giving the appearance
of the chatbot "typing" in real time.

---

## 8. External Services Summary

| Service         | Provider     | Used in node    | Purpose                              |
|-----------------|--------------|-----------------|--------------------------------------|
| Cohere API      | Cohere       | `retrieve`      | Convert text queries to vectors      |
| Qdrant Cloud    | Qdrant       | `retrieve`      | Store + search 2,000+ text chunks    |
| Groq Inference  | Groq         | `rewrite_query`, `generate` | Run the LLaMA 3.3 70B LLM |

All three require API keys stored as environment variables (`.env` locally,
Render environment variables in production).
