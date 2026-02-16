"""
Generate embeddings for document chunks and upload to Qdrant
Now using Cohere API (no local models = low memory!)
"""
import os
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv
import cohere

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

# Configuration
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
COLLECTION_NAME = "national_parks"
COHERE_MODEL = "embed-english-light-v3.0"
EMBEDDING_DIM = 1024  # Cohere embedding dimension

INPUT_DIR = Path("../data/processed")
CHUNKS_FILE = INPUT_DIR / "all_chunks.json"


def load_chunks() -> List[Dict]:
    """Load all document chunks"""
    if not CHUNKS_FILE.exists():
        print(f"Error: {CHUNKS_FILE} not found")
        print("Please run chunk_documents.py first")
        return []

    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")
    return chunks


def initialize_qdrant() -> QdrantClient:
    """Initialize Qdrant client and create collection if needed"""
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("Error: QDRANT_URL and QDRANT_API_KEY must be set in environment variables")
        print("\nPlease set them in a .env file:")
        print("QDRANT_URL=https://your-cluster.qdrant.io")
        print("QDRANT_API_KEY=your_api_key")
        exit(1)

    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if COLLECTION_NAME in collection_names:
        print(f"Collection '{COLLECTION_NAME}' already exists")
        recreate = input("Recreate collection? This will delete existing data (y/n): ")
        if recreate.lower() == 'y':
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        else:
            print("Using existing collection")
            return client

    # Create collection
    print(f"Creating collection '{COLLECTION_NAME}' with dimension {EMBEDDING_DIM}...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE
        )
    )
    print(f"✓ Collection created")

    return client


def initialize_cohere() -> cohere.Client:
    """Initialize Cohere client"""
    if not COHERE_API_KEY:
        print("Error: COHERE_API_KEY must be set in environment variables")
        print("\nPlease set it in a .env file:")
        print("COHERE_API_KEY=your_api_key")
        print("\nGet a free API key at: https://dashboard.cohere.com")
        exit(1)

    print(f"Connecting to Cohere API...")
    client = cohere.Client(COHERE_API_KEY)
    print(f"✓ Connected to Cohere (model: {COHERE_MODEL})")
    return client


def generate_embeddings(chunks: List[Dict], cohere_client: cohere.Client) -> List[List[float]]:
    """Generate embeddings for all chunks using Cohere API"""
    print(f"Generating embeddings for {len(chunks)} chunks using Cohere...")

    texts = [chunk["text"] for chunk in chunks]
    all_embeddings = []

    # Process in batches (Cohere limit is 96 texts per request)
    batch_size = 96
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]

        # Get embeddings from Cohere
        response = cohere_client.embed(
            texts=batch_texts,
            model=COHERE_MODEL,
            input_type="search_document"
        )

        all_embeddings.extend(response.embeddings)

    print(f"✓ Generated {len(all_embeddings)} embeddings")
    return all_embeddings


def upload_to_qdrant(client: QdrantClient, chunks: List[Dict], embeddings: List[List[float]]):
    """Upload chunks and embeddings to Qdrant"""
    print(f"Uploading {len(chunks)} points to Qdrant...")

    points = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point = PointStruct(
            id=idx,
            vector=embedding,
            payload={
                "text": chunk["text"],
                "park_code": chunk["park_code"],
                "park_name": chunk["park_name"],
                "chunk_index": chunk["chunk_index"],
                "source_url": chunk["source_url"],
                "chunk_id": chunk["id"]
            }
        )
        points.append(point)

    # Upload in batches
    batch_size = 100
    for i in tqdm(range(0, len(points), batch_size)):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )

    print(f"✓ Uploaded {len(points)} points to Qdrant")


def test_retrieval(qdrant_client: QdrantClient, cohere_client: cohere.Client):
    """Test retrieval with sample queries"""
    print("\n" + "=" * 50)
    print("Testing retrieval with sample queries...")
    print("=" * 50)

    test_queries = [
        "What wildlife can I see in Yellowstone?",
        "Best hiking trails in Yosemite",
        "Camping information for Grand Canyon",
        "When is the best time to visit Zion?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")

        try:
            # Embed query with Cohere
            query_response = cohere_client.embed(
                texts=[query],
                model=COHERE_MODEL,
                input_type="search_query"
            )
            query_vector = query_response.embeddings[0]

            # Search
            results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=3
            )

            print(f"Top {len(results)} results:")
            for idx, result in enumerate(results, 1):
                park_name = result.payload.get("park_name", "Unknown")
                text_preview = result.payload.get("text", "")[:100] + "..."
                score = result.score
                print(f"  {idx}. {park_name} (score: {score:.3f})")
                print(f"     {text_preview}")

        except Exception as e:
            print(f"  ✗ Error testing query: {e}")


def main():
    """Main execution"""
    print("National Parks Embeddings Creator (Cohere API)")
    print("=" * 50)

    # Load chunks
    chunks = load_chunks()
    if not chunks:
        return

    # Initialize Qdrant
    qdrant_client = initialize_qdrant()

    # Initialize Cohere
    cohere_client = initialize_cohere()

    # Generate embeddings
    embeddings = generate_embeddings(chunks, cohere_client)

    # Upload to Qdrant
    upload_to_qdrant(qdrant_client, chunks, embeddings)

    # Test retrieval
    test_retrieval(qdrant_client, cohere_client)

    print("\n" + "=" * 50)
    print("✓ All done! Your vector database is ready.")
    print(f"✓ Collection: {COLLECTION_NAME}")
    print(f"✓ Total vectors: {len(chunks)}")
    print(f"✓ Embedding dimension: {EMBEDDING_DIM}")
    print("=" * 50)


if __name__ == "__main__":
    main()
