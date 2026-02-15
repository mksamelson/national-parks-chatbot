"""
Generate embeddings for document chunks and upload to Qdrant
"""
import os
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

# Configuration
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
COLLECTION_NAME = "national_parks"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

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
    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE
        )
    )
    print(f"✓ Collection created")

    return client


def load_embedding_model() -> SentenceTransformer:
    """Load sentence transformer model"""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✓ Model loaded (embedding dimension: {EMBEDDING_DIM})")
    return model


def generate_embeddings(chunks: List[Dict], model: SentenceTransformer) -> List[List[float]]:
    """Generate embeddings for all chunks"""
    print(f"Generating embeddings for {len(chunks)} chunks...")

    texts = [chunk["text"] for chunk in chunks]

    # Generate embeddings in batches
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )

    print(f"✓ Generated {len(embeddings)} embeddings")
    return embeddings.tolist()


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


def test_retrieval(client: QdrantClient, model: SentenceTransformer):
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

        # Embed query
        query_vector = model.encode(query).tolist()

        # Search
        results = client.search(
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


def main():
    """Main execution"""
    print("National Parks Embeddings Creator")
    print("=" * 50)

    # Load chunks
    chunks = load_chunks()
    if not chunks:
        return

    # Initialize Qdrant
    client = initialize_qdrant()

    # Load embedding model
    model = load_embedding_model()

    # Generate embeddings
    embeddings = generate_embeddings(chunks, model)

    # Upload to Qdrant
    upload_to_qdrant(client, chunks, embeddings)

    # Test retrieval
    test_retrieval(client, model)

    print("\n" + "=" * 50)
    print("✓ All done! Your vector database is ready.")
    print(f"✓ Collection: {COLLECTION_NAME}")
    print(f"✓ Total vectors: {len(chunks)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
