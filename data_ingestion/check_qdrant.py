"""
Quick script to verify Qdrant database status
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv(dotenv_path="../.env")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "national_parks"
EMBEDDING_DIM = 1024  # Cohere embed-english-v3.0


def check_qdrant():
    """Check if Qdrant is set up and populated"""

    print("=" * 60)
    print("QDRANT DATABASE STATUS CHECK")
    print("=" * 60)

    # Check environment variables
    print("\n1. Checking environment variables...")
    if not QDRANT_URL:
        print("   FAIL: QDRANT_URL not found in .env file")
        return False
    if not QDRANT_API_KEY:
        print("   FAIL: QDRANT_API_KEY not found in .env file")
        return False

    print(f"   OK: QDRANT_URL: {QDRANT_URL}")
    print(f"   OK: QDRANT_API_KEY: {QDRANT_API_KEY[:10]}...")

    # Try to connect
    print("\n2. Connecting to Qdrant...")
    try:
        client = QdrantClient(
            url=QDRANT_URL.strip(),
            api_key=QDRANT_API_KEY.strip(),
        )
        print(f"   OK: Connected to {QDRANT_URL}")
    except Exception as e:
        print(f"   FAIL: Connection failed: {e}")
        return False

    # Check collection exists
    print("\n3. Checking collections...")
    try:
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        print(f"   Found {len(collection_names)} collection(s): {collection_names}")

        if COLLECTION_NAME not in collection_names:
            print(f"   FAIL: Collection '{COLLECTION_NAME}' not found")
            print("   -> Run: python create_embeddings.py")
            return False

        print(f"   OK: Collection '{COLLECTION_NAME}' exists")
    except Exception as e:
        print(f"   FAIL: {e}")
        return False

    # Check vector count
    print("\n4. Checking vector count...")
    try:
        count_result = client.count(collection_name=COLLECTION_NAME, exact=True)
        vector_count = count_result.count
        print(f"   OK: {vector_count:,} vectors in collection")

        if vector_count == 0:
            print("   FAIL: Collection is empty — run: python create_embeddings.py")
            return False
        if vector_count < 1000:
            print(f"   WARN: Only {vector_count} vectors (expected ~2,000+)")
            print("   -> You might want to re-run: python create_embeddings.py")
    except Exception as e:
        print(f"   FAIL: {e}")
        return False

    # Test search (supports both old and new qdrant-client APIs)
    print("\n5. Testing vector search...")
    try:
        test_vector = [0.0] * EMBEDDING_DIM
        try:
            # qdrant-client >= 1.9.0
            results = client.query_points(collection_name=COLLECTION_NAME, query=test_vector, limit=3)
            points = results.points
        except AttributeError:
            # qdrant-client < 1.9.0 fallback
            points = client.search(collection_name=COLLECTION_NAME, query_vector=test_vector, limit=3)
        print(f"   OK: Search works — retrieved {len(points)} results")
        if points:
            print(f"   Sample: {points[0].payload.get('park_name', 'N/A')}")
    except Exception as e:
        print(f"   FAIL: Search failed: {e}")
        return False

    # Check park_code index by attempting a filtered search
    print("\n6. Checking park_code payload index...")
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        test_filter = Filter(must=[FieldCondition(key="park_code", match=MatchValue(value="yell"))])
        try:
            client.query_points(collection_name=COLLECTION_NAME, query=[0.0] * EMBEDDING_DIM, query_filter=test_filter, limit=1)
        except AttributeError:
            client.search(collection_name=COLLECTION_NAME, query_vector=[0.0] * EMBEDDING_DIM, query_filter=test_filter, limit=1)
        print("   OK: park_code keyword index exists (filtered search succeeded)")
    except Exception as fe:
        if "Index required" in str(fe):
            print("   WARN: park_code index missing — run: python create_index.py")
        else:
            print(f"   WARN: Could not verify index: {fe}")

    # Summary
    print("\n" + "=" * 60)
    print("QDRANT DATABASE IS READY!")
    print("=" * 60)
    print(f"OK: {vector_count:,} vectors in '{COLLECTION_NAME}'")
    print(f"OK: Search is working")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = check_qdrant()

    if not success:
        print("\n" + "=" * 60)
        print("SETUP INCOMPLETE")
        print("=" * 60)
        print("Next steps:")
        print("1. Ensure .env has QDRANT_URL and QDRANT_API_KEY")
        print("2. Run: python create_embeddings.py")
        print("=" * 60)
