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

def check_qdrant():
    """Check if Qdrant is set up and populated"""

    print("=" * 60)
    print("QDRANT DATABASE STATUS CHECK")
    print("=" * 60)

    # Check environment variables
    print("\n1. Checking environment variables...")
    if not QDRANT_URL:
        print("   ✗ QDRANT_URL not found in .env file")
        return False
    if not QDRANT_API_KEY:
        print("   ✗ QDRANT_API_KEY not found in .env file")
        return False

    print(f"   ✓ QDRANT_URL: {QDRANT_URL}")
    print(f"   ✓ QDRANT_API_KEY: {QDRANT_API_KEY[:10]}...")

    # Try to connect
    print("\n2. Connecting to Qdrant...")
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        print(f"   ✓ Connected successfully to {QDRANT_URL}")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        return False

    # Check collections
    print("\n3. Checking collections...")
    try:
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        print(f"   Found {len(collection_names)} collection(s): {collection_names}")

        if COLLECTION_NAME not in collection_names:
            print(f"   ✗ Collection '{COLLECTION_NAME}' not found")
            print("   → You need to run: python create_embeddings.py")
            return False

        print(f"   ✓ Collection '{COLLECTION_NAME}' exists")
    except Exception as e:
        print(f"   ✗ Error checking collections: {e}")
        return False

    # Check collection info
    print("\n4. Checking collection details...")
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        vector_count = collection_info.points_count
        vector_dim = collection_info.config.params.vectors.size

        print(f"   ✓ Vector count: {vector_count:,}")
        print(f"   ✓ Vector dimension: {vector_dim}")

        if vector_count == 0:
            print("   ⚠ Warning: Collection is empty!")
            print("   → You need to run: python create_embeddings.py")
            return False

        if vector_count < 1000:
            print(f"   ⚠ Warning: Only {vector_count} vectors (expected ~2,000+)")
            print("   → You might want to re-run: python create_embeddings.py")

    except Exception as e:
        print(f"   ✗ Error checking collection info: {e}")
        return False

    # Test search
    print("\n5. Testing vector search...")
    try:
        # Create a simple test query vector (all zeros for simplicity)
        test_vector = [0.0] * vector_dim

        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=test_vector,
            limit=3
        )

        print(f"   ✓ Search works! Retrieved {len(results)} results")

        if results:
            print(f"\n   Sample result:")
            print(f"   - Park: {results[0].payload.get('park_name', 'N/A')}")
            print(f"   - Text preview: {results[0].payload.get('text', '')[:80]}...")

    except Exception as e:
        print(f"   ✗ Search failed: {e}")
        return False

    # Summary
    print("\n" + "=" * 60)
    print("✅ QDRANT DATABASE IS READY!")
    print("=" * 60)
    print(f"✓ Connected to Qdrant Cloud")
    print(f"✓ Collection '{COLLECTION_NAME}' exists")
    print(f"✓ {vector_count:,} vectors loaded")
    print(f"✓ Search is working")
    print("\n🎉 Your vector database is fully operational!")
    print("=" * 60)

    return True

if __name__ == "__main__":
    success = check_qdrant()

    if not success:
        print("\n" + "=" * 60)
        print("❌ SETUP INCOMPLETE")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Make sure .env file has QDRANT_URL and QDRANT_API_KEY")
        print("2. Run: python create_embeddings.py")
        print("=" * 60)
