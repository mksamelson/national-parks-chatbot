"""
Create a keyword payload index on park_code in the Qdrant collection.

Qdrant requires a payload index on any field used for filtering.  Without it,
filtered searches raise:
    "Bad request: Index required but not found for 'park_code' of type [keyword]"

Run this script once after create_embeddings.py to enable efficient park filtering.
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

load_dotenv(dotenv_path="../.env")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "national_parks"


def create_park_code_index():
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("Error: QDRANT_URL and QDRANT_API_KEY must be set in .env")
        return False

    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL.strip(), api_key=QDRANT_API_KEY.strip())

    print(f"Creating keyword index on 'park_code' in collection '{COLLECTION_NAME}'...")
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="park_code",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print("OK Index created successfully.")
        print("  Qdrant will now filter by park_code without falling back to full scans.")
        return True
    except Exception as e:
        if "already exists" in str(e).lower():
            print("OK Index already exists - nothing to do.")
            return True
        print(f"FAILED to create index: {e}")
        return False


if __name__ == "__main__":
    create_park_code_index()
