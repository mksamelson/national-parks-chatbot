"""
Chunk scraped documents into smaller pieces for embedding
"""
import json
import os
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Configuration
INPUT_DIR = Path("../data/raw")
OUTPUT_DIR = Path("../data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 800  # tokens (approximate by characters)
CHUNK_OVERLAP = 200  # overlap between chunks


def count_tokens_approx(text: str) -> int:
    """Approximate token count (rough: 1 token ≈ 4 characters)"""
    return len(text) // 4


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks"""
    if not text:
        return []

    # Convert chunk size from tokens to characters (rough approximation)
    chunk_chars = chunk_size * 4
    overlap_chars = overlap * 4

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_chars

        # Find a good breaking point (end of sentence or paragraph)
        if end < text_length:
            # Look for sentence endings
            for delimiter in ['\n\n', '\n', '. ', '! ', '? ']:
                last_delim = text.rfind(delimiter, start + chunk_chars - 200, end)
                if last_delim != -1:
                    end = last_delim + len(delimiter)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap_chars

    return chunks


def extract_text_from_park_data(park_data: Dict) -> str:
    """Extract and combine all text from park data"""
    texts = []

    # API data
    api_data = park_data.get("api_data", {})
    if api_data:
        # Full name and description
        if api_data.get("fullName"):
            texts.append(f"Park Name: {api_data['fullName']}")

        if api_data.get("description"):
            texts.append(api_data["description"])

        # Weather info
        if api_data.get("weatherInfo"):
            texts.append(f"Weather Information: {api_data['weatherInfo']}")

        # Directions
        if api_data.get("directionsInfo"):
            texts.append(f"Directions: {api_data['directionsInfo']}")

        # Activities
        if api_data.get("activities"):
            activities = [a.get("name", "") for a in api_data["activities"]]
            if activities:
                texts.append(f"Activities: {', '.join(activities)}")

        # Topics
        if api_data.get("topics"):
            topics = [t.get("name", "") for t in api_data["topics"]]
            if topics:
                texts.append(f"Topics: {', '.join(topics)}")

    # Website content
    website_data = park_data.get("website_data", {})
    if website_data.get("content"):
        texts.append(website_data["content"])

    # Alerts
    alerts = park_data.get("alerts", [])
    for alert in alerts:
        if alert.get("title") and alert.get("description"):
            texts.append(f"ALERT - {alert['title']}: {alert['description']}")

    # Campgrounds
    campgrounds = park_data.get("campgrounds", [])
    for camp in campgrounds:
        if camp.get("name") and camp.get("description"):
            texts.append(f"Campground: {camp['name']} - {camp['description']}")

    return "\n\n".join(texts)


def create_chunks_from_park_data(park_file: Path) -> List[Dict]:
    """Process a single park file and create chunks with metadata"""
    with open(park_file, 'r', encoding='utf-8') as f:
        park_data = json.load(f)

    park_code = park_data.get("park_code", "unknown")
    park_name = park_data.get("api_data", {}).get("fullName", park_code.upper())

    # Extract all text
    full_text = extract_text_from_park_data(park_data)

    # Chunk the text
    text_chunks = chunk_text(full_text)

    # Create chunk objects with metadata
    chunks = []
    for idx, chunk_text in enumerate(text_chunks):
        chunk = {
            "id": f"{park_code}_chunk_{idx}",
            "park_code": park_code,
            "park_name": park_name,
            "chunk_index": idx,
            "text": chunk_text,
            "token_count": count_tokens_approx(chunk_text),
            "source_url": f"https://www.nps.gov/{park_code}/index.htm",
            "metadata": {
                "park_code": park_code,
                "park_name": park_name,
                "chunk_index": idx,
            }
        }
        chunks.append(chunk)

    return chunks


def process_all_parks():
    """Process all park files and create chunks"""
    park_files = list(INPUT_DIR.glob("*.json"))

    # Exclude the combined file
    park_files = [f for f in park_files if f.name != "all_parks.json"]

    if not park_files:
        print(f"No park data files found in {INPUT_DIR}")
        print("Please run scrape_nps.py first.")
        return

    print(f"Processing {len(park_files)} park files...")

    all_chunks = []
    stats = {
        "total_parks": len(park_files),
        "total_chunks": 0,
        "total_tokens": 0
    }

    for park_file in tqdm(park_files):
        chunks = create_chunks_from_park_data(park_file)
        all_chunks.extend(chunks)

        stats["total_chunks"] += len(chunks)
        stats["total_tokens"] += sum(c["token_count"] for c in chunks)

        # Save individual park chunks
        park_code = park_file.stem
        output_file = OUTPUT_DIR / f"{park_code}_chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    # Save all chunks combined
    combined_file = OUTPUT_DIR / "all_chunks.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # Save statistics
    stats_file = OUTPUT_DIR / "chunking_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Chunking complete!")
    print(f"✓ Total parks: {stats['total_parks']}")
    print(f"✓ Total chunks: {stats['total_chunks']}")
    print(f"✓ Total tokens (approx): {stats['total_tokens']:,}")
    print(f"✓ Average chunks per park: {stats['total_chunks'] / stats['total_parks']:.1f}")
    print(f"✓ Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    print("National Parks Document Chunker")
    print("=" * 50)
    process_all_parks()
