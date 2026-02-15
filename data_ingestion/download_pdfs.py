"""
Download PDF brochures, maps, and publications from NPS.gov
"""
import os
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from tqdm import tqdm
import json

# Configuration
OUTPUT_DIR = Path("../data/raw/pdfs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = OUTPUT_DIR / "downloaded_pdfs.json"

# Park codes from scrape_nps.py
POPULAR_PARKS = [
    "grca", "yose", "yell", "zion", "romo", "acad", "grte", "olym", "glac", "jotr",
    "brca", "grsm", "arch", "seki", "cany", "deva", "redw", "badl", "crla", "petr",
    "shen", "meve", "hale", "bisc", "ever", "gumo", "cave", "thro", "wica", "kefj",
]

# Known brochure URL patterns
BROCHURE_PATTERNS = [
    "https://www.nps.gov/{park}/planyourvisit/upload/{park}map1.pdf",
    "https://www.nps.gov/{park}/planyourvisit/upload/{park}map.pdf",
    "https://www.nps.gov/{park}/planyourvisit/upload/{PARK}map1.pdf",
    "https://www.nps.gov/{park}/planyourvisit/upload/{PARK}map.pdf",
    "https://www.nps.gov/{park}/planyourvisit/upload/{PARK}_brochure.pdf",
    "https://www.nps.gov/{park}/planyourvisit/upload/{park}_brochure.pdf",
]


def download_file(url: str, output_path: Path) -> bool:
    """Download a file from URL"""
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        # Check if it's actually a PDF
        content_type = response.headers.get('content-type', '')
        if 'pdf' not in content_type.lower():
            return False

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except Exception as e:
        return False


def try_download_brochure(park_code: str) -> Optional[Path]:
    """Try multiple URL patterns to find and download park brochure"""
    park_upper = park_code.upper()

    for pattern in BROCHURE_PATTERNS:
        url = pattern.format(park=park_code, PARK=park_upper)
        output_file = OUTPUT_DIR / f"{park_code}_brochure.pdf"

        if output_file.exists():
            continue

        if download_file(url, output_file):
            print(f"  ✓ Downloaded brochure: {url}")
            return output_file

        time.sleep(0.5)  # Be respectful

    return None


def scrape_park_publications_page(park_code: str) -> List[Dict]:
    """Scrape the park's publications/planyourvisit page for PDF links"""
    pdfs = []

    # Try multiple page patterns
    pages_to_check = [
        f"https://www.nps.gov/{park_code}/planyourvisit/maps.htm",
        f"https://www.nps.gov/{park_code}/planyourvisit/brochures.htm",
        f"https://www.nps.gov/{park_code}/learn/photosmultimedia/upload.htm",
        f"https://www.nps.gov/{park_code}/planyourvisit/upload.htm",
    ]

    for page_url in pages_to_check:
        try:
            response = requests.get(page_url, timeout=10)
            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all PDF links
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.endswith('.pdf'):
                    # Make absolute URL
                    if href.startswith('http'):
                        pdf_url = href
                    elif href.startswith('/'):
                        pdf_url = f"https://www.nps.gov{href}"
                    else:
                        pdf_url = f"https://www.nps.gov/{park_code}/planyourvisit/{href}"

                    # Get link text for description
                    description = link.get_text(strip=True) or "Park document"

                    pdfs.append({
                        "url": pdf_url,
                        "description": description,
                        "source_page": page_url
                    })

            time.sleep(0.5)
        except Exception as e:
            continue

    # Remove duplicates
    unique_pdfs = []
    seen_urls = set()
    for pdf in pdfs:
        if pdf['url'] not in seen_urls:
            unique_pdfs.append(pdf)
            seen_urls.add(pdf['url'])

    return unique_pdfs


def download_park_pdfs(park_code: str) -> Dict:
    """Download all available PDFs for a park"""
    print(f"\n{'='*60}")
    print(f"Processing: {park_code.upper()}")
    print(f"{'='*60}")

    park_data = {
        "park_code": park_code,
        "brochure": None,
        "additional_pdfs": []
    }

    # Try to download main brochure
    print("Searching for park brochure...")
    brochure_path = try_download_brochure(park_code)
    if brochure_path:
        park_data["brochure"] = {
            "filename": brochure_path.name,
            "size_mb": brochure_path.stat().st_size / (1024 * 1024)
        }
    else:
        print("  ✗ No brochure found")

    # Scrape for additional PDFs
    print("Searching for additional PDFs...")
    found_pdfs = scrape_park_publications_page(park_code)

    if found_pdfs:
        print(f"  Found {len(found_pdfs)} potential PDFs")

        # Download up to 5 additional PDFs per park
        for idx, pdf_info in enumerate(found_pdfs[:5]):
            filename = f"{park_code}_doc_{idx+1}.pdf"
            output_path = OUTPUT_DIR / filename

            if output_path.exists():
                continue

            print(f"  Downloading: {pdf_info['description'][:50]}...")
            if download_file(pdf_info['url'], output_path):
                park_data["additional_pdfs"].append({
                    "filename": filename,
                    "description": pdf_info['description'],
                    "url": pdf_info['url'],
                    "size_mb": output_path.stat().st_size / (1024 * 1024)
                })
                print(f"    ✓ Saved as {filename}")

            time.sleep(1)  # Be respectful
    else:
        print("  ✗ No additional PDFs found")

    return park_data


def download_all_pdfs():
    """Download PDFs for all parks"""
    print("National Parks PDF Downloader")
    print("=" * 60)
    print(f"Downloading PDFs for {len(POPULAR_PARKS)} parks")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    all_metadata = []
    stats = {
        "total_parks": len(POPULAR_PARKS),
        "parks_with_brochures": 0,
        "total_pdfs": 0,
        "total_size_mb": 0.0
    }

    for park_code in POPULAR_PARKS:
        park_data = download_park_pdfs(park_code)
        all_metadata.append(park_data)

        # Update stats
        if park_data["brochure"]:
            stats["parks_with_brochures"] += 1
            stats["total_pdfs"] += 1
            stats["total_size_mb"] += park_data["brochure"]["size_mb"]

        stats["total_pdfs"] += len(park_data["additional_pdfs"])
        for pdf in park_data["additional_pdfs"]:
            stats["total_size_mb"] += pdf["size_mb"]

        time.sleep(2)  # Be respectful between parks

    # Save metadata
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": all_metadata,
            "stats": stats
        }, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("✓ PDF Download Complete!")
    print("=" * 60)
    print(f"Parks processed: {stats['total_parks']}")
    print(f"Parks with brochures: {stats['parks_with_brochures']}")
    print(f"Total PDFs downloaded: {stats['total_pdfs']}")
    print(f"Total size: {stats['total_size_mb']:.1f} MB")
    print(f"Metadata saved to: {METADATA_FILE}")
    print("=" * 60)

    print("\nNext steps:")
    print("1. Run: python process_pdfs.py")
    print("2. Then: python chunk_documents.py")
    print("3. Finally: python create_embeddings.py")


if __name__ == "__main__":
    download_all_pdfs()
