"""
Scrape National Parks data from NPS.gov website and NPS API
"""
import os
import json
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Configuration
NPS_API_KEY = os.getenv('NPS_API_KEY', '')
BASE_URL = "https://developer.nps.gov/api/v1"
OUTPUT_DIR = Path("../data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Top 30 most visited national parks (by park code)
POPULAR_PARKS = [
    "grca",  # Grand Canyon
    "yose",  # Yosemite
    "yell",  # Yellowstone
    "zion",  # Zion
    "romo",  # Rocky Mountain
    "acad",  # Acadia
    "grte",  # Grand Teton
    "olym",  # Olympic
    "glac",  # Glacier
    "jotr",  # Joshua Tree
    "brca",  # Bryce Canyon
    "grsm",  # Great Smoky Mountains
    "arch",  # Arches
    "seki",  # Sequoia & Kings Canyon
    "cany",  # Canyonlands
    "deva",  # Death Valley
    "redw",  # Redwood
    "badl",  # Badlands
    "crla",  # Crater Lake
    "petr",  # Petrified Forest
    "shen",  # Shenandoah
    "meve",  # Mesa Verde
    "hale",  # Haleakala
    "bisc",  # Biscayne
    "ever",  # Everglades
    "gumo",  # Guadalupe Mountains
    "cave",  # Carlsbad Caverns
    "thro",  # Theodore Roosevelt
    "wica",  # Wind Cave
    "kefj",  # Kenai Fjords
]


def fetch_park_data_from_api(park_code: str) -> Dict:
    """Fetch park data from NPS API"""
    url = f"{BASE_URL}/parks"
    params = {
        "parkCode": park_code,
        "api_key": NPS_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("data"):
            return data["data"][0]
        return {}
    except Exception as e:
        print(f"Error fetching {park_code} from API: {e}")
        return {}


def scrape_park_page(park_code: str) -> Dict:
    """Scrape park information from NPS.gov website"""
    url = f"https://www.nps.gov/{park_code}/index.htm"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text content from main content area
        main_content = soup.find('div', {'id': 'main-content'}) or soup.find('main')

        text_content = ""
        if main_content:
            # Remove script and style elements
            for script in main_content(["script", "style"]):
                script.decompose()
            text_content = main_content.get_text(separator='\n', strip=True)

        return {
            "url": url,
            "content": text_content,
            "title": soup.title.string if soup.title else ""
        }
    except Exception as e:
        print(f"Error scraping {park_code} website: {e}")
        return {"url": url, "content": "", "title": ""}


def fetch_park_alerts(park_code: str) -> List[Dict]:
    """Fetch current alerts for a park"""
    if not NPS_API_KEY:
        return []

    url = f"{BASE_URL}/alerts"
    params = {
        "parkCode": park_code,
        "api_key": NPS_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    except Exception as e:
        print(f"Error fetching alerts for {park_code}: {e}")
        return []


def fetch_park_campgrounds(park_code: str) -> List[Dict]:
    """Fetch campground information for a park"""
    if not NPS_API_KEY:
        return []

    url = f"{BASE_URL}/campgrounds"
    params = {
        "parkCode": park_code,
        "api_key": NPS_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    except Exception as e:
        print(f"Error fetching campgrounds for {park_code}: {e}")
        return []


def scrape_all_parks():
    """Scrape data for all popular parks"""
    all_parks_data = []

    print(f"Scraping data for {len(POPULAR_PARKS)} national parks...")
    print(f"NPS API Key available: {bool(NPS_API_KEY)}")

    for park_code in tqdm(POPULAR_PARKS):
        print(f"\nProcessing: {park_code}")

        park_data = {
            "park_code": park_code,
            "api_data": {},
            "website_data": {},
            "alerts": [],
            "campgrounds": []
        }

        # Fetch from API
        if NPS_API_KEY:
            park_data["api_data"] = fetch_park_data_from_api(park_code)
            time.sleep(0.5)  # Rate limiting

            park_data["alerts"] = fetch_park_alerts(park_code)
            time.sleep(0.5)

            park_data["campgrounds"] = fetch_park_campgrounds(park_code)
            time.sleep(0.5)

        # Scrape website
        park_data["website_data"] = scrape_park_page(park_code)
        time.sleep(1)  # Be respectful to NPS servers

        all_parks_data.append(park_data)

        # Save individual park data
        output_file = OUTPUT_DIR / f"{park_code}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(park_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved to {output_file}")

    # Save combined data
    combined_file = OUTPUT_DIR / "all_parks.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_parks_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Scraping complete! Data saved to {OUTPUT_DIR}")
    print(f"✓ Combined data: {combined_file}")
    print(f"✓ Total parks: {len(all_parks_data)}")


if __name__ == "__main__":
    print("National Parks Data Scraper")
    print("=" * 50)

    if not NPS_API_KEY:
        print("\nWARNING: NPS_API_KEY not found in environment variables.")
        print("You can still scrape website data, but API features will be limited.")
        print("Get a free API key at: https://www.nps.gov/subjects/developer/get-started.htm\n")

        proceed = input("Continue anyway? (y/n): ")
        if proceed.lower() != 'y':
            print("Exiting...")
            exit(0)

    scrape_all_parks()
