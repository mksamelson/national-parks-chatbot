"""
Scrape Wikipedia articles for national parks to supplement NPS data
"""
import requests
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path
from typing import Dict
from tqdm import tqdm

# Configuration
OUTPUT_DIR = Path("../data/raw/wikipedia")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Wikipedia article titles for parks
PARK_WIKIPEDIA = {
    "grca": "Grand_Canyon_National_Park",
    "yose": "Yosemite_National_Park",
    "yell": "Yellowstone_National_Park",
    "zion": "Zion_National_Park",
    "romo": "Rocky_Mountain_National_Park",
    "acad": "Acadia_National_Park",
    "grte": "Grand_Teton_National_Park",
    "olym": "Olympic_National_Park",
    "glac": "Glacier_National_Park_(U.S.)",
    "jotr": "Joshua_Tree_National_Park",
    "brca": "Bryce_Canyon_National_Park",
    "grsm": "Great_Smoky_Mountains_National_Park",
    "arch": "Arches_National_Park",
    "seki": "Sequoia_and_Kings_Canyon_National_Parks",
    "cany": "Canyonlands_National_Park",
    "deva": "Death_Valley_National_Park",
    "redw": "Redwood_National_and_State_Parks",
    "badl": "Badlands_National_Park",
    "crla": "Crater_Lake_National_Park",
    "petr": "Petrified_Forest_National_Park",
    "shen": "Shenandoah_National_Park",
    "meve": "Mesa_Verde_National_Park",
    "hale": "Haleakalā_National_Park",
    "bisc": "Biscayne_National_Park",
    "ever": "Everglades_National_Park",
    "gumo": "Guadalupe_Mountains_National_Park",
    "cave": "Carlsbad_Caverns_National_Park",
    "thro": "Theodore_Roosevelt_National_Park",
    "wica": "Wind_Cave_National_Park",
    "kefj": "Kenai_Fjords_National_Park",
}


def fetch_wikipedia_article(article_title: str) -> Dict:
    """Fetch Wikipedia article content"""
    url = f"https://en.wikipedia.org/wiki/{article_title}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Get the main content
        content_div = soup.find('div', {'id': 'mw-content-text'})

        if not content_div:
            return {"url": url, "text": "", "error": "Content not found"}

        # Remove unwanted elements
        for element in content_div(['script', 'style', 'table', 'sup', 'span.mw-editsection']):
            element.decompose()

        # Extract paragraphs
        paragraphs = content_div.find_all('p')
        text = '\n\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

        # Get sections
        sections = []
        for heading in content_div.find_all(['h2', 'h3']):
            section_title = heading.get_text(strip=True).replace('[edit]', '')
            if section_title and section_title.lower() not in ['references', 'external links', 'see also', 'notes']:
                sections.append(section_title)

        return {
            "url": url,
            "title": article_title.replace('_', ' '),
            "text": text,
            "sections": sections,
            "char_count": len(text),
            "word_count": len(text.split())
        }

    except Exception as e:
        return {
            "url": url,
            "title": article_title,
            "text": "",
            "error": str(e)
        }


def scrape_all_wikipedia():
    """Scrape Wikipedia articles for all parks"""
    print("Wikipedia National Parks Scraper")
    print("=" * 60)
    print(f"Scraping {len(PARK_WIKIPEDIA)} Wikipedia articles")
    print("=" * 60)

    all_data = []
    stats = {
        "total_parks": len(PARK_WIKIPEDIA),
        "successful": 0,
        "failed": 0,
        "total_chars": 0,
        "total_words": 0
    }

    for park_code, wiki_title in tqdm(PARK_WIKIPEDIA.items()):
        print(f"\nFetching: {wiki_title}")

        data = fetch_wikipedia_article(wiki_title)
        data["park_code"] = park_code

        if data.get("error"):
            print(f"  ✗ Error: {data['error']}")
            stats["failed"] += 1
        else:
            print(f"  ✓ {data['word_count']:,} words")
            stats["successful"] += 1
            stats["total_chars"] += data["char_count"]
            stats["total_words"] += data["word_count"]

        all_data.append(data)

        # Save individual file
        output_file = OUTPUT_DIR / f"{park_code}_wikipedia.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        time.sleep(1)  # Be respectful to Wikipedia

    # Save combined data
    combined_file = OUTPUT_DIR / "all_wikipedia.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    # Save stats
    stats_file = OUTPUT_DIR / "wikipedia_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("✓ Wikipedia scraping complete!")
    print("=" * 60)
    print(f"Total parks: {stats['total_parks']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total characters: {stats['total_chars']:,}")
    print(f"Total words: {stats['total_words']:,}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    scrape_all_wikipedia()
