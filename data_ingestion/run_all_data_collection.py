"""
Master script to run all data collection steps in order
"""
import subprocess
import sys
from pathlib import Path

def run_script(script_name: str, description: str):
    """Run a Python script and handle errors"""
    print("\n" + "=" * 70)
    print(f"STEP: {description}")
    print("=" * 70)

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        print(f"Continue anyway? (y/n): ", end='')
        response = input().lower()
        return response == 'y'
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def main():
    """Run all data collection scripts"""
    print("=" * 70)
    print("NATIONAL PARKS CHATBOT - COMPLETE DATA COLLECTION")
    print("=" * 70)
    print("\nThis will run all data collection steps:")
    print("1. Scrape NPS.gov and NPS API")
    print("2. Download PDF brochures and maps")
    print("3. Scrape Wikipedia articles")
    print("4. Process PDFs to extract text")
    print("5. Chunk all documents")
    print("6. Generate embeddings and upload to Qdrant")
    print("\nEstimated time: 45-90 minutes")
    print("=" * 70)

    proceed = input("\nProceed with full data collection? (y/n): ")
    if proceed.lower() != 'y':
        print("Aborted.")
        return

    # Step 1: Scrape NPS
    if not run_script("scrape_nps.py", "Scrape NPS.gov and NPS API"):
        print("Exiting due to error.")
        return

    # Step 2: Download PDFs
    if not run_script("download_pdfs.py", "Download PDF brochures and maps"):
        print("Exiting due to error.")
        return

    # Step 3: Scrape Wikipedia
    if not run_script("scrape_wikipedia.py", "Scrape Wikipedia articles"):
        print("Exiting due to error.")
        return

    # Step 4: Process PDFs
    if not run_script("process_pdfs.py", "Extract text from PDFs"):
        print("Warning: PDF processing had issues, but continuing...")

    # Step 5: Chunk documents
    if not run_script("chunk_documents.py", "Chunk all documents"):
        print("Exiting due to error.")
        return

    # Step 6: Create embeddings
    print("\n" + "=" * 70)
    print("FINAL STEP: Generate embeddings and upload to Qdrant")
    print("=" * 70)
    print("\nIMPORTANT: Make sure you have set these environment variables:")
    print("  - QDRANT_URL")
    print("  - QDRANT_API_KEY")
    print("  - GROQ_API_KEY")
    print()

    proceed_embeddings = input("Ready to create embeddings? (y/n): ")
    if proceed_embeddings.lower() != 'y':
        print("\nSkipping embeddings. You can run it later with:")
        print("  python create_embeddings.py")
        return

    if not run_script("create_embeddings.py", "Generate embeddings and upload to Qdrant"):
        print("Exiting due to error.")
        return

    # Done!
    print("\n" + "=" * 70)
    print("🎉 ALL DATA COLLECTION COMPLETE!")
    print("=" * 70)
    print("\nYour chatbot is now ready with data from:")
    print("  ✓ NPS.gov official data")
    print("  ✓ NPS API (alerts, campgrounds)")
    print("  ✓ Park brochures and maps (PDFs)")
    print("  ✓ Wikipedia articles")
    print("\nNext steps:")
    print("  1. Test your backend: cd ../backend && uvicorn main:app --reload")
    print("  2. Deploy to Render")
    print("  3. Build frontend in Lovable.ai")
    print("=" * 70)


if __name__ == "__main__":
    main()
