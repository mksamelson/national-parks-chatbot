"""
Extract text from PDF park brochures
Note: This script processes PDFs if you manually download them from NPS.gov
"""
import os
from pathlib import Path
from typing import Dict
import PyPDF2
import pdfplumber
from tqdm import tqdm
import json

# Configuration
PDF_DIR = Path("../data/raw/pdfs")
OUTPUT_DIR = Path("../data/raw/pdf_texts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_text_pypdf2(pdf_path: Path) -> str:
    """Extract text using PyPDF2"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"PyPDF2 error for {pdf_path.name}: {e}")
        return ""


def extract_text_pdfplumber(pdf_path: Path) -> str:
    """Extract text using pdfplumber (better for complex layouts)"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"pdfplumber error for {pdf_path.name}: {e}")
        return ""


def process_pdf(pdf_path: Path) -> Dict:
    """Process a single PDF and extract text"""
    print(f"Processing: {pdf_path.name}")

    # Try pdfplumber first (usually better quality)
    text = extract_text_pdfplumber(pdf_path)

    # Fallback to PyPDF2 if pdfplumber fails
    if not text:
        text = extract_text_pypdf2(pdf_path)

    if not text:
        print(f"Warning: No text extracted from {pdf_path.name}")

    return {
        "filename": pdf_path.name,
        "text": text,
        "char_count": len(text),
        "word_count": len(text.split())
    }


def process_all_pdfs():
    """Process all PDFs in the PDF directory"""
    if not PDF_DIR.exists():
        print(f"PDF directory not found: {PDF_DIR}")
        print("\nTo use this script:")
        print("1. Create directory: data/raw/pdfs/")
        print("2. Download park brochures from: https://www.nps.gov/subjects/publications/")
        print("3. Place PDF files in data/raw/pdfs/")
        print("4. Run this script again")
        return

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        print("Please download park brochures and place them in this directory")
        return

    print(f"Found {len(pdf_files)} PDF files")
    print("=" * 50)

    all_results = []
    total_chars = 0
    total_words = 0

    for pdf_file in tqdm(pdf_files):
        result = process_pdf(pdf_file)
        all_results.append(result)

        total_chars += result["char_count"]
        total_words += result["word_count"]

        # Save individual text file
        output_file = OUTPUT_DIR / f"{pdf_file.stem}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result["text"])

        # Save metadata
        metadata_file = OUTPUT_DIR / f"{pdf_file.stem}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

    # Save summary
    summary = {
        "total_pdfs": len(pdf_files),
        "total_characters": total_chars,
        "total_words": total_words,
        "files": all_results
    }

    summary_file = OUTPUT_DIR / "pdf_processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 50)
    print("✓ PDF processing complete!")
    print(f"✓ Processed: {len(pdf_files)} PDFs")
    print(f"✓ Total characters: {total_chars:,}")
    print(f"✓ Total words: {total_words:,}")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    print("National Parks PDF Processor")
    print("=" * 50)
    process_all_pdfs()
