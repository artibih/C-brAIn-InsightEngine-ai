import os
import re
import shutil
from pathlib import Path
from PyPDF2 import PdfReader
import json

class DOIFilter:
    def extract_doi(self, text: str):
        doi_pattern = r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b'
        matches = re.findall(doi_pattern, text, flags=re.IGNORECASE)
        return matches[0] if matches else None

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        reader = PdfReader(pdf_path)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)


def filter_papers_with_doi(input_dir: str, temp_dir: str, json_path: str):
    input_dir = Path(input_dir)
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    doi_filter = DOIFilter()
    doi_map = {}   # 👈 store pdf → doi

    for paper in input_dir.iterdir():
        if paper.suffix.lower() != ".pdf":
            continue

        try:
            text = doi_filter.extract_text_from_pdf(paper)
            doi = doi_filter.extract_doi(text)

            if doi:
                shutil.copy2(paper, temp_dir / paper.name)
                doi_map[paper.name] = doi   # 👈 save mapping
                print(f"✅ DOI found ({doi}) → copied: {paper.name}")
            else:
                print(f"❌ No DOI: {paper.name}")

        except Exception as e:
            print(f"⚠️ Error processing {paper.name}: {e}")

    # ---- WRITE JSON ----
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doi_map, f, indent=2)

    print(f"\n📄 JSON written to: {json_path}")
filter_papers_with_doi(
    input_dir="./papers",
    temp_dir="papers_with_doi",
    json_path="papers_with_doi.json"
)

