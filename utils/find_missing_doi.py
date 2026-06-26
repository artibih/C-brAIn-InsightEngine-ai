import json

with open("path-to-metadata", "r") as f:
    metadata_data = json.load(f)

with open("path-to-tables", "r") as f:
    summary_data = json.load(f)

metadata_paper_ids = {
    p["paper_id"] for p in metadata_data.get("data", {}).get("Get", {}).get("PaperMetadata_Test", [])
}

summary_paper_ids = {
    p["paper_id"] for p in summary_data.get("data", {}).get("Get", {}).get("Summary_Test", [])
}

papers_without_doi = summary_paper_ids - metadata_paper_ids

# Output result
print(f"Number of papers in Summary_Test without DOI: {len(papers_without_doi)}")

for pid in papers_without_doi:
    print(pid)

output = [{"paper_id": pid} for pid in papers_without_doi]
with open("papers_without_doi_comb.json", "w") as f:
    json.dump({"papers_without_doi_comb": output}, f, indent=2)