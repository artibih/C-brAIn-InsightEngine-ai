import json

with open("path-to-metadata", "r") as f:
    data = json.load(f)

papers = data.get("papers_without_doi_comb", [])

seen = set()
unique_papers = []
for paper in papers:
    pid = paper.get("paper_id")
    if pid and pid not in seen:
        seen.add(pid)
        unique_papers.append(paper)

data["papers_without_doi_comb"] = unique_papers

with open("papers_without_doi_comb_unique.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Removed duplicates. {len(papers) - len(unique_papers)} duplicates removed.")
print("Unique papers saved to 'papers_without_doi_comb_unique.json'.")