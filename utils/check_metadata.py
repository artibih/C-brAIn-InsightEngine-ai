import json

with open("path-to-the-table-content", "r") as f:
    data = json.load(f)

papers = data.get("data", {}).get("Get", {}).get("Summary_Test", [])

missing_paper_id = []
missing_doi = []

for paper in papers:
    paper_id = paper.get("paper_id")

    
    if not paper_id or str(paper_id).strip() == "":
        missing_paper_id.append(paper)

if missing_paper_id:
    print(f"Papers with missing paper_id ({len(missing_paper_id)}):")
    for p in missing_paper_id:
        print(p)
else:
    print("All papers have a valid paper_id.")
