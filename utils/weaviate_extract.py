from weaviate import Client

client = Client("http://localhost:8080")  # change if needed

query = """
{
  Get {
    PaperMetadata_Test(limit: 100) {
      paper_id
      doi
    }
  }
}
"""

def missing_doi(p):
    doi = p.get("doi")
    return doi is None or doi.strip() == ""

def main():
    result = client.query.raw(query)
    papers = result["data"]["Get"]["PaperMetadata_Test"]

    filtered = [p for p in papers if missing_doi(p)]

    print("Total fetched:", len(papers))
    print("Missing DOI:", len(filtered))

    for p in filtered[:5]:
        print(p)

if __name__ == "__main__":
    main()