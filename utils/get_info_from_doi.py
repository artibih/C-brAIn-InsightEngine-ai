import requests
import re

def clean_abstract(text):
    if not text:
        return None
    return re.sub("<.*?>", "", text).strip()


def get_paper_metadata(doi: str):
    url = f"https://api.crossref.org/works/{doi}"

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()

        data = r.json()["message"]

        title = data.get("title", [None])[0]

        authors_list = data.get("author", [])
        authors = ", ".join(
            f"{a.get('given','')} {a.get('family','')}".strip()
            for a in authors_list
        ) if authors_list else None

        abstract_raw = data.get("abstract")
        abstract = clean_abstract(abstract_raw)

        return {
            "title": title,
            "authors": authors,
            "abstract": abstract
        }

    except Exception:
        return {
            "title": None,
            "authors": None,
            "abstract": None
        }