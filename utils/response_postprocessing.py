from typing import Any

from utils.citation_utils import remove_invalid_citations, extract_used_citations, deduplicate_sources, reindex_citations

def postprocess_answer(answer: str, sources: list[dict]):
    valid_ids = {s["citation"] for s in sources}

    answer = remove_invalid_citations(answer, valid_ids)
    used_citations = extract_used_citations(answer)

    sources = [
        s for s in sources
        if s["citation"] in used_citations
    ]

    sources = deduplicate_sources(sources)
    answer, sources = reindex_citations(answer, sources)

    return answer, sources

def response_text(response: Any) -> str:
             content = getattr(response, "content", "")
             if isinstance(content, str):
                 return content
             if isinstance(content, list):
                 parts = []
                 for item in content:
                     if isinstance(item, str):
                         parts.append(item)
                     elif isinstance(item, dict):
                         parts.append(item.get("text", ""))
                 return "".join(parts)
             return str(content)