import re

def remap_inline_citations(text: str, mapping: dict):
    """
    Handles:
    - [1]
    - [1,2]
    - [1, 2, 3]
    - deduplicates → [1,1] -> [1]
    - sorts → [3,1] -> [1,3]
    """
    def replace(match):
        numbers = match.group(1).split(",")

        remapped = set()
        for n in numbers:
            n = n.strip()
            if not n:
                continue

            try:
                n_int = int(n)
                remapped.add(mapping.get(n_int, n_int))
            except ValueError:
                continue  
        if not remapped:
            return ""

        return f"[{', '.join(str(x) for x in sorted(remapped))}]"

    return re.sub(r"\[([\d,\s]+)\]", replace, text)

def find_invalid_citations(answer: str, valid_ids: set[int]) -> set[int]:

    found = set()
    matches = re.findall(r"\[([\d,\s]+)\]", answer)
    for match in matches:
        for n in match.split(","):
            n = n.strip()
            if n.isdigit():
                n_int = int(n)
                if n_int not in valid_ids:
                    found.add(n_int)

    return found

def remove_invalid_citations(answer: str, valid_ids: set[int]) -> str:

    def replace(match):
        nums = match.group(1).split(",")

        valid = []
        for n in nums:
            n = n.strip()
            if n.isdigit() and int(n) in valid_ids:
                valid.append(n)

        if not valid:
            return ""  

        return f"[{', '.join(valid)}]"

    return re.sub(r"\[([\d,\s]+)\]", replace, answer)


def extract_results(step_results):
    literature = []
    stats = []
    for step_id, result in step_results.items():
        if result.get("agent") == "retrieval":
            literature.append({
                "text": result.get("answer"),
                "sources": result.get("sources")
            })
        elif result.get("agent")== "statistics":
            stats.append(result.get("output", {}).get("structured_results", {}))
    return literature, stats

def build_paper_index(step_results: dict) -> dict:
    index = {}

    for step in step_results.values():
        for src in step.get("sources", []):
            pid = src.get("paper_id")
            if pid:
                index[pid] = src

    return index

def group_chunks_by_paper(chunks):
        papers = {}

        for chunk in chunks:
            pid = chunk.get("paper_id")
            if not pid:
                continue

            if pid not in papers:
                papers[pid] = chunk 
        return list(papers.values())

def extract_used_citations(text: str) -> set[int]: 
        matches = re.findall(r"\[([\d,\s]+)\]", text)
        used = set()

        for match in matches:
            nums = match.split(",")
            for n in nums:
                n = n.strip()
                if n.isdigit():
                    used.add(int(n))

        return used

def deduplicate_sources(sources):
        seen = {}
        for src in sources:
            pid = src.get("paper_id")
            if not pid:
                continue

            if pid not in seen:
                seen[pid] = src

        return list(seen.values())

def reindex_citations(answer, sources):
    mapping = {}
    
    for i, src in enumerate(sources, 1):
        old = src.get("citation")
        if old is not None:
            mapping[old] = i
        src["citation"] = i

    new_answer = remap_inline_citations(answer, mapping)

    return new_answer, sources
