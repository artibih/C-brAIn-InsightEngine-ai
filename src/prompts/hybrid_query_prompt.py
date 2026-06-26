HYBRID_SYSTEM_TEMPLATE = """You are a Scientist expert in Alzheimer's Disease. Your task is to answer the user's question using ONLY the following sources of context.

## 1. Knowledge graph schema (for interpreting the table below)

{schema}

## 2. Results from the knowledge graph

{markdown_table}

## 3. Relevant excerpts from the literature

{chunks_block}

---

Answer the user's question clearly and concisely.
If there are no results from the knowledge graph, your should answer the question based on the literature excerpts only.
When using literature excerpts, cite them using bracket numbers like [1], [2].
Only cite numbers that appear in the provided excerpts.
Do NOT invent citations.
If citing multiple sources, format like [1,2].
If the provided context does not contain enough information, ask user to rephrase or provide more context."""


def build_hybrid_system_prompt(
    schema: str,
    markdown_table: str,
    chunks: list[dict],
) -> str:
    """
    Build the system prompt for the HybridQueryAgent.

    Args:
        schema: Knowledge graph schema (from apoc.meta.schema or similar).
        markdown_table: Cypher query results formatted as Markdown.
        chunks: List of RAG chunk texts (numbered in the prompt).

    Returns:
        Formatted system prompt string.
    """
    chunks_block = (
        "\n\n".join(
            f"[{i}] (DOI: {c.get('doi', 'Unknown')})\n{c['content'].strip()}"
            for i, c in enumerate(chunks, 1)
            if c.get("content")
        )
        if chunks
        else "(No excerpts retrieved.)"
    )
    return HYBRID_SYSTEM_TEMPLATE.format(
        schema=schema,
        markdown_table=markdown_table,
        chunks_block=chunks_block,
    )
