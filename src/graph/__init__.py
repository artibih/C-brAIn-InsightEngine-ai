"""Knowledge graph: extraction (deterministic + Judge), schema, and ingestion."""

from src.graph.schema.ontology import (
    ExtractedContent,
    Experiment,
    Claim,
    Result,
    Method,
    Cohort,
    ResultTrend,
)
from src.graph.schema.relationships import RelationType, is_valid_relation_type
from src.graph.ingestion import ingest_deterministic_graph, write_judge_edge

__all__ = [
    "ExtractedContent",
    "Experiment",
    "Claim",
    "Result",
    "Method",
    "Cohort",
    "ResultTrend",
    "RelationType",
    "is_valid_relation_type",
    "ingest_deterministic_graph",
    "write_judge_edge",
    "extract_content",
    "judge_and_link",
]

# Extraction imports instructor; import only when needed to avoid requiring it for schema/ingestion.
def __getattr__(name):
    if name in ("extract_content", "judge_and_link"):
        from src.graph.extraction import extract_content, judge_and_link
        return extract_content if name == "extract_content" else judge_and_link
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
