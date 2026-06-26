"""Pydantic DTOs for the Knowledge Graph API.

Response shapes are shared by all graph-serving endpoints so the frontend
can rely on a single {nodes, edges, truncated} contract regardless of how
the subgraph was derived (single-node expansion, multi-seed, path, export).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# --- Core DTOs ---------------------------------------------------------------

class NodeDTO(BaseModel):
    id: str = Field(..., description="Stable node id (doi, scoped experiment id, or hash)")
    label: Optional[str] = Field(None, description="Primary Neo4j label: Paper|Claim|Experiment|Method|Cohort|Result")
    properties: Dict[str, Any] = Field(default_factory=dict, description="All node properties")
    degree: Optional[int] = Field(None, description="Total 1-hop degree; present on search/detail/batch results")


class EdgeDTO(BaseModel):
    id: str = Field(..., description="Stable edge id derived from Neo4j element id")
    source: str = Field(..., description="Source node id")
    target: str = Field(..., description="Target node id")
    type: str = Field(..., description="Relationship type (see ontology)")
    properties: Dict[str, Any] = Field(default_factory=dict)


class TruncationInfo(BaseModel):
    nodes: bool = Field(False, description="True if the node list was capped by `limit`")
    hidden_neighbors: int = Field(0, description="Number of additional 1-hop neighbors not returned")


class SubgraphResponse(BaseModel):
    nodes: List[NodeDTO]
    edges: List[EdgeDTO]
    truncated: TruncationInfo = Field(default_factory=TruncationInfo)


# --- Search ------------------------------------------------------------------

class SearchHit(BaseModel):
    id: str
    label: str
    name: str = Field(..., description="Display text for the hit (paper name, claim text, method name, etc.)")
    degree: int = Field(0, description="1-hop degree; useful for 'is this a hub?' warnings")


class SearchResponse(BaseModel):
    query: str
    results: List[SearchHit]
    total: int = Field(
        0,
        description="Total number of matches across all labels, independent of `limit`/`offset`. Use this to decide when to prompt the user to refine.",
    )
    facets: Dict[str, int] = Field(
        default_factory=dict,
        description="Match counts per label (e.g. {'Paper': 12, 'Claim': 3}). Drives label-pivot UI affordances.",
    )
    limit: int = Field(
        20,
        description="Page size used to produce this response (echoed from the request).",
    )
    offset: int = Field(
        0,
        description="Offset used to produce this response (echoed from the request). Clients implement 'load more' by requesting offset = offset + limit until offset + len(results) >= total.",
    )


# --- Paths -------------------------------------------------------------------

class PathResponse(BaseModel):
    found: bool
    nodes: List[NodeDTO] = Field(default_factory=list)
    edges: List[EdgeDTO] = Field(default_factory=list)
    length: Optional[int] = Field(None, description="Number of relationships in the path")


# --- Batch / export requests -------------------------------------------------

class NodeBatchRequest(BaseModel):
    ids: List[str] = Field(..., min_length=1, max_length=500)


class NodeBatchResponse(BaseModel):
    nodes: List[NodeDTO]


class ExportRequest(BaseModel):
    node_ids: List[str] = Field(..., min_length=1, max_length=2000)


# --- Schema / stats ----------------------------------------------------------

class SchemaResponse(BaseModel):
    labels: List[str]
    relationship_types: List[str] = Field(..., description="Relationship types actually present in the DB")
    ontology_relationship_types: List[str] = Field(..., description="Canonical types defined in the ontology")
    searchable_labels: List[str]
    label_display_fields: Dict[str, str]


class StatsResponse(BaseModel):
    node_total: int
    relationship_total: int
    nodes_by_label: Dict[str, int]
    relationships_by_type: Dict[str, int]
