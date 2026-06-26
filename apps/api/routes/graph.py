"""
Knowledge Graph API.

Exposes a small set of REST endpoints plus one SSE endpoint for serving
the Neo4j knowledge graph to a frontend for interactive exploration.

Design: the frontend drives exploration node-by-node ("Google Maps for graphs").
Every endpoint returns a bounded subgraph so responses stay small regardless of
the total graph size.

Endpoints:
  GET  /search?q=...&label=...&limit=...&offset=...
  GET  /node?id=...
  POST /nodes/batch
  GET  /neighbors?id=...&depth=...&limit=...&rel_types=...
  GET  /subgraph?seeds=a,b,c&depth=...&limit=...&rel_types=...
  GET  /paths?from_id=...&to_id=...&max_depth=...
  GET  /paths/stream?from_id=...&to_id=...&max_depth=...&limit=...   (SSE)
  POST /export
  GET  /schema
  GET  /stats
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from apps.api.schemas.graph import (
    EdgeDTO,
    ExportRequest,
    NodeBatchRequest,
    NodeBatchResponse,
    NodeDTO,
    PathResponse,
    SchemaResponse,
    SearchHit,
    SearchResponse,
    StatsResponse,
    SubgraphResponse,
    TruncationInfo,
)
from src.graph import query as graph_query
from src.graph.query import SEARCHABLE_LABELS

logger = structlog.get_logger()

router = APIRouter()


# --- Helpers -----------------------------------------------------------------

def _parse_csv_param(value: Optional[str]) -> list[str]:
    """Parse a comma-separated query parameter into a list of trimmed non-empty strings."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _to_subgraph_response(data: dict) -> SubgraphResponse:
    return SubgraphResponse(
        nodes=[NodeDTO(**n) for n in data.get("nodes", [])],
        edges=[EdgeDTO(**e) for e in data.get("edges", [])],
        truncated=TruncationInfo(**data.get("truncated", {})),
    )


# --- Search ------------------------------------------------------------------

@router.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, description="Substring to search for (case-insensitive)"),
    label: Optional[str] = Query(None, description=f"Restrict to one label. Allowed: {', '.join(SEARCHABLE_LABELS)}"),
    limit: int = Query(20, ge=1, le=100, description="Page size."),
    offset: int = Query(
        0,
        ge=0,
        le=1000,
        description=(
            "Number of results to skip for pagination."
        ),
    ),
):
    """Search nodes by display text across Paper, Claim, Method, Cohort, Result."""
    if label is not None and label not in SEARCHABLE_LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown label '{label}'. Allowed: {list(SEARCHABLE_LABELS)}",
        )
    try:
        data = await asyncio.to_thread(
            graph_query.search_nodes, q, label, limit, offset
        )
        return SearchResponse(
            query=q,
            results=[SearchHit(**row) for row in data["results"]],
            total=data["total"],
            facets=data["facets"],
            limit=data.get("limit", limit),
            offset=data.get("offset", offset),
        )
    except Exception:
        logger.exception("graph.search failed", q=q, label=label, offset=offset)
        raise HTTPException(status_code=500, detail="Search failed")


# --- Node details ------------------------------------------------------------

@router.get("/node", response_model=NodeDTO)
async def get_node(
    id: str = Query(..., description="Node id (e.g. DOI for Papers, stable hash for Methods/Cohorts)"),
):
    """Fetch full properties and 1-hop degree for a single node."""
    try:
        node = await asyncio.to_thread(graph_query.get_node, id)
    except Exception:
        logger.exception("graph.get_node failed", node_id=id)
        raise HTTPException(status_code=500, detail="Failed to fetch node")
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node not found: {id}")
    return NodeDTO(**node)


@router.post("/nodes/batch", response_model=NodeBatchResponse)
async def nodes_batch(request: NodeBatchRequest):
    """Fetch many nodes by id in one round-trip. Missing ids are silently skipped."""
    try:
        rows = await asyncio.to_thread(graph_query.get_nodes_batch, request.ids)
    except Exception:
        logger.exception("graph.nodes_batch failed", count=len(request.ids))
        raise HTTPException(status_code=500, detail="Batch fetch failed")
    return NodeBatchResponse(nodes=[NodeDTO(**n) for n in rows])


# --- Neighborhood expansion --------------------------------------------------

@router.get("/neighbors", response_model=SubgraphResponse)
async def neighbors(
    id: str = Query(..., description="Node id to expand around"),
    depth: int = Query(1, ge=1, le=4, description="Hops to expand (1 is the common case)"),
    limit: int = Query(50, ge=1, le=500, description="Max nodes to return (including the root)"),
    rel_types: Optional[str] = Query(
        None,
        description="Comma-separated relationship types to include, e.g. 'SUPPORTS,CONTRADICTS'",
    ),
):
    """Return the subgraph immediately around a node."""
    rel_type_list = _parse_csv_param(rel_types) or None
    try:
        data = await asyncio.to_thread(
            graph_query.get_neighbors, id, depth, limit, rel_type_list
        )
    except Exception:
        logger.exception("graph.neighbors failed", node_id=id, depth=depth)
        raise HTTPException(status_code=500, detail="Neighbor expansion failed")
    if not data["nodes"]:
        raise HTTPException(status_code=404, detail=f"Node not found: {id}")
    return _to_subgraph_response(data)


# --- Multi-seed subgraph -----------------------------------------------------

@router.get("/subgraph", response_model=SubgraphResponse)
async def subgraph(
    seeds: str = Query(..., description="Comma-separated seed node ids"),
    depth: int = Query(1, ge=1, le=4),
    limit: int = Query(200, ge=1, le=1000),
    rel_types: Optional[str] = Query(None),
):
    """Return the induced subgraph around multiple seed nodes."""
    seed_ids = _parse_csv_param(seeds)
    if not seed_ids:
        raise HTTPException(status_code=400, detail="At least one seed id is required")
    # See /neighbors: absent/empty rel_types param means "no filter" (None).
    rel_type_list = _parse_csv_param(rel_types) or None
    try:
        data = await asyncio.to_thread(
            graph_query.get_subgraph, seed_ids, depth, limit, rel_type_list
        )
    except Exception:
        logger.exception("graph.subgraph failed", seed_count=len(seed_ids))
        raise HTTPException(status_code=500, detail="Subgraph expansion failed")
    return _to_subgraph_response(data)


# --- Paths -------------------------------------------------------------------

@router.get("/paths", response_model=PathResponse)
async def paths(
    from_id: str = Query(..., description="Starting node id"),
    to_id: str = Query(..., description="Target node id"),
    max_depth: int = Query(4, ge=1, le=6),
):
    """Return the single shortest undirected path between two nodes (or found=false)."""
    try:
        result = await asyncio.to_thread(graph_query.shortest_path, from_id, to_id, max_depth)
    except Exception:
        logger.exception("graph.shortest_path failed", from_id=from_id, to_id=to_id)
        raise HTTPException(status_code=500, detail="Path query failed")
    if result is None:
        return PathResponse(found=False)
    return PathResponse(
        found=True,
        nodes=[NodeDTO(**n) for n in result["nodes"]],
        edges=[EdgeDTO(**e) for e in result["edges"]],
        length=len(result["edges"]),
    )


@router.get("/paths/stream")
async def paths_stream(
    from_id: str = Query(..., description="Starting node id"),
    to_id: str = Query(..., description="Target node id"),
    max_depth: int = Query(4, ge=1, le=6),
    limit: int = Query(20, ge=1, le=100),
):
    """
    Stream multiple paths (shortest first) via Server-Sent Events.

    Event types:
      - event: path  data: {"nodes":[...], "edges":[...], "length": n}
      - event: done  data: {"total": n}
      - event: error data: {"message": "..."}
    """

    async def event_generator():
        sent = 0
        error_occurred = False
        try:
            # Run the blocking Neo4j iteration in a worker thread; push each
            # path through an asyncio.Queue so the HTTP response stays async.
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()
            sentinel = object()

            def producer():
                try:
                    for path in graph_query.stream_paths(from_id, to_id, max_depth, limit):
                        asyncio.run_coroutine_threadsafe(queue.put(path), loop).result()
                except Exception as exc:
                    asyncio.run_coroutine_threadsafe(
                        queue.put({"__error__": str(exc)}), loop
                    ).result()
                finally:
                    asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop).result()

            task = asyncio.create_task(asyncio.to_thread(producer))

            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, dict) and "__error__" in item:
                    error_occurred = True
                    yield f"event: error\ndata: {json.dumps({'message': item['__error__']})}\n\n"
                    break
                yield f"event: path\ndata: {json.dumps(item)}\n\n"
                sent += 1

            await task
            if not error_occurred:
                yield f"event: done\ndata: {json.dumps({'total': sent})}\n\n"
        except Exception as exc:
            logger.exception("graph.paths_stream failed", from_id=from_id, to_id=to_id)
            yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- Export ------------------------------------------------------------------

@router.post("/export", response_model=SubgraphResponse)
async def export(request: ExportRequest):
    """Return the induced subgraph on the provided set of node ids (no expansion)."""
    try:
        data = await asyncio.to_thread(graph_query.export_subgraph, request.node_ids)
    except Exception:
        logger.exception("graph.export failed", count=len(request.node_ids))
        raise HTTPException(status_code=500, detail="Export failed")
    return _to_subgraph_response({**data, "truncated": {}})


# --- Schema / stats ----------------------------------------------------------

@router.get("/schema", response_model=SchemaResponse)
async def schema():
    """Catalog of labels and relationship types present in the graph."""
    try:
        data = await asyncio.to_thread(graph_query.get_schema)
    except Exception:
        logger.exception("graph.schema failed")
        raise HTTPException(status_code=500, detail="Schema fetch failed")
    return SchemaResponse(**data)


@router.get("/stats", response_model=StatsResponse)
async def stats():
    """Node and relationship counts, grouped by label / type."""
    try:
        data = await asyncio.to_thread(graph_query.get_stats)
    except Exception:
        logger.exception("graph.stats failed")
        raise HTTPException(status_code=500, detail="Stats fetch failed")
    return StatsResponse(**data)
