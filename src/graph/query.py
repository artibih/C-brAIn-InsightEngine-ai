"""
Read-only Cypher queries for serving the Knowledge Graph to a frontend.

All functions return plain dicts shaped as {nodes: [...], edges: [...], ...}
so they can be serialized to JSON without any transformation in the route layer.

Security: Relationship type parameters are always validated against
src.graph.schema.relationships.RelationType before being interpolated into
Cypher. Never pass a raw string from user input into Cypher outside this module.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from src.graph.main import get_driver
from src.graph.schema.relationships import RelationType, is_valid_relation_type

logger = logging.getLogger(__name__)

LABEL_DISPLAY_FIELD: dict[str, str] = {
    "Paper": "name",
    "Claim": "text",
    "Method": "name",
    "Cohort": "group_name",
    "Result": "description",
    "Experiment": "id",
}

SEARCHABLE_LABELS: tuple[str, ...] = tuple(LABEL_DISPLAY_FIELD.keys())
_SEARCH_PER_LABEL_SCAN_CAP: int = 100

def _validate_rel_types(rel_types: Optional[Iterable[str]]) -> Optional[list[str]]:
    """Whitelist rel types against the RelationType enum."""

    if rel_types is None:
        return None
    valid: list[str] = []
    for rt in rel_types:
        if is_valid_relation_type(rt):
            valid.append(rt)
        else:
            logger.warning("Ignoring invalid relationship type: %s", rt)
    return valid


def _build_rel_filter(rel_types: Optional[list[str]]) -> str:
    """Build an APOC relationshipFilter string.
    """
    if not rel_types:
        return ""
    return "|".join(rel_types)


def _derive_node_id(node: Any) -> str:
    """Resolve a stable string id for a Neo4j Node.
    """
    raw_id = node.get("id")
    if raw_id is not None:
        return str(raw_id)
    element_id = getattr(node, "element_id", None)
    if element_id is not None:
        return str(element_id)
    raise ValueError(f"Cannot derive id for node with labels {list(node.labels)}")


def _paper_url_from_doi(doi: str) -> str:
    """Build the canonical https://doi.org/<doi> resolver URL for a DOI."""
    return f"https://doi.org/{doi.strip()}"


def _enrich_paper_properties(
    label: Optional[str], properties: dict[str, Any]
) -> dict[str, Any]:
    """
    Ensure Paper nodes expose `paper_url` in their `properties` dict.
    """
    if label != "Paper":
        return properties
    if properties.get("paper_url"):
        return properties
    doi = properties.get("doi")
    if doi:
        properties["paper_url"] = _paper_url_from_doi(str(doi))
    return properties


def _node_to_dict(node: Any) -> dict[str, Any]:
    """Convert a Neo4j Node object to a JSON-safe dict."""
    labels = list(node.labels)
    label = labels[0] if labels else None
    return {
        "id": _derive_node_id(node),
        "label": label,
        "properties": _enrich_paper_properties(label, dict(node)),
    }


def _rel_to_dict(rel: Any) -> dict[str, Any]:
    """Convert a Neo4j Relationship object to a JSON-safe dict."""
    return {
        "id": str(rel.element_id),
        "source": _derive_node_id(rel.start_node),
        "target": _derive_node_id(rel.end_node),
        "type": rel.type,
        "properties": dict(rel),
    }


def search_nodes(
    query: str,
    label: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """
    Case-insensitive substring search across node display fields.
    If label is provided (and valid), restrict to that label.

    Implemented as two Cypher round-trips:
      1. A cheap count-only query that gives an *exact* `total` and per-label
         `facets`. No degree expansion, just label scan + substring filter.
      2. A bounded results query that caps each label's scan at
         max(`_SEARCH_PER_LABEL_SCAN_CAP`, offset + limit) rows before
         computing degree, then orders globally by degree DESC and applies
         SKIP $offset / LIMIT $limit for the requested page.

    Pagination semantics: `offset` is the number of results to skip; the page
    returned is the `[offset, offset + limit)` slice of the globally ordered
    result set. Clients drive a "load more" affordance by bumping `offset` by
    `limit` until `offset + len(results) >= total`.

    The per-label scan cap grows with `offset + limit` so deep pages still
    have enough candidates to fill the page. For pathological wildcard
    queries (e.g. q="a") the `results` for any given page are still drawn
    from a bounded per-label sample rather than a true global top-N; the
    caller gets accurate `total` and `facets` and is expected to prompt the
    user to refine.

    Response shape:
      {
        "results": [{id, label, name, degree}, ...],   # at most `limit`
        "total": int,                                   # exact
        "facets": {label: count, ...},                  # exact
        "limit": int,                                   # echoed
        "offset": int,                                  # echoed
      }
    """
    limit = max(1, limit)
    offset = max(0, offset)
    empty_page = {
        "results": [],
        "total": 0,
        "facets": {},
        "limit": limit,
        "offset": offset,
    }
    if not query or not query.strip():
        return empty_page

    sanitized_query = query.strip()

    labels = [label] if (label and label in LABEL_DISPLAY_FIELD) else list(SEARCHABLE_LABELS)

    per_label_cap = max(_SEARCH_PER_LABEL_SCAN_CAP, offset + limit)


    count_fragments: list[str] = []
    result_fragments: list[str] = []
    for lbl in labels:
        field = LABEL_DISPLAY_FIELD[lbl]
        count_fragments.append(
            f"MATCH (n:`{lbl}`) "
            f"WHERE toLower(coalesce(n.`{field}`, '')) CONTAINS toLower($q) "
            f"RETURN '{lbl}' AS label"
        )
        result_fragments.append(
            f"MATCH (n:`{lbl}`) "
            f"WHERE toLower(coalesce(n.`{field}`, '')) CONTAINS toLower($q) "
            f"WITH n ORDER BY n.id ASC LIMIT {per_label_cap} "
            f"RETURN n.id AS id, '{lbl}' AS label, "
            f"coalesce(n.`{field}`, n.id) AS name, "
            f"size([(n)--() | 1]) AS degree"
        )

    count_cypher = (
        "CALL { " + " UNION ALL ".join(count_fragments) + " } "
        "RETURN label, count(*) AS count"
    )
    results_cypher = (
        "CALL { " + " UNION ".join(result_fragments) + " } "
        "RETURN id, label, name, degree "
        "ORDER BY degree DESC, id ASC "
        "SKIP $offset "
        "LIMIT $limit"
    )

    driver = get_driver()
    with driver.session() as session:
        count_rows = session.run(count_cypher, q=sanitized_query)
        facets: dict[str, int] = {row["label"]: row["count"] for row in count_rows}
        total = sum(facets.values())

        if total == 0 or offset >= total:
            return {
                "results": [],
                "total": total,
                "facets": facets,
                "limit": limit,
                "offset": offset,
            }

        result_rows = session.run(
            results_cypher, q=sanitized_query, offset=offset, limit=limit
        )
        results = [dict(record) for record in result_rows]

    return {
        "results": results,
        "total": total,
        "facets": facets,
        "limit": limit,
        "offset": offset,
    }


def get_node(node_id: str) -> Optional[dict[str, Any]]:
    """Fetch a single node by id with its degree. Returns None if not found."""
    cypher = (
        "MATCH (n {id: $id}) "
        "RETURN n.id AS id, head(labels(n)) AS label, "
        "properties(n) AS properties, size([(n)--() | 1]) AS degree"
    )
    driver = get_driver()
    with driver.session() as session:
        record = session.run(cypher, id=node_id).single()
        if record is None:
            return None
        row = dict(record)
        row["properties"] = _enrich_paper_properties(
            row.get("label"), dict(row.get("properties") or {})
        )
        return row


def get_nodes_batch(node_ids: list[str]) -> list[dict[str, Any]]:
    """Fetch many nodes by id in a single query. Missing ids are silently skipped."""
    if not node_ids:
        return []
    cypher = (
        "MATCH (n) WHERE n.id IN $ids "
        "RETURN n.id AS id, head(labels(n)) AS label, "
        "properties(n) AS properties, size([(n)--() | 1]) AS degree"
    )
    driver = get_driver()
    with driver.session() as session:
        result = session.run(cypher, ids=node_ids)
        rows: list[dict[str, Any]] = []
        for record in result:
            row = dict(record)
            row["properties"] = _enrich_paper_properties(
                row.get("label"), dict(row.get("properties") or {})
            )
            rows.append(row)
        return rows


def get_neighbors(
    node_id: str,
    depth: int = 1,
    limit: int = 50,
    rel_types: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Return the subgraph around `node_id` up to `depth` hops, capped at `limit` nodes.
    Direction-agnostic (both incoming and outgoing edges).

    Response shape:
      {
        "nodes": [{id, label, properties, degree}, ...],
        "edges": [{id, source, target, type, properties}, ...],
        "truncated": {"nodes": bool, "hidden_neighbors": int},
      }
    """
    depth = max(1, min(depth, 4))
    limit = max(1, min(limit, 500))
    valid_rels = _validate_rel_types(rel_types)

    driver = get_driver()
    with driver.session() as session:
        root = session.run(
            "MATCH (n {id: $id}) "
            "RETURN n, size([(n)--() | 1]) AS total_degree",
            id=node_id,
        ).single()
        if root is None:
            return {"nodes": [], "edges": [], "truncated": {"nodes": False, "hidden_neighbors": 0}}

        root_node = root["n"]
        total_degree = root["total_degree"]

        if valid_rels == []:
            return {
                "nodes": [_node_to_dict(root_node)],
                "edges": [],
                "truncated": {"nodes": False, "hidden_neighbors": total_degree},
            }

        rel_filter = _build_rel_filter(valid_rels)

        cypher = (
            "MATCH (n {id: $id}) "
            "CALL apoc.path.subgraphAll(n, { "
            "  maxLevel: $depth, "
            "  relationshipFilter: $rel_filter, "
            "  limit: $limit "
            "}) YIELD nodes, relationships "
            "RETURN nodes, relationships"
        )
        record = session.run(
            cypher,
            id=node_id,
            depth=depth,
            rel_filter=rel_filter,
            limit=limit,
        ).single()

        if record is None:
            nodes_raw, rels_raw = [root_node], []
        else:
            nodes_raw = record["nodes"] or [root_node]
            rels_raw = record["relationships"] or []

        nodes = [_node_to_dict(n) for n in nodes_raw]
        edges = [_rel_to_dict(r) for r in rels_raw]

        returned_neighbors = max(0, len(nodes) - 1)
        hidden = max(0, total_degree - returned_neighbors) if depth == 1 else 0
        truncated = len(nodes) >= limit or hidden > 0

        return {
            "nodes": nodes,
            "edges": edges,
            "truncated": {"nodes": truncated, "hidden_neighbors": hidden},
        }



def get_subgraph(
    seed_ids: list[str],
    depth: int = 1,
    limit: int = 200,
    rel_types: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Return the induced subgraph around multiple seed nodes."""
    if not seed_ids:
        return {"nodes": [], "edges": [], "truncated": {"nodes": False, "hidden_neighbors": 0}}

    depth = max(1, min(depth, 4))
    limit = max(1, min(limit, 1000))
    valid_rels = _validate_rel_types(rel_types)
    if valid_rels == []:
        return {"nodes": [], "edges": [], "truncated": {"nodes": False, "hidden_neighbors": 0}}
    rel_filter = _build_rel_filter(valid_rels)

    cypher = (
        "MATCH (n) WHERE n.id IN $seeds "
        "WITH collect(n) AS seeds "
        "CALL apoc.path.subgraphAll(seeds, { "
        "  maxLevel: $depth, "
        "  relationshipFilter: $rel_filter, "
        "  limit: $limit "
        "}) YIELD nodes, relationships "
        "RETURN nodes, relationships"
    )
    driver = get_driver()
    with driver.session() as session:
        record = session.run(
            cypher, seeds=seed_ids, depth=depth, rel_filter=rel_filter, limit=limit
        ).single()
        if record is None:
            return {"nodes": [], "edges": [], "truncated": {"nodes": False, "hidden_neighbors": 0}}

        nodes = [_node_to_dict(n) for n in record["nodes"] or []]
        edges = [_rel_to_dict(r) for r in record["relationships"] or []]
        return {
            "nodes": nodes,
            "edges": edges,
            "truncated": {"nodes": len(nodes) >= limit, "hidden_neighbors": 0},
        }



def shortest_path(from_id: str, to_id: str, max_depth: int = 4) -> Optional[dict[str, Any]]:
    """Find the shortest undirected path between two nodes. Returns None if no path."""
    max_depth = max(1, min(max_depth, 6))

    cypher = (
        "MATCH (a {id: $from_id}) "
        "MATCH (b {id: $to_id}) "
        f"MATCH path = shortestPath((a)-[*..{max_depth}]-(b)) "
        "RETURN nodes(path) AS nodes, relationships(path) AS rels"
    )
    driver = get_driver()
    with driver.session() as session:
        record = session.run(cypher, from_id=from_id, to_id=to_id).single()
        if record is None:
            return None
        return {
            "nodes": [_node_to_dict(n) for n in record["nodes"]],
            "edges": [_rel_to_dict(r) for r in record["rels"]],
        }


def stream_paths(
    from_id: str,
    to_id: str,
    max_depth: int = 4,
    limit: int = 20,
):
    """
    Generator yielding individual paths as they are discovered.
    Each yielded item is {"nodes": [...], "edges": [...], "length": int}.
    Caller is responsible for SSE framing.
    """
    max_depth = max(1, min(max_depth, 6))
    limit = max(1, min(limit, 100))
    cypher = (
        "MATCH (a {id: $from_id}) "
        "MATCH (b {id: $to_id}) "
        f"MATCH path = (a)-[*..{max_depth}]-(b) "
        "RETURN nodes(path) AS nodes, relationships(path) AS rels, length(path) AS length "
        "ORDER BY length(path) ASC "
        "LIMIT $limit"
    )
    driver = get_driver()
    with driver.session() as session:
        result = session.run(cypher, from_id=from_id, to_id=to_id, limit=limit)
        materialized = [
            {
                "nodes": [_node_to_dict(n) for n in record["nodes"]],
                "edges": [_rel_to_dict(r) for r in record["rels"]],
                "length": record["length"],
            }
            for record in result
        ]

    for path in materialized:
        yield path



def get_schema() -> dict[str, Any]:
    """Return the catalog of labels, relationship types, and property keys."""
    driver = get_driver()
    with driver.session() as session:
        labels_result = session.run("CALL db.labels() YIELD label RETURN label")
        labels = [r["label"] for r in labels_result]

        rels_result = session.run(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        )
        rel_types_db = [r["relationshipType"] for r in rels_result]

    return {
        "labels": sorted(labels),
        "relationship_types": sorted(rel_types_db),
        "ontology_relationship_types": [rt.value for rt in RelationType],
        "searchable_labels": list(SEARCHABLE_LABELS),
        "label_display_fields": LABEL_DISPLAY_FIELD,
    }


def get_stats() -> dict[str, Any]:
    """Return node counts per label and relationship counts per type."""
    driver = get_driver()
    with driver.session() as session:
        node_rows = session.run(
            "MATCH (n) UNWIND labels(n) AS lbl "
            "RETURN lbl AS label, count(*) AS count ORDER BY count DESC"
        )
        node_counts = {r["label"]: r["count"] for r in node_rows}

        rel_rows = session.run(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
        )
        rel_counts = {r["type"]: r["count"] for r in rel_rows}

        node_total_row = session.run(
            "MATCH (n) RETURN count(n) AS node_total"
        ).single()
        rel_total_row = session.run(
            "MATCH ()-[r]->() RETURN count(r) AS rel_total"
        ).single()

    return {
        "node_total": node_total_row["node_total"] if node_total_row else 0,
        "relationship_total": rel_total_row["rel_total"] if rel_total_row else 0,
        "nodes_by_label": node_counts,
        "relationships_by_type": rel_counts,
    }



def export_subgraph(node_ids: list[str]) -> dict[str, Any]:
    """
    Given a set of node ids (typically the user's current visible canvas),
    return all nodes and the edges that connect them (no expansion).
    """
    if not node_ids:
        return {"nodes": [], "edges": []}

    cypher = (
        "MATCH (n) WHERE n.id IN $ids "
        "WITH collect(n) AS nodes "
        "UNWIND nodes AS a "
        "OPTIONAL MATCH (a)-[r]->(b) WHERE b IN nodes "
        "RETURN nodes AS all_nodes, collect(DISTINCT r) AS all_rels"
    )
    driver = get_driver()
    with driver.session() as session:
        record = session.run(cypher, ids=node_ids).single()
        if record is None:
            return {"nodes": [], "edges": []}
        nodes = [_node_to_dict(n) for n in record["all_nodes"] or []]
        edges = [_rel_to_dict(r) for r in record["all_rels"] or [] if r is not None]
        return {"nodes": nodes, "edges": edges}
