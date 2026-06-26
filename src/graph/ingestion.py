"""
Idempotent ingestion of extracted content into Neo4j.
Paper is linked to all Experiments and Claims; re-running on the same paper does not duplicate nodes.
"""

import logging

from src.graph.schema.ontology import ExtractedContent
from src.graph.schema.relationships import RelationType, JUDGE_RELATION_TYPES

logger = logging.getLogger(__name__)

def _scoped_id(pid: str, local_id: str) -> str:
    """Prefix a paper-local ID (e.g. 'Exp_1') with the paper ID to make it globally unique."""
    if local_id.startswith(f"{pid}_"):
        return local_id
    return f"{pid}_{local_id}"


def ingest_deterministic_graph(
    tx,
    content: ExtractedContent,
    paper_id: str,
    doi: str | None = None,
    name: str | None = None,
) -> None:
    """
    Ingest atomic experiment clusters and claims into Neo4j. Idempotent:
    running twice on the same paper (same DOI or paper_id) does not create duplicate nodes or relationships.
    When doi is provided, it is used as the Paper node id (canonical identifier).

    Experiment and Claim IDs are scoped to the paper (e.g. 'Exp_1' -> '{pid}_Exp_1')
    to prevent collisions across papers.
    """
    pid = doi if doi else paper_id
    logger.info(f"Ingesting {len(content.experiments)} experiments and {len(content.claims)} claims into Neo4j (paper id={pid})...")

    tx.run(
        "MERGE (p:Paper {id: $pid}) SET p.id = $pid, p.doi = $doi, p.name = $name",
        pid=pid,
        doi=doi or "",
        name=name or "",
    )

    for claim in content.claims:
        scoped_cid = _scoped_id(pid, claim.claim_id)
        tx.run(
            """
            MERGE (p:Paper {id: $pid})
            MERGE (c:Claim {id: $cid})
            SET c.text = $text, c.status = $status
            MERGE (p)-[:MAKES_CLAIM]->(c)
            """,
            pid=pid,
            cid=scoped_cid,
            text=claim.text,
            status=claim.status,
        )

    for exp in content.experiments:
        scoped_eid = _scoped_id(pid, exp.experiment_id)
        rid = exp.result.get_stable_id(scoped_eid)
        mid = exp.method.stable_id
        cid = exp.cohort.stable_id
        trend_val = exp.result.trend.value

        tx.run(
            """
            MERGE (p:Paper {id: $pid})
            MERGE (e:Experiment {id: $eid})
            MERGE (p)-[:DESCRIBES_EXP]->(e)

            MERGE (m:Method {id: $mid})
            SET m.name = $m_name, m.parameters = $m_params
            MERGE (e)-[:USED_METHOD]->(m)

            MERGE (c:Cohort {id: $cid})
            SET c.group_name = $c_name, c.species = $c_species, c.characteristics = $c_char, c.sample_size = $c_n
            MERGE (e)-[:USED_SAMPLE]->(c)

            MERGE (r:Result {id: $rid})
            SET r.description = $r_desc, r.p_value = $r_pval, r.trend = $r_trend
            MERGE (e)-[:YIELDED]->(r)
            """,
            pid=pid,
            eid=scoped_eid,
            mid=mid,
            m_name=exp.method.name,
            m_params=exp.method.parameters or "",
            cid=cid,
            c_name=exp.cohort.group_name,
            c_species=exp.cohort.species,
            c_char=exp.cohort.characteristics or "",
            c_n=exp.cohort.sample_size,
            rid=rid,
            r_desc=exp.result.description,
            r_pval=exp.result.p_value or "",
            r_trend=trend_val,
        )


def write_judge_edge(tx, result_id: str, claim_id: str, relation_type: str, reason: str) -> None:
    """
    Create a single Result->Claim edge with a reason property (SUPPORTS, CONTRADICTS, or INCONCLUSIVE).
    Validates relation_type against RelationType to prevent Cypher injection.
    Requires APOC plugin for dynamic relationship creation.
    """
    try:
        rel_type = RelationType(relation_type)
    except ValueError:
        raise ValueError(f"Invalid relationship type for judge edge: {relation_type}")
    if rel_type not in JUDGE_RELATION_TYPES:
        raise ValueError("Judge edge must be SUPPORTS, CONTRADICTS, or INCONCLUSIVE")

    tx.run(
        """
        MATCH (r:Result {id: $rid})
        MATCH (c:Claim {id: $cid})
        CALL apoc.merge.relationship(r, $rel_type, {}, {reason: $reason}, c, {reason: $reason})
        YIELD rel
        RETURN count(rel)
        """,
        rid=result_id,
        cid=claim_id,
        rel_type=relation_type,
        reason=reason or "",
    )
    
