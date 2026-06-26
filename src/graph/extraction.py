"""
Hybrid extraction: deterministic experiment clusters + probabilistic Judge (Result–Claim).
Uses Instructor for all LLM interactions.
"""

import logging
import os
from typing import Literal

import instructor
from openai import OpenAI, AzureOpenAI
from pydantic import BaseModel, Field

from config.settings import settings
from src.graph.schema.ontology import ExtractedContent
from src.graph.ingestion import write_judge_edge
from src.graph.prompts import EXTRACT_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

client: OpenAI | None = None


def _use_azure() -> bool:
    return bool(settings.azure_openai_api_key and settings.azure_openai_endpoint)


def get_client() -> OpenAI:
    global client
    if client is None:
        if _use_azure():
            base = AzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
            )
            logger.info("Using Azure OpenAI (endpoint=%s)", settings.azure_openai_endpoint)
        else:
            base = OpenAI()
            logger.info("Using standard OpenAI")
        client = instructor.patch(base, mode=instructor.Mode.JSON)
    return client


def extraction_model() -> str:
    """Model/deployment for extraction. Env override: GRAPH_EXTRACTION_MODEL."""
    default = settings.azure_openai_deployment_graph if _use_azure() else "gpt-4o"
    return os.environ.get("GRAPH_EXTRACTION_MODEL", default)


def judge_model() -> str:
    """Model/deployment for Judge. Env override: GRAPH_JUDGE_MODEL; falls back to extraction_model()."""
    return os.environ.get("GRAPH_JUDGE_MODEL") or extraction_model()

class Verdict(BaseModel):
    """Judge output: relationship of a Result to a Claim, with reasoning."""
    relationship: Literal["SUPPORTS", "CONTRADICTS", "NEUTRAL"] = Field(
        ...,
        description="SUPPORTS only if the result strongly supports the claim; CONTRADICTS only if it strongly contradicts; otherwise NEUTRAL."
    )
    reason: str = Field(
        ...,
        description="Brief explanation for the verdict. Required especially for SUPPORTS and CONTRADICTS."
    )


def extract_content(text_chunk: str) -> ExtractedContent:
    """Extract experiments and claims from text using Instructor."""
    logger.info("Extracting structured data from text...")
    client = get_client()
    logger.info(f"Extracting entities with model {extraction_model()}")
    return client.chat.completions.create(
        model=extraction_model(),
        response_model=ExtractedContent,
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": text_chunk},
        ],
        temperature=0
    )

def judge_and_link(tx, content: ExtractedContent, paper_id: str) -> list[tuple[str, str, str, str]]:
    """
    For each (Result, Claim) pair, ask the LLM to judge. Create a SUPPORTS or CONTRADICTS
    edge only when the verdict is not NEUTRAL; store the LLM's reason on the edge.
    Returns list of (result_id, claim_id, relationship, reason) for tabular export.

    IDs are scoped to paper_id to match what ingest_deterministic_graph writes.
    """
    from src.graph.ingestion import _scoped_id

    client = get_client()
    written: list[tuple[str, str, str, str]] = []
    logger.info(f"Judging relationships with model {judge_model()}")
    for exp in content.experiments:
        scoped_eid = _scoped_id(paper_id, exp.experiment_id)
        result_id = exp.result.get_stable_id(scoped_eid)
        trend_str = exp.result.trend.value
        p_val = exp.result.p_value or "not reported"

        for claim in content.claims:
            scoped_cid = _scoped_id(paper_id, claim.claim_id)
            prompt = f"""Claim: "{claim.text}"

Result: "{exp.result.description}"
Trend: {trend_str}. Statistical significance: {p_val}.

Does this result SUPPORT, CONTRADICT, or is it NEUTRAL with respect to the claim? Be conservative: only SUPPORT or CONTRADICT if the evidence is strong."""

            verdict = client.chat.completions.create(
                model=judge_model(),
                response_model=Verdict,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )

            if verdict.relationship in ("SUPPORTS", "CONTRADICTS"):
                logger.info(f"   Result {exp.experiment_id} {verdict.relationship} Claim {claim.claim_id}: {verdict.reason[:60]}...")
                write_judge_edge(tx, result_id, scoped_cid, verdict.relationship, verdict.reason)
                written.append((result_id, scoped_cid, verdict.relationship, verdict.reason))
    return written
