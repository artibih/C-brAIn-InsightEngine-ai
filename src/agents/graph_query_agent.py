"""
GraphQueryAgent: Natural language -> Cypher -> Neo4j -> Markdown table.

Key features:
- Dynamic schema retrieval.
- Text-to-Cypher using OpenAI by default.
- Read-only safety guardrails (blocks write keywords).
- Returns results formatted as a Markdown table.
"""
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any
import pandas as pd
from neo4j import AsyncGraphDatabase
from openai import AsyncAzureOpenAI
from config.settings import settings


import structlog

logger = structlog.get_logger(__name__)

FENCE_RE = re.compile(r"^```(?:cypher)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class _SchemaCache:
    value: str
    fetched_at: float


class GraphQueryAgent:
    """
    Converts natural-language questions into Cypher, executes against Neo4j,
    and returns results as a Markdown table.
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        *,
        azure_openai_api_key: str | None = None,
        schema_ttl_seconds: int = 300,
        default_limit: int = 25,
    ) -> None:
        """
        Args:
            neo4j_uri: Neo4j URI (e.g., bolt://localhost:7687)
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            azure_openai_api_key: Azure OpenAI API key
            schema_ttl_seconds: Cache TTL for schema retrieval
            default_limit: Default LIMIT to encourage bounded queries
        """


        logger.info(
            "graph_query_agent_initializing"
        )
        if not neo4j_uri or not neo4j_user:
            logger.error(
                "graph_query_agent_missing_neo4j_uri_or_user",
            )
            raise ValueError("neo4j_uri and neo4j_user are required")
        if not neo4j_password:
            logger.error(
                "graph_query_agent_missing_neo4j_password",
            )
            raise ValueError("neo4j_password is required")

        self.driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        if not azure_openai_api_key:
            logger.error(
                "graph_query_agent_missing_azure_openai_api_key",
            )
            raise ValueError("azure_openai_api_key is required")
        self.openai = AsyncAzureOpenAI(
            api_key=azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        self.model = settings.azure_openai_deployment_graph
        self.schema_ttl_seconds = int(schema_ttl_seconds)
        self.default_limit = int(default_limit)
        self.schema_cache: _SchemaCache | None = None

        logger.info(
            "graph_query_agent_initialized",
            model=self.model,
        )
    
    async def close(self):
        logger.info(
            "graph_query_agent_closing_driver",
        )
        await self.driver.close()
        logger.info(
            "graph_query_agent_driver_closed",
        )
        await self.openai.close()

    async def get_schema(self, *, force_refresh: bool = False) -> str:
        """
        Retrieve a schema summary string for prompt injection.
        """
        logger.info(
            "graph_get_schema_started",
            force_refresh=force_refresh,
        )
        now = time.time()
        if (
            not force_refresh
            and self.schema_cache is not None
            and (now - self.schema_cache.fetched_at) < self.schema_ttl_seconds
        ):
            logger.info(
                "graph_get_schema_cache_hit",
                cached_schema_length=len(self.schema_cache.value or ""),
            )
            return self.schema_cache.value

        try:
            async with self.driver.session() as session:
                schema = await self.schema_via_apoc(session)
        except Exception as e:
            logger.exception(
                "graph_get_schema_failed",
                error=str(e),
            )
            raise

        schema = schema.strip()
        self.schema_cache = _SchemaCache(value=schema, fetched_at=now)

        logger.info(
            "graph_get_schema_completed",
            schema_length=len(schema or ""),
        )
        return schema

    async def schema_via_apoc(self, session) -> str | None:
        """Try `CALL apoc.meta.schema()`; return None if APOC isn't available."""

        try:
            result = await session.run(
            "CALL apoc.meta.schema() YIELD value RETURN value"
            )
            record = await result.single()
            value = record["value"] if record else None
            logger.info(
                "graph_schema_via_apoc_completed",
                has_schema=bool(value)
            )
            return "APOC schema (apoc.meta.schema):\n" + json.dumps(value, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception(
                "graph_schema_via_apoc_failed",
                error=str(e),
            )
            raise RuntimeError(f"Failed to extract graph schema: {e}") from e


    async def generate_cypher(self, question: str, experiment_id: str) -> str:
        """
        Generate a read-only Cypher query from a natural-language question.

        The prompt includes the dynamically retrieved schema and strict constraints:
        - Output ONLY Cypher.
        - Read-only: no CREATE/MERGE/DELETE/DETACH/SET/DROP/etc.
        """
        logger.debug(
            "graph_generate_cypher_started",
            experiment_id=experiment_id,
            question=question,
        )
        if not question or not question.strip():
            logger.error(
                "graph_generate_cypher_empty_question",
                experiment_id=experiment_id,
            )
            raise ValueError("question must be a non-empty string")

        schema = await self.get_schema()
        system_prompt = (
            "You are a Neo4j Cypher expert. Your job is to write a Cypher query that answers the user's question.\n\n"
            "QUERY STRATEGY:\n"
            "This graph stores structured experimental evidence extracted from biomedical papers.\n"
            "Your queries should prioritize the rich experimental structure over plain Paper/Claim lookups:\n"
            "- PREFER traversing: Experiment, Method, Cohort, Result nodes and their relationships.\n"
            "  These contain specific methods used, cohort characteristics (species, sample size), "
            "quantitative results (p-values, trends, effect descriptions), and judge verdicts.\n"
            "- Use Paper and Claim nodes mainly to JOIN context (paper title, DOI) onto experimental results, "
            "not as the primary query target.\n"
            "- A separate RAG system already handles finding relevant papers by text similarity. "
            "Your unique value is returning STRUCTURED experimental evidence: which methods were used, "
            "on which cohorts, with what results.\n"
            "- When the question mentions a topic, search for it across Method.name, Result.description, "
            "Cohort.characteristics, and Claim.text — not just Paper.name.\n"
            "- Always include the Paper DOI or title when returning experimental data so results can be traced.\n\n"
            "STRICT RULES:\n"
            "- Output ONLY the Cypher query text. No explanations. No markdown. No code fences.\n"
            "- Generate READ-ONLY queries only. Use MATCH / OPTIONAL MATCH / WITH / WHERE / RETURN.\n"
            "- NEVER generate write or admin operations: CREATE, MERGE, DELETE, DETACH, SET, DROP, REMOVE, "
            "CALL db.* (except schema is already provided), CALL apoc.* , LOAD CSV, CREATE INDEX, DROP INDEX.\n"
            f"- Always include a LIMIT {self.default_limit} unless the user explicitly asks for all results.\n"
            "- Prefer returning human-readable fields (e.g., .id, .text, .description) and include column aliases.\n"
            "- Arrow syntax: use (a)-[:REL]->(b) for outgoing and (a)<-[:REL]-(b) for incoming. "
            "NEVER put the arrow after the brackets like -[:REL]<- as that is invalid.\n"
            "- For string-valued properties, always use case-insensitive matching: "
            "use toLower(property) in WHERE filters and CONTAINS checks "
            "(e.g., WHERE toLower(c.species) = 'mouse', WHERE toLower(m.name) CONTAINS 'western blot'). "
            "Do not apply toLower to numeric or date fields such as years, sample sizes, or p-values; "
            "use exact or numeric comparisons for those.\n"
            "- If the question is ambiguous, make a reasonable assumption and produce the best read-only query.\n\n"
            "GRAPH SCHEMA:\n"
            f"{schema}\n"
        )

        user_prompt = f"USER QUESTION:\n{question}\n\nCYPHER:"

        try:
            raw = await self.llm_generate(system_prompt, user_prompt, experiment_id=experiment_id)
            cypher = raw.strip()
        except Exception as e:
            logger.exception(
                "graph_generate_cypher_llm_failed",
                experiment_id=experiment_id,
                error=str(e),
            )
            raise RuntimeError("Failed to generate query") from e

        cypher = self.clean_cypher(cypher, experiment_id=experiment_id)
        self.enforce_readonly(cypher, experiment_id=experiment_id)
        logger.debug(
            "graph_generate_cypher_completed",
            experiment_id=experiment_id,
            cypher=cypher,
        )
        return cypher

    async def llm_generate(self, system_prompt: str, user_prompt: str, experiment_id: str) -> str:
        """Generate raw text from the configured LLM provider."""
        logger.info(
            "graph_llm_generate_started",
            experiment_id=experiment_id,
            model=self.model,
            system_prompt_length=len(system_prompt or ""),
            user_prompt_length=len(user_prompt or ""),
        )

        try:
            resp = await self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            content = resp.choices[0].message.content or ""

            logger.info(
                "graph_llm_generate_completed",
                experiment_id=experiment_id,
                model=self.model,
                output_length=len(content or ""),
            )
            return content

        except Exception as e:
            logger.exception(
                "graph_llm_generate_failed",
                experiment_id=experiment_id,
                model=self.model,
                error=str(e),
            )
            raise

    def clean_cypher(self, cypher: str, experiment_id: str = "") -> str:
        """Remove accidental markdown fences, fix common LLM Cypher mistakes, and trim whitespace."""
        cypher = (cypher or "").strip()
        if "```" in cypher:
            cypher = FENCE_RE.sub("", cypher).strip()
        cypher = re.sub(r"^\s*cypher\s*:\s*", "", cypher, flags=re.IGNORECASE)
        cypher = re.sub(r'-(\[.*?\])<-', r'<-\1-', cypher)
        return cypher.strip()

    def enforce_readonly(self, cypher: str, experiment_id: str) -> None:
        """Block obvious write/admin operations. Raises ValueError if disallowed."""
        logger.debug(
            "graph_enforce_readonly_started",
            experiment_id=experiment_id,
            cypher=cypher,
        )
        if not cypher:
            logger.error(
                "graph_enforce_readonly_empty_cypher",
                experiment_id=experiment_id,
            )
            raise ValueError("Empty Cypher generated by the model")

        forbidden_pattern = re.compile(
            r'\b(CREATE|MERGE|DELETE|DETACH|DROP|REMOVE)\b'
            r'|(?<![.\w])SET\s+[a-zA-Z]' 
            r'|\bLOAD\s+CSV\b'
            r'|\bCALL\s+(apoc\.|db\.)',
            re.IGNORECASE
        )
        if forbidden_pattern.search(cypher):
            logger.error(
                "graph_enforce_readonly_forbidden_operation",
                experiment_id=experiment_id,
                cypher=cypher,
            )
            raise ValueError(
                "Refusing to run non-read-only Cypher. Generated query contains forbidden operation."
            )

        if not re.match(r"^\s*(MATCH|OPTIONAL\s+MATCH|WITH)\b", cypher, flags=re.IGNORECASE):
            logger.error(
                "graph_enforce_readonly_invalid_start",
                experiment_id=experiment_id,
                cypher=cypher,
            )
            raise ValueError(
                "Refusing to run Cypher that does not start with MATCH/OPTIONAL MATCH/WITH."
            )

    async def run(self, question: str, experiment_id: str = "") -> str:
        logger.debug(
            "graph_run_started",
            experiment_id=experiment_id or None,
            question=question,
        )
        try:
            cypher = await self.generate_cypher(question, experiment_id=experiment_id)
        except ValueError as e:
            logger.warning(
                "graph_run_generate_validation_failed",
                experiment_id=experiment_id or None,
                error=str(e),
            )
            return str(e)
        except Exception:
            logger.exception(
                "graph_run_generate_failed",
                experiment_id=experiment_id or None,
            )
            return "Failed to generate query"

        logger.debug(
            "graph_run_cypher_generated",
            experiment_id=experiment_id or None,
            cypher=cypher,
        )

        try:
            rows = await self.run_cypher(cypher, experiment_id=experiment_id)
        except Exception:
            logger.exception(
                "graph_run_execute_failed",
                experiment_id=experiment_id or None,
            )
            return "Unexpected error executing query"

        logger.info("GraphQueryAgent | rows returned=%d", len(rows))
        if not rows:
            logger.debug(
                "graph_run_no_results",
                experiment_id=experiment_id,
                cypher=cypher,
            )
            return "No results found for that question."
        
        markdown = pd.DataFrame(rows).to_markdown(index=False)

        logger.info(
            "graph_run_completed",
            experiment_id=experiment_id,
            row_count=len(rows),
            markdown_length=len(markdown or ""),
        )
        return markdown

    async def run_cypher(self, cypher: str, experiment_id: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute read-only Cypher and return rows as list[dict]."""
        params = params or {}
        
        logger.debug(
            "graph_run_cypher_started",
            experiment_id=experiment_id,
            has_params=bool(params),
            cypher=cypher,
        )

        try:
            async with self.driver.session() as session:
                result = await session.run(cypher, parameters=params)
                rows = [record.data() async for record in result]

            logger.info(
                "graph_run_cypher_completed",
                experiment_id=experiment_id,
                row_count=len(rows),
            )
            return rows

        except Exception as e:
            logger.exception(
                "graph_run_cypher_execution_failed",
                experiment_id=experiment_id,
                error=str(e),
                cypher=cypher,
            )
            raise

def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise SystemExit(f"Missing required environment variable: {name}")
    return val

