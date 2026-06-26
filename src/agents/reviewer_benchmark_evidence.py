import asyncio
import json
import re
import time
from typing import Any, Dict, List

import structlog
from langchain_core.prompts import ChatPromptTemplate

from config.llm_config import get_llm
from config.llm_selection import ResolvedLlmSelection
from src.agents.claim_extractor import ClaimExtractor
from src.agents.hallucination_detector import HallucinationDetector
from src.agents.hybrid_query_agent import HybridQueryAgent
from src.prompts.reviewer_benchmark_evidence_prompt import (
    REVIEWER_BENCHMARK_EVIDENCE_PROMPT,
)
from utils.review_parameters import build_review_parameter_guidance

logger = structlog.get_logger()


_DEFAULT_FALLBACK = {
    "summary": "",
    "evidence_grounding": "weak",
    "supported_claims": [],
    "unsubstantiated_claims": [],
    "contradicted_claims": [],
    "replication_findings": [],
    "strengths": [],
    "weaknesses": [],
    "recommendation": "major_revision",
}


class ReviewerBenchmarkEvidence:
    """
    Fourth reviewer: benchmarks manuscript findings against published literature
    and graph evidence, and detects unsubstantiated claims.

    Internal pipeline:
        1) ClaimExtractor (MVP2)               -> claims[]
        2) Per-claim retrieval (parallel):
             - HybridQueryAgent.run            -> graph + vector evidence (MVP2)
             - BosnaRagPipeline.retrieve_with_metadata
                                               -> raw paper chunks with DOIs (MVP1)
        3) HallucinationDetector (MVP2)        -> verdicts[]
        4) Reviewer-format LLM synthesis       -> reviewer JSON
    """

    def __init__(
        self,
        hybrid_agent: HybridQueryAgent,
        llm_selection: ResolvedLlmSelection | None = None,
    ):
        self.hybrid = hybrid_agent
        self.llm_selection = llm_selection
        self.rag = hybrid_agent.rag
        self.llm = get_llm(temperature=0, workload="online", llm_selection=llm_selection)
        self.prompt = ChatPromptTemplate.from_template(
            REVIEWER_BENCHMARK_EVIDENCE_PROMPT
        )

    async def review(
        self,
        sections: Dict[str, Any],
        experiment_id: str,
        feedback: str | None = None,
        previous_reviews: Dict[str, Any] | None = None,
        review_parameters: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        logger.info(
            "reviewer_benchmark_evidence_started",
            experiment_id=experiment_id,
        )

        section_list = self._coerce_sections(sections)

        claims = await self._extract_claims(section_list, experiment_id)

        if not claims:
            logger.warning(
                "reviewer_benchmark_evidence_no_claims",
                experiment_id=experiment_id,
            )
            return {
                **_DEFAULT_FALLBACK,
                "summary": "No verifiable scientific claims could be extracted from the manuscript.",
                "evidence_grounding": "weak",
                "recommendation": "major_revision",
            }

        per_claim_evidence = await asyncio.gather(
            *[self._evidence_for_claim(c, experiment_id) for c in claims],
            return_exceptions=False,
        )

        retrieved_literature: List[Dict[str, Any]] = []
        for entry in per_claim_evidence:
            retrieved_literature.extend(entry.get("sources", []))

        detect_result = await HallucinationDetector(
            experiment_id,
            llm_selection=self.llm_selection,
        ).detect(
            claims=claims,
            retrieved_literature=retrieved_literature,
        )
        verdicts = detect_result.get("verdicts", [])

        result = await self._synthesize(
            sections=sections,
            per_claim_evidence=per_claim_evidence,
            verdicts=verdicts,
            feedback=feedback,
            previous_reviews=previous_reviews,
            review_parameters=review_parameters,
            experiment_id=experiment_id,
        )

        end = time.perf_counter()
        logger.info(
            "reviewer_benchmark_evidence_completed",
            experiment_id=experiment_id,
            duration=end - start,
            num_claims=len(claims),
            num_verdicts=len(verdicts),
            recommendation=result.get("recommendation"),
        )

        return result

    @staticmethod
    def _coerce_sections(sections: Any) -> List[Dict[str, Any]]:
        """
        ClaimExtractor expects `synth_sections: List[Dict]`. The reviewer graph's
        `split_node` produces a dict (LLM-segmented). Normalize to a list.
        """
        if isinstance(sections, list):
            return sections
        if isinstance(sections, dict):
            inner = sections.get("sections")
            if isinstance(inner, list):
                return inner
            return [
                {"name": k, "content": v if isinstance(v, str) else json.dumps(v)}
                for k, v in sections.items()
            ]
        return [{"name": "manuscript", "content": str(sections)}]

    MAX_CLAIMS = 3

    async def _extract_claims(
        self, section_list: List[Dict[str, Any]], experiment_id: str
    ) -> List[Dict[str, Any]]:
        try:
            result = await ClaimExtractor(
                experiment_id,
                llm_selection=self.llm_selection,
            ).extract_claims(
                hypothesis=(
                    f"Evaluate the factual grounding of this manuscript by extracting "
                    f"the {self.MAX_CLAIMS} most important, atomically verifiable "
                    f"scientific claims. Focus on central findings, headline results, "
                    f"and key mechanistic assertions — skip trivial or background "
                    f"statements. Return at most {self.MAX_CLAIMS} claims."
                ),
                synth_sections=section_list,
            )
            claims = result.get("claims", []) or []
            return claims[: self.MAX_CLAIMS]
        except Exception as exc:
            logger.exception(
                "reviewer_benchmark_evidence_claim_extraction_failed",
                experiment_id=experiment_id,
                error=str(exc),
            )
            return []

    async def _evidence_for_claim(
        self, claim: Dict[str, Any], experiment_id: str
    ) -> Dict[str, Any]:
        question = claim.get("text") or ""
        claim_id = claim.get("claim_id")

        if not question:
            return {
                "claim_id": claim_id,
                "claim": question,
                "hybrid_answer": None,
                "sources": [],
            }

        hybrid_task = self._safe_hybrid(question, experiment_id)
        rag_task = asyncio.to_thread(
            self._safe_rag_retrieval, question, experiment_id
        )
        hybrid_out, rag_chunks = await asyncio.gather(hybrid_task, rag_task)

        sources: List[Dict[str, Any]] = []
        sources.extend(hybrid_out.get("sources") or [])
        sources.extend(rag_chunks or [])

        return {
            "claim_id": claim_id,
            "claim": question,
            "hybrid_answer": hybrid_out.get("answer"),
            "sources": sources,
        }

    async def _safe_hybrid(self, question: str, experiment_id: str) -> Dict[str, Any]:
        try:
            return await self.hybrid.run(
                question,
                experiment_id=experiment_id,
                llm_selection=self.llm_selection,
            )
        except Exception as exc:
            logger.warning(
                "reviewer_benchmark_evidence_hybrid_failed",
                experiment_id=experiment_id,
                error=str(exc),
            )
            return {"answer": None, "sources": []}

    def _safe_rag_retrieval(
        self, question: str, experiment_id: str
    ) -> List[Dict[str, Any]]:
        try:
            return self.rag.retrieve_with_metadata(
                question, experiment_id=experiment_id, top_k=3
            )
        except Exception as exc:
            logger.warning(
                "reviewer_benchmark_evidence_rag_failed",
                experiment_id=experiment_id,
                error=str(exc),
            )
            return []

    async def _synthesize(
        self,
        sections: Any,
        per_claim_evidence: List[Dict[str, Any]],
        verdicts: List[Dict[str, Any]],
        feedback: str | None,
        previous_reviews: Dict[str, Any] | None,
        review_parameters: Dict[str, Any] | None,
        experiment_id: str,
    ) -> Dict[str, Any]:
        chain = self.prompt | self.llm
        response = None
        try:
            response = await chain.ainvoke(
                {
                    "sections": json.dumps(sections, indent=2, default=str),
                    "per_claim_evidence": json.dumps(
                        per_claim_evidence, indent=2, default=str
                    ),
                    "verdicts": json.dumps(verdicts, indent=2, default=str),
                    "feedback": feedback,
                    "previous_reviews": (
                        json.dumps(previous_reviews, indent=2, default=str)
                        if previous_reviews
                        else None
                    ),
                    "review_parameter_guidance": build_review_parameter_guidance(
                        review_parameters
                    ),
                }
            )
            return json.loads(response.content)

        except Exception as exc:
            logger.exception(
                "reviewer_benchmark_evidence_synthesis_failed",
                experiment_id=experiment_id,
                error=str(exc),
            )
            content = getattr(response, "content", "") if response else ""
            try:
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    return json.loads(match.group())
            except Exception:
                pass

            return {
                **_DEFAULT_FALLBACK,
                "raw_output": content or "no_response",
            }
