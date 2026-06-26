# gives final synthesis of findings and confidence score
import json
import re
from typing import Dict, Any, List
from utils.extract_json import extract_json
from src.prompts.synthesizer_instructions import SYNTHESIS_PROMPT
from config.llm_config import get_llm
from config.llm_selection import ResolvedLlmSelection
from langchain_core.prompts import ChatPromptTemplate
import structlog
logger = structlog.get_logger(__name__)

class Synthesizer:
    def __init__(self, experiment_id: str, llm_selection: ResolvedLlmSelection | None = None):
        self.experiment_id = experiment_id
        self.llm = get_llm(
            temperature=0.2,
            workload="online",
            llm_selection=llm_selection,
        )
        self.prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)

    async def synthesize(self, hypothesis: str, test_plan: Dict[str, Any], results: Dict[str, Any], methodology: List[str], literature: List[Dict[str, Any]], critic_feedback: Dict[str, Any] = None, extracted_claims: List[Dict[str, Any]] = None, hallucination_feedback: Dict[str, Any] = None) -> Dict[str, Any]:
        chain = self.prompt | self.llm
        logger.info(
            "synthesis_started",
            experiment_id=self.experiment_id,
        )
        response = await chain.ainvoke({
            "hypothesis": hypothesis,
            "test_plan": json.dumps(test_plan, indent=2),
            "methodology": json.dumps(methodology, indent=2),
            "results": json.dumps(results, indent=2),
            "literature": json.dumps(literature, indent=2),
            "critic_feedback": json.dumps(critic_feedback or {}, indent=2),
            "extracted_claims": json.dumps(extracted_claims or [], indent=2),
            "hallucination_feedback": json.dumps(hallucination_feedback or {}, indent=2),
        })
        data = extract_json(response.content)
        logger.info(
            "synthesis_response_received",
            experiment_id=self.experiment_id,
        )
        return self._validate_output(data, hallucination_feedback)

    def _validate_output(self, data: Dict[str, Any], hallucination_feedback: Dict[str, Any] = None) -> Dict[str, Any]:
        logger.info(
            "validating_synthesis_output",
            experiment_id=self.experiment_id,
        )
        sections = data.get("sections", {}) or {}
        def normalize(section):
            if isinstance(section, str):
                return {
                    "paper_id": None,
                    "detail": section
                }

            if isinstance(section, dict):
                return {
                    "paper_id": section.get("paper_id"),
                    "detail": str(section.get("detail", ""))
                }

            return {"paper_id": None, "detail": ""}

        data["sections"] = {
            "background_context": normalize(sections.get("background_context")),
            "conceptual_framework": normalize(sections.get("conceptual_framework")),
            "methodology_evaluation": normalize(sections.get("methodology_evaluation")),
            "literature_synthesis": normalize(sections.get("literature_synthesis")),
            "statistical_interpretation": normalize(sections.get("statistical_interpretation")),
            "mechanistic_explanation": normalize(sections.get("mechanistic_explanation")),
            "evidence_integration": normalize(sections.get("evidence_integration")),
            "contradictions": normalize(sections.get("contradictions")),
            "limitations": normalize(sections.get("limitations")),
            "hypothesis_implications": normalize(sections.get("hypothesis_implications")),
            "broader_implications": normalize(sections.get("broader_implications")),
            "conclusion": normalize(sections.get("conclusion")),
        }
      
        data["contradictions"] = list(data.get("contradictions", []))
        data["failure_modes"] = list(data.get("failure_modes", []))

        confidence_score = 1.0 - hallucination_feedback.get("summary", {}).get("hallucination_risk_score", 0.0) if hallucination_feedback else 0.0

        data["confidence_score"] = confidence_score

        logger.info(
            "synthesis_output_validated",
            experiment_id=self.experiment_id,
            confidence_score=data.get("confidence_score"),
        )
        
        return data
