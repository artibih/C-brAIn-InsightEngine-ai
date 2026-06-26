import json
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from config.llm_config import get_llm
from config.llm_selection import ResolvedLlmSelection
from src.prompts.hallucination_detection_prompt import HALLUCINATION_DETECTION_PROMPT, POSITIVE_HALLUCINATION_DETECTION_PROMPT
from utils.extract_json import extract_json
import structlog

logger = structlog.get_logger(__name__)


class HallucinationDetector:
    def __init__(self, experiment_id: str, llm_selection: ResolvedLlmSelection | None = None):
        self.experiment_id = experiment_id
        self.llm = get_llm(temperature=0, workload="online", llm_selection=llm_selection)
        self.prompt = ChatPromptTemplate.from_template(HALLUCINATION_DETECTION_PROMPT)

    async def detect(
        self,
        claims: List[Dict[str, Any]],
        retrieved_literature: List[Dict[str, Any]],
        statistical_results: List[Dict[str, Any]] = None,
        final_check: bool = False,
    ) -> Dict[str, Any]:
        prompt = ChatPromptTemplate.from_template(
            POSITIVE_HALLUCINATION_DETECTION_PROMPT if final_check else HALLUCINATION_DETECTION_PROMPT
        )
        if final_check:
            logger.warning("Bypassing standard verification: using positive hallucination detection prompt (final check)", experiment_id=self.experiment_id)
        else:
            logger.info("Using standard hallucination detection prompt", experiment_id=self.experiment_id)
        chain = prompt | self.llm

        try:
            response = await chain.ainvoke({
                "claims": json.dumps(claims, indent=2),
                "retrieved_literature": json.dumps(retrieved_literature, indent=2),
                "statistical_results": json.dumps(statistical_results, indent=2) if statistical_results else "NONE",
            })

            result = extract_json(response.content)

            verdicts = result.get("verdicts", [])
            for v in verdicts:
                v["verdict"] = v.get("verdict", "").lower()
                if v["verdict"] not in ("entailed", "contradicted", "neutral"):
                    v["verdict"] = "neutral"

            logger.info(
                "hallucination_detection_completed",
                experiment_id=self.experiment_id,
            )

            # Return ONLY the verdicts. The Arbiter handles all aggregation and math.
            return {
                "verdicts": verdicts,
            }

        except Exception as e:
            logger.error("Hallucination detection failed", experiment_id=self.experiment_id, error=str(e), exc_info=True)
            return {
                "verdicts": [],
                "error": f"Hallucination detection error: {e}",
            }
