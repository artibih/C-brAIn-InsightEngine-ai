import json
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from config.llm_config import get_llm
from config.llm_selection import ResolvedLlmSelection
from config.settings import settings
from src.prompts.semantic_arbiter_prompt import SEMANTIC_ARBITER_PROMPT
from utils.extract_json import extract_json
import structlog

logger = structlog.get_logger(__name__)

class SemanticArbiter:
    def __init__(self, experiment_id: str, llm_selection: ResolvedLlmSelection | None = None):
        self.llm = get_llm(temperature=0, workload="online", llm_selection=llm_selection)
        self.prompt = ChatPromptTemplate.from_template(SEMANTIC_ARBITER_PROMPT)
        self.experiment_id = experiment_id
        self.n_samples = settings.number_of_claim_extractors

    async def arbitrate(self, extracted_samples: List[dict]) -> dict:
        actual_samples_count = len(extracted_samples)
        if actual_samples_count != self.n_samples:
            logger.warning(
                "arbiter_sample_mismatch",
                expected=self.n_samples,
                received=actual_samples_count
            )

        threshold = (actual_samples_count // 2) + 1

        samples_block = ""
        for idx, sample in enumerate(extracted_samples, start=1):
            samples_block += f"Sample Set {idx}:\n{json.dumps(sample, indent=2)}\n\n"

        chain = self.prompt | self.llm

        try:
            response = await chain.ainvoke({
                "n_samples": actual_samples_count,
                "threshold": threshold,
                "samples_json_block": samples_block.strip()
            })

            result = extract_json(response.content)
            validated_claims = result.get("validated_claims", [])
            
            logger.info(
                "semantic_arbitration_successful",
                experiment_id=self.experiment_id,
                validated_count=len(validated_claims),
                rejected_outliers=result.get("rejected_outliers_count", 0),
                threshold_applied=threshold
            )
            
            return {"extracted_claims": validated_claims}
            
        except Exception as e:
            logger.exception("semantic_arbitration_failed", error=str(e))
            return {"extracted_claims": []}