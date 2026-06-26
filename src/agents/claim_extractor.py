import json
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from config.llm_config import get_llm
from config.llm_selection import ResolvedLlmSelection
from src.prompts.claim_extraction_prompt import CLAIM_EXTRACTION_PROMPT
from utils.extract_json import extract_json
import structlog
logger = structlog.get_logger(__name__)

class ClaimExtractor:
    def __init__(self, experiment_id: str, llm_selection: ResolvedLlmSelection | None = None):
        self.llm = get_llm(temperature=0, workload="online", llm_selection=llm_selection)
        self.prompt = ChatPromptTemplate.from_template(CLAIM_EXTRACTION_PROMPT)
        self.experiment_id = experiment_id

    async def extract_claims(
        self,
        hypothesis: str,
        synth_sections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        chain = self.prompt | self.llm

        try:
            response = await chain.ainvoke({
                "hypothesis": hypothesis,
                "sections": json.dumps(synth_sections, indent=2),
            })

            result = extract_json(response.content)
            claims = result.get("claims", [])
            for i, claim in enumerate(claims):
                claim.setdefault("claim_id", f"c{i + 1}")
                claim.setdefault("text", "")
            logger.info(
                "claim_extraction_successful",
                experiment_id=self.experiment_id,
                claim_count=len(claims),
            )
            return {"claims": claims}

        except Exception as e:
            logger.exception(
                "claim_extraction_failed",
                experiment_id=self.experiment_id,
                error=str(e)
            )

            return {
                "claims": [],
                "error": f"Claim extraction error: {e}",
            }

