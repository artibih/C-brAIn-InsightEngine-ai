import json
import re
import time
from typing import Dict, Any

import structlog
from langchain_core.prompts import ChatPromptTemplate
from config.llm_config import get_llm
from config.llm_selection import ResolvedLlmSelection
from src.prompts.reviewer_domain_expert_prompt import REVIEWER_DOMAIN_EXPERT_PROMPT
from utils.response_postprocessing import response_text
from utils.review_parameters import build_review_parameter_guidance

logger = structlog.get_logger()


class ReviewerDomainExpert:
    def __init__(self, llm_selection: ResolvedLlmSelection | None = None):
        self.llm = get_llm(temperature=0, workload="online", llm_selection=llm_selection)
        self.prompt = ChatPromptTemplate.from_template(REVIEWER_DOMAIN_EXPERT_PROMPT)

    async def review(
        self,
        sections: Dict[str, Any],
        experiment_id: str,
        feedback: str = None,
        previous_reviews: Dict[str, Any] = None,
        review_parameters: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        start = time.perf_counter()

        logger.info(
            "reviewer_domain_expert_llm_started",
            experiment_id=experiment_id,
            num_sections=len(sections) if isinstance(sections, dict) else None,
            feedback=feedback
        )

        chain = self.prompt | self.llm
        response = None
        try:
            response = await chain.ainvoke({
                "sections": json.dumps(sections, indent=2),
                "feedback": feedback,
                "previous_reviews": json.dumps(previous_reviews, indent=2) if previous_reviews else None,
                "review_parameter_guidance": build_review_parameter_guidance(review_parameters),
            })

            parsed = json.loads(response.content)
            content = response_text(response)

            end = time.perf_counter()
            logger.info(
                "reviewer_domain_expert_llm_completed",
                experiment_id=experiment_id,
                duration=end - start,
                recommendation=parsed.get("recommendation"),
            )

            return parsed

        except Exception as e:
            logger.exception(
                "reviewer_domain_expert_llm_failed",
                experiment_id=experiment_id,
                error=str(e),
            )

            try:
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())

                    logger.info(
                        "reviewer_domain_expert_llm_recovered_partial_json",
                        experiment_id=experiment_id,
                    )

                    return parsed
            except Exception:
                pass

            return {
                "summary": "",
                "strengths": [],
                "weaknesses": [],
                "technical_concerns": [],
                "recommendation": "unknown",
                "raw_output": getattr(response, "content", "no_response"),
            }