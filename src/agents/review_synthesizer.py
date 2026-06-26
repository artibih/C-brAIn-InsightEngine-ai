import json
import re
import time
from typing import List, Dict, Any

import structlog
from langchain_core.prompts import ChatPromptTemplate
from config.llm_config import get_llm
from config.llm_selection import ResolvedLlmSelection
from src.prompts.reviewer_synthesizer_prompt import REVIEW_SYNTHESIZER_PROMPT
from utils.review_parameters import build_review_parameter_guidance

logger = structlog.get_logger()


class ReviewSynthesizer:
    def __init__(self, llm_selection: ResolvedLlmSelection | None = None):
        self.llm = get_llm(temperature=0, workload="online", llm_selection=llm_selection)
        self.prompt = ChatPromptTemplate.from_template(REVIEW_SYNTHESIZER_PROMPT)

    async def synthesize_reviews(
        self,
        reviews: List[Dict[str, Any]],
        experiment_id: str,
        feedback: str | None = None,
        review_parameters: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:

        start = time.perf_counter()

        logger.info(
            "synthesizer_started",
            experiment_id=experiment_id,
            num_reviews=len(reviews),
        )

        chain = self.prompt | self.llm

        cleaned_reviews = [
            r["output"] if isinstance(r, dict) and "output" in r else r
            for r in reviews if r
        ]
        response = None
        try:
            response = await chain.ainvoke({
                "reviews": json.dumps(cleaned_reviews, indent=2),
                "feedback": feedback or "No user feedback provided.",
                "review_parameter_guidance": build_review_parameter_guidance(review_parameters)
            })

            parsed = json.loads(response.content)

            end = time.perf_counter()
            logger.info(
                "synthesizer_completed",
                experiment_id=experiment_id,
                duration=end - start,
                recommendation=parsed.get("final_recommendation"),
            )

            return parsed

        except Exception as e:
            logger.exception(
                "synthesizer_failed",
                experiment_id=experiment_id,
                error=str(e),
            )

            try:
                match = re.search(r"\{.*\}", response.content, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())

                    logger.info(
                        "synthesizer_recovered_from_partial_json",
                        experiment_id=experiment_id,
                    )

                    return parsed
            except Exception:
                pass

            return {
                "consensus": "",
                "disagreements": "",
                "key_risks": [],
                "final_recommendation": "unknown",
                "justification": "",
                "raw_output": getattr(response, "content", "no_response"),
            }