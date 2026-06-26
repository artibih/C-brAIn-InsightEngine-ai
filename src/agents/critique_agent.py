from typing import Dict, Any, Optional
import json
from langchain_core.prompts import ChatPromptTemplate
from config.llm_config import get_llm
from config.llm_selection import ResolvedLlmSelection
from src.prompts.critic_prompt import CRITIC_PROMPT, POSITIVE_CRITIC_PROMPT
from utils.extract_json import extract_json
import structlog

logger = structlog.get_logger(__name__)

class CriticAgent:
    def __init__(self, experiment_id: str, llm_selection: ResolvedLlmSelection | None = None):
        self.llm = get_llm(temperature=0, workload="online", llm_selection=llm_selection)
        self.experiment_id = experiment_id

        logger.info(
            "initializing_critic_agent",
            experiment_id=self.experiment_id
        )
        

    async def critique(self, hypothesis, synthesis, literature, has_uploaded_data, final_critique=False) -> Dict[str, Any]:
        prompt = ChatPromptTemplate.from_template(
        POSITIVE_CRITIC_PROMPT if final_critique else CRITIC_PROMPT
    )
        if final_critique:
            logger.info("Bypassing standard critique: using positive critic prompt (final critique)", experiment_id=self.experiment_id)
        else:
            logger.info("Using standard critic prompt", experiment_id=self.experiment_id)
        chain = prompt | self.llm

        synthesis_safe = {
            "extracted_claims": synthesis.get("extracted_claims", []),
            "contradictions": synthesis.get("contradictions", []),
            "failure_modes": synthesis.get("failure_modes", []),
            "confidence_score": synthesis.get("confidence_score", 0.0)
        }

        response = await chain.ainvoke({
            "hypothesis": hypothesis,
            "synthesis": json.dumps(synthesis_safe, indent=2),
            "literature": json.dumps(literature, indent=2),
            "has_uploaded_data": has_uploaded_data
        })

        try:
            feedback = extract_json(response.content)
            if isinstance(feedback.get("needs_revision"), str):
                feedback["needs_revision"] = feedback["needs_revision"].lower() == "true"

            if isinstance(feedback.get("needs_revision"), int):
                feedback["needs_revision"] = bool(feedback["needs_revision"])

            for key in ["revise_planner", "revise_retrieval", "revise_statistics", "revise_synthesis"]:
                if isinstance(feedback.get(key), str):
                    feedback[key] = feedback[key].lower() == "true"
                if isinstance(feedback.get(key), int):
                    feedback[key] = bool(feedback[key])

            feedback.setdefault("needs_revision", True)
            feedback.setdefault("revise_planner", False)
            feedback.setdefault("revise_retrieval", False)
            feedback.setdefault("revise_statistics", False)
            feedback.setdefault("revise_synthesis", True)
            feedback.setdefault("priority_agent", "synthesis")

            feedback.setdefault("epistemic_status", "inconclusive")

            feedback.setdefault("issues", [])
            feedback.setdefault("contradictions", [])
            feedback.setdefault("revision_instructions", [])
            feedback.setdefault("strengths", [])
            feedback.setdefault("validation_summary", "")
            feedback.setdefault("quality_score", 0.0)


            logger.info(
                "critic_completed",
                experiment_id=self.experiment_id,
                needs_revision=feedback.get("needs_revision"),
                quality_score=feedback.get("quality_score"),
                epistemic_status=feedback.get("epistemic_status"),
            )
            return feedback
        
        except Exception as e:
            logger.exception(
                "critic_json_parsing_failed",
                experiment_id=self.experiment_id,
                error=str(e),
            )
            return {
                "needs_revision": True,
                "issues": [f"Critic JSON parsing error: {e}"],
                "revision_instructions": ["Fix critic JSON formatting"],
                "quality_score": 0.0
            }
