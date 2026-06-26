from typing import Dict, List, Optional
import structlog
from langchain_core.prompts import ChatPromptTemplate
from config.llm_config import get_llm
from config.llm_selection import ResolvedLlmSelection
from src.prompts.hypothesis_planning import PLANNING_PROMPT
from utils.extract_json import extract_json
logger = structlog.get_logger(__name__)

class HypothesisPlanner:
    """Agent responsible for creating hypothesis test plans (control plane only)."""

    def __init__(self, llm_selection: ResolvedLlmSelection | None = None):
        self.llm = get_llm(temperature=0, workload="online", llm_selection=llm_selection)
        self.prompt = ChatPromptTemplate.from_template(PLANNING_PROMPT)
        
    def create_plan(self, hypothesis: str, dataset_schemas: List[Dict], experiment_id: str, critic_feedback: dict = None, scientist_feedback: str = None, previous_context: dict = None) -> dict:
        """
        Create a structured test plan for hypothesis evaluation.
        """
        logger.info(
            "hypothesis_planning_started",
            experiment_id=experiment_id,
            hypothesis=hypothesis,
            dataset_count=len(dataset_schemas or []),
            has_critic_feedback=bool(critic_feedback),
            has_scientist_feedback=bool(scientist_feedback),
            has_previous_context=bool(previous_context),
        )
        chain = self.prompt | self.llm
        response = chain.invoke({
            "hypothesis": hypothesis,
            "dataset_schemas": dataset_schemas,
            "critic_feedback": critic_feedback or {},
            "scientist_feedback": scientist_feedback,
            "previous_context": previous_context,
            "has_datasets": bool(dataset_schemas)
        })

        try:
            logger.info(
                "Extracting plan from LLM response",
                experiment_id=experiment_id,
            )
            plan = extract_json(response.content)
            logger.info(
                "Plan extraction successful",
                experiment_id=experiment_id,
            )
        except Exception as e:
            logger.exception(
                "plan_extraction_failed",
                experiment_id=experiment_id,
            )
            raise ValueError(f"Plan extraction failed: {e}")

        self._validate_plan(plan, experiment_id)
        logger.info(
            "hypothesis_plan_created",
            experiment_id=experiment_id,
            objective=plan.get("objective"),
            num_analysis_steps=len(plan.get("analysis_steps", [])),
        )

        # plan = PLAN  # Use mock plan for now
        return plan

    def _validate_plan(self, plan: dict, experiment_id: str):
        """Lightweight schema validation."""

        required_keys = {
            "objective",
            "execution_flags",
            "methodology_checks",
            "analysis_steps",
            "validation_criteria",
        }

        missing = required_keys - plan.keys()
        if missing:
            logger.error(
                "hypothesis_plan_validation_failed",
                missing_keys=missing,
                experiment_id=experiment_id,
            )
            raise ValueError(f"Plan missing required keys: {missing}")

        flags = plan["execution_flags"]
        for key in ["requires_statistics", "allow_python_execution", "data_source"]:
            if key not in flags:
                logger.error(
                    "hypothesis_plan_validation_failed",
                    missing_execution_flag=key,
                    experiment_id=experiment_id,
                )
                raise ValueError(f"execution_flags missing required key: {key}")

        for step in plan["analysis_steps"]:
            required = {"step_id", "agent", "task"}
            missing = required - step.keys()
            if missing:
                logger.error(
                    "hypothesis_plan_validation_failed",
                    step_id=step.get("step_id"),
                    missing_keys=missing,
                    experiment_id=experiment_id,
                )
                raise ValueError(f"analysis_step missing required keys: {missing}")

