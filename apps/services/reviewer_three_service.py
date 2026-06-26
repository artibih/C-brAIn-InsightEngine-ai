import uuid

from apps.api.schemas.requests import ReviewerTestRequest
import structlog
from typing import AsyncGenerator, Dict, Any

from apps.services.workflow_registry import review_workflow
from config.llm_selection import ResolvedLlmSelection
from utils.format_request_context import normalize_previous_reviews
from utils.review_parameters import normalize_review_parameters

logger = structlog.get_logger()


async def stream_reviewer_three(
    request: ReviewerTestRequest,
    llm_selection: ResolvedLlmSelection | None = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streams LangGraph reviewer workflow events for a PDF manuscript.
    """

    files = request.dataset_paths if request.dataset_paths else None
    review_parameters = normalize_review_parameters(request.review_parameters)

    state = {
        "experiment_id": (
            request.previous_reviews.get("experiment_id")
            if request.previous_reviews
            else None
        ) or str(uuid.uuid4()),
        "files": files,
        "feedback": request.feedback,
        "previous_reviews": normalize_previous_reviews(request.previous_reviews),
        "llm_selection": llm_selection,
        "review_parameters": review_parameters,
    }

    logger.info(
        "reviewer_three_started",
        experiment_id=state["experiment_id"],
        files=files,
        review_parameters=review_parameters,
    )

    try:

        async for event in review_workflow.astream(state):

            for node_name, node_output in event.items():
                if node_name != "split":

                    yield {
                        "data": node_output,
                    }


    except Exception as e:
        logger.exception("reviewer_three_failed")

        yield {
            "agent": "system",
            "event": "error",
            "data": {"error": str(e)},
        }