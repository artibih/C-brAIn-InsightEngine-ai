from fastapi import APIRouter, HTTPException
from apps.api.schemas.requests import HypothesisTestRequest
import structlog
from fastapi.responses import StreamingResponse
import json

from utils.blob_storage import upload_hallucination_result, download_hallucination_result, clear_hallucination_result
from config.llm_catalog import ConversationMode
from config.llm_selection import resolve_llm_selection

from apps.services.hypothesis_runner import (
    stream_hypothesis_graph
)

logger = structlog.get_logger()
from utils.format_request_context import build_full_context

router = APIRouter()


@router.post("/test/stream")
async def stream_test(request: HypothesisTestRequest):
    llm_selection = resolve_llm_selection(ConversationMode.CHAT.value, request.llm_selection)

    async def event_generator():
        last_hallucination_event = None
        clear_hallucination_result()
        try:
            formatted_context = build_full_context(request.context)
            async for event in stream_hypothesis_graph(
                request.hypothesis,
                request.dataset_paths,
                request.feedback,
                formatted_context,
                llm_selection,
            ):
               
                if event.get("agent") == "hallucination_detector":
                    last_hallucination_event = event

                yield f"data: {json.dumps(event)}\n\n"

        finally:
            if last_hallucination_event:
                try:
                    upload_hallucination_result(last_hallucination_event)
                    logger.info("Uploaded final hallucination report to blob storage")
                except Exception as e:
                    logger.error("Failed to upload hallucination report", error=str(e))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@router.get("/hallucination-test")
async def get_hallucination_test():
    """Return the stored hallucination report from the last completed run."""
    try:
        result = download_hallucination_result()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"No hallucination report found: {e}",
        )
