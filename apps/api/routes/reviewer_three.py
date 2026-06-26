from fastapi import APIRouter, UploadFile, File, HTTPException
from apps.api.schemas.requests import ReviewerTestRequest
import structlog
from fastapi.responses import StreamingResponse, FileResponse
import json
import tempfile
import os
from pathlib import Path
import zipfile
from typing import List

from apps.services.reviewer_three_service import stream_reviewer_three
from config.llm_catalog import ConversationMode
from config.llm_selection import resolve_llm_selection
from utils.format_request_context import build_full_context
from src.agents.document_preprocessor import DocumentPreprocessor

logger = structlog.get_logger()
router = APIRouter()
UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


@router.post("/test/stream")
async def stream_test(request: ReviewerTestRequest):
    llm_selection = resolve_llm_selection(ConversationMode.REVIEWER.value, request.llm_selection)

    async def event_generator():

        async for event in stream_reviewer_three(
            request=request,
            llm_selection=llm_selection,
        ):
            yield f"{json.dumps(event, default=str)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

@router.post("/test/preprocessor/simple")
async def test_preprocessor_simple(files: List[UploadFile] = File(...)):
    """
    Simpler version that returns JSON with extracted sections.
    
    Args:
        files: List of uploaded files
        
    Returns:
        JSON with extracted sections for each file
    """
    temp_dir = None
    
    try:
      
        temp_dir = tempfile.mkdtemp(prefix='preprocessor_test_')
        logger.info("test_preprocessor_simple_started", file_count=len(files))
        
        temp_root = Path(temp_dir).resolve()
        results = []
        
        for uploaded_file in files:
            try:
                safe_name = Path(uploaded_file.filename or "").name
                if not safe_name:
                    raise ValueError("Invalid uploaded filename")

                file_path = (temp_root / safe_name).resolve()
                if temp_root not in file_path.parents:
                    raise ValueError("Invalid upload path")

                with open(file_path, 'wb') as f:
                    size = 0
                    while chunk := await uploaded_file.read(UPLOAD_CHUNK_SIZE):
                        if size + len(chunk) > MAX_UPLOAD_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail=f"File too large: {uploaded_file.filename}",
                            )
                        size += len(chunk)
                        f.write(chunk)

                logger.info(
                    "processing_file",
                    filename=uploaded_file.filename,
                    size=size
                )

                preprocessor = DocumentPreprocessor()
                result = await preprocessor.process_files([str(file_path)])
                
                formatted_sections = {}
                for section in result.get('sections', []):
                    section_title = section.get('title', 'UNTITLED').upper()
                    section_content = section.get('content', '')
                    formatted_sections[section_title] = section_content
                
                results.append({
                    'filename': uploaded_file.filename,
                    'success': True,
                    'sections': formatted_sections,
                    'sections_count': len(formatted_sections),
                    'error': result.get('error')
                })
                
                logger.info(
                    "file_processed",
                    filename=uploaded_file.filename,
                    sections_count=len(formatted_sections)
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(
                    "file_processing_failed",
                    filename=uploaded_file.filename,
                    error=str(e),
                    exc_info=True
                )
                
                results.append({
                    'filename': uploaded_file.filename,
                    'success': False,
                    'error': str(e)
                })
        
        return {
            'total_files': len(files),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'results': results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "test_preprocessor_simple_failed",
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail="Critical error in preprocessor test",
        ) from e
    
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning("temp_dir_cleanup_failed", temp_dir=temp_dir, error=str(e))
