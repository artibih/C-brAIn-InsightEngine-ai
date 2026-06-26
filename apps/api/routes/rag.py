from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi import File, UploadFile, HTTPException
import tempfile
import os

from src.document.loaders.mistral_ocr import MistralOCRPDFLoader
from src.pipelines.rag_pipeline import BosnaRagPipeline
from apps.api.schemas.rag import RAGRetrieveRequest
from config.llm_catalog import ConversationMode
from config.llm_selection import resolve_llm_selection
import structlog
from src.vector_db.weaviate_sections import WeaviateSectionChunksDB
from src.vector_db.weaviate_images import WeaviateImageDB
from src.vector_db.weaviate_metadata import WeaviateMetadataDB
from src.vector_db.weaviate_summaries import WeaviateSummaryDB
from src.vector_db.weaviate_tables import WeaviateTableDB
from apps.api.schemas.rag import PromptAnalyzeRequest, PromptAnalyzeResponse
from apps.services.prompt_refinement_service import PromptRefinementService
from azure.storage.blob import BlobServiceClient
from azure.storage.queue import QueueClient
from config.settings import settings
import hashlib
import json
from src.checkpoint.checkpoint_db import CheckpointDB
router = APIRouter()

logger = structlog.get_logger()


_prompt_refinement_service_instance = None
_blob_service = None
_queue_client = None


def get_blob_service():
    global _blob_service
    if _blob_service is None:
        _blob_service = BlobServiceClient.from_connection_string(
            settings.AZURE_STORAGE_CONNECTION_STRING
        )
    return _blob_service


def get_queue_client():
    global _queue_client
    if _queue_client is None:
        _queue_client = QueueClient.from_connection_string(
            settings.AZURE_STORAGE_CONNECTION_STRING,
            "paper-processing-dev"
        )
    return _queue_client

def get_prompt_refinement_service() -> PromptRefinementService:
    global _prompt_refinement_service_instance
    if _prompt_refinement_service_instance is None:
        _prompt_refinement_service_instance = PromptRefinementService()
    return _prompt_refinement_service_instance


checkpoint_db = CheckpointDB()
@router.post("/upload/v5")
async def upload_file(file: UploadFile = File(...)):
    try:

        file_extension = os.path.splitext(file.filename)[1]
        logger.info("Reading uploaded file", filename=file.filename)

        content = file.file.read()
        file.file.close() 
        logger.info("File read into memory", size=len(content))

        with open(os.path.join(tempfile.gettempdir(), f"upload_temp{file_extension}"), "wb") as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        logger.info("File written to temporary path", temp_path=temp_path)
        doc = MistralOCRPDFLoader().extract_text_from_pdf(temp_path)
        logger.info("Text extracted from PDF", text_length=len(doc.text))
        rag = BosnaRagPipeline()
        logger.info("Rag pipeline initialized")
        rag.add_document(doc)
        logger.info("Document added to RAG pipeline")

        os.remove(temp_path)
        
        return {"message": "File uploaded and stored successfully"}
        
    except Exception as e:
        import traceback
        logger.info(traceback.format_exc()) 
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/batch")
async def upload_files(files: list[UploadFile] = File(...), collection_name: str = Form("default"), ):
    try:
        blob_service = get_blob_service()
        queue_client = get_queue_client()

        for file in files:

            
            file_obj = file.file
            sha = hashlib.sha256()
            file_obj.seek(0)
            while chunk:= file_obj.read(8192):
                sha.update(chunk)
            
            file_hash = sha.hexdigest()
            file_obj.seek(0)
            status = checkpoint_db.get_status(file_hash)

            if status in [
                "queued",
                "processing",
                "completed"
            ]:
                logger.info(
                    f"{file.filename} already handled with status={status}"
                )
                continue

            blob_name = f"{file_hash}_{file.filename}"
            blob_client = blob_service.get_blob_client(
                container="papers",
                blob=blob_name
            )

            blob_url = blob_client.url

            needs_upload = status in [None, "pending_upload"]

            if needs_upload:

                if status is None:
                    checkpoint_db.add_pending_document(
                        file_hash,
                        file.filename,
                        collection_name
                    )

                blob_client.upload_blob(
                    file_obj,
                    overwrite=True
                )

                checkpoint_db.set_blob_url(
                    file_hash,
                    blob_url
                )

                checkpoint_db.update_status(
                    file_hash,
                    "uploaded"
                )

            message = {
                "blob_name": blob_name,
                "file_hash": file_hash,
                "file_name": file.filename,
                "collection_name": collection_name
            }

            try:
                queue_client.send_message(
                    json.dumps(message)
                )

                checkpoint_db.update_status(
                    file_hash,
                    "queued"
                )

        
            except Exception as e:

                checkpoint_db.update_status(
                    file_hash,
                    "queue_failed",
                    str(e)
                )

                raise

        return {
            "message": f"{len(files)} files queued for processing", 
            "collection_name": collection_name,
         
        }

    except Exception as e:
        logger.exception("Failed to queue uploaded files")
        raise HTTPException(status_code=500, detail="Failed to queue uploaded files")
    

@router.get("/ask/rag/v5")
async def ask_rag_v5(query: str):
    llm_selection = resolve_llm_selection(ConversationMode.RAG.value, None)
    rag = BosnaRagPipeline(llm_selection=llm_selection)
    res = rag.generate_enhanced_response(query, experiment_id="ask_rag_v5")
    return {'response': res}

@router.get("/retrieve/v5")
async def retrieve(query: str):
    llm_selection = resolve_llm_selection(ConversationMode.RAG.value, None)
    rag = BosnaRagPipeline(llm_selection=llm_selection)
    response = rag.generate_enhanced_response(query, experiment_id="retrieve_v5")
    return response


@router.post("/retrieve/v5")
async def retrieve_with_attachments(request: RAGRetrieveRequest):
    try:
        llm_selection = resolve_llm_selection(
            ConversationMode.RAG.value,
            request.llm_selection,
        )
        rag = BosnaRagPipeline(llm_selection=llm_selection)
        response = rag.generate_enhanced_response(
            request.question,
            experiment_id="retrieve_v5",
            dataset_paths=request.dataset_paths or [],
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to retrieve RAG response with attachments")
        raise HTTPException(status_code=500, detail="Failed to retrieve RAG response with attachments")


@router.delete("/papers/{paper_id}")
async def delete_paper(paper_id: str):
    try:
        logger.info("Deleting paper everywhere", paper_id=paper_id)

        WeaviateSectionChunksDB().delete_by_paper_id(paper_id)
        WeaviateSummaryDB().delete_by_paper_id(paper_id) 
        WeaviateImageDB().delete_by_paper_id(paper_id)
        WeaviateTableDB().delete_by_paper_id(paper_id)
        WeaviateMetadataDB().delete_by_paper_id(paper_id)

        return {
            "status": "success",
            "paper_id": paper_id,
            "message": "Paper deleted from all Weaviate collections"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        logger.exception("Failed to delete paper")
        raise HTTPException(status_code=500, detail="Failed to delete paper")

    
@router.get("/papers")
async def list_papers():
    try:
        db = WeaviateSectionChunksDB()
        paper_ids = db.list_paper_ids()

        return {
            "count": len(paper_ids),
            "paper_ids": paper_ids
        }

    except Exception:
        logger.exception("Failed to list papers")
        raise HTTPException(status_code=500, detail="Failed to list papers")
    
@router.post("/analyze-prompt", response_model=PromptAnalyzeResponse)
async def analyze_prompt(request: PromptAnalyzeRequest, service: PromptRefinementService = Depends(get_prompt_refinement_service)):
    try:
        result = await service.analyze_draft(request.draft_query)
        return result
    except Exception:
        logger.exception("Error in analyze_prompt")
        raise HTTPException(status_code=500, detail=f"Error analyzing prompt")