import structlog
import asyncio

from src.pipelines.rag_pipeline import BosnaRagPipeline
from src.document.loaders.mistral_ocr import MistralOCRPDFLoader
from utils.get_info_from_doi import get_paper_metadata
from utils.extract_metadata import extract_metadata, parse_llm_output
from src.processing.md_splitter import MarkdownDocumentPreprocessor

logger = structlog.get_logger()


async def process_local_file(temp_path: str, file_hash: str, file_name: str, collection_name: str | None = None):
    logger.info("Starting processing", file_name=file_name, file_hash=file_hash, collection_name=collection_name)

    rag = BosnaRagPipeline(collection_name=collection_name)


    try:

        logger.info("OCR start", file_hash=file_hash)

        doc = await asyncio.to_thread(
            MistralOCRPDFLoader().extract_text_from_pdf,
            temp_path
        )
        if not doc or not doc.text:
            raise ValueError("OCR returned empty document")
        
        rag.checkpoint_db.mark_ocr_done(file_hash)
    except Exception as e:
        logger.error("OCR failed", error=str(e), file_hash=file_hash, exc_info=True)
        raise

    try:

        rag.checkpoint_db.set_doc_id(file_hash, doc.id)

        doi = MarkdownDocumentPreprocessor().extract_doi(doc.text)
        doc.metadata["doi"] = doi

        logger.info("DOI extracted from text", doi=doi)

    except Exception as e:
        logger.warning("DOI extraction from text failed", error=str(e), file_hash=file_hash, exc_info=True)

    try:

        first_pages_text = doc.text[:4000]

        logger.info("PARALLEL LLM + DOI FETCH START")

        llm_task = extract_metadata(first_pages_text)

        doi_task = (
            asyncio.to_thread(get_paper_metadata, doi)
            if doi else asyncio.sleep(0, result={})
        )

        llm_output, doi_metadata = await asyncio.gather(
            llm_task,
            doi_task,
            return_exceptions=True
        )
        
        if isinstance(llm_output, Exception):
            logger.warning("LLM metadata extraction failed", error=str(llm_output))
            llm_output= None

        if isinstance(doi_metadata, Exception):
            logger.warning("DOI metadata exctraction failed", error=str(doi_metadata))
            doi_metadata = {}

    except Exception as e:
        logger.error("LLM extraction of metadata failed", error=str(e), file_hash=file_hash, exc_info=True)
        llm_output, doi_metadata = None, {}

    try:
        metadata = parse_llm_output(llm_output) or {}

        metadata = {
            "title": metadata.get("title"),
            "authors": metadata.get("authors"),
            "abstract": metadata.get("abstract"),
        }

        doc.metadata["title"] = doi_metadata.get("title") or metadata["title"]
        doc.metadata["authors"] = doi_metadata.get("authors") or metadata["authors"]
        doc.metadata["abstract"] = doi_metadata.get("abstract") or metadata["abstract"]
    except Exception as e:
        logger.warning("Metadata parsing failed", error=str(e), file_hash=file_hash,exc_info=True)

    try:
        logger.info("RAG INSERT START", file_hash=file_hash)

        await asyncio.to_thread(
            rag.add_document,
            doc
        )

    except Exception as e:
        logger.error("Vector DB insert failed", error=str(e), file_hash=file_hash, exc_info=True)
        raise
    
    logger.info("Processing completed", file_name=file_name)

    return file_hash