import structlog
from fastapi import UploadFile
from src.pipelines.rag_pipeline import BosnaRagPipeline
import os
import tempfile
import hashlib
from src.document.loaders.mistral_ocr import MistralOCRPDFLoader
from utils.get_info_from_doi import get_paper_metadata
from utils.extract_metadata import extract_first_pages, extract_metadata, parse_llm_output
from src.processing.md_splitter import MarkdownDocumentPreprocessor

logger = structlog.get_logger()

async def process_upload_file(file: UploadFile, rag: BosnaRagPipeline):
    file_extension = os.path.splitext(file.filename)[1]
    file_name = file.filename

    logger.info("Reading uploaded file", filename=file.filename)

    content = await file.read()
    logger.info("File read into memory", size=len(content))

    file_hash = hashlib.sha256(content).hexdigest()
    
    logger.info("Generated file hash", file_hash=file_hash)

    if rag.checkpoint_db.hash_exists(file_hash):
        logger.info("File already exists")
        return
    with open(os.path.join(tempfile.gettempdir(), f"upload_temp{file_extension}"), "wb") as temp_file:
        temp_file.write(content)
        temp_path = temp_file.name

    logger.info("File written to temporary path", temp_path=temp_path)

    doc = MistralOCRPDFLoader().extract_text_from_pdf(temp_path)
    logger.info(f"This is document {doc}")
    logger.info(f"This is content of the document {doc.text}")
    doi = MarkdownDocumentPreprocessor().extract_doi(doc.text)
    logger.info(f"This is doi {doi}")
    doc.metadata["doi"] = doi if doi else None


    print(doc)
    
    extracted_doi = doc.metadata.get("doi")
    first_pages_text = doc.text[:12000]    
    logger.info(f"First pages are {first_pages_text}")
    llm_output = await extract_metadata(first_pages_text)
    metadata = parse_llm_output(llm_output) or {}
    logger.info(f"LLM extracted metadata is {metadata}")
    
    if not metadata:
        logger.error("LLM metadata extraction failed, using empty defaults")
        metadata = {
            "title": None, 
            "abstract": None, 
            "authors": None
        }

    if extracted_doi:
        logger.info("Doi found, data will be extracted from .bib")
        try:
            extracted_metadata = get_paper_metadata(extracted_doi) 
        except Exception as e: 
            logger.error("Failed to fetch metadata from DOI", error = str(e))
            extracted_metadata = {}

        doc.metadata["title"] =  extracted_metadata["title"]
        doc.metadata["authors"] = extracted_metadata["authors"]
        doc.metadata["abstract"] = extracted_metadata["abstract"] 
        if not doc.metadata["title"]:
            logger.info("Title from .bib is missing")
            doc.metadata["title"] = metadata.get("title")
        if not doc.metadata["authors"]:
            logger.info("Authors from .bib are missing")
            doc.metadata["authors"] = metadata.get("authors")
        if not doc.metadata["abstract"]:
            logger.info("Abstract from .bib is missing")
            doc.metadata["abstract"] = metadata.get("abstract")
            logger.info(f"Abstract filled with {doc.metadata['abstract']}")

    else: 
        logger.info("Doi not found in text, using LLM extraction")
        doc.metadata["title"] = metadata.get("title")
        doc.metadata["authors"] = metadata.get("authors")
        doc.metadata["abstract"] = metadata.get("abstract")

    rag.add_document(doc)
    rag.checkpoint_db.add_document(doc, file_hash, file_name)
    rag.checkpoint_db.mark_ocr_done(doc)
    rag.checkpoint_db.mark_embed_done(doc)
    logger.info(f"Document uploaded: {doc}")
    os.remove(temp_path)

    return file_hash