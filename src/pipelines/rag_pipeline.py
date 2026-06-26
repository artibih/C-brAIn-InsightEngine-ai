from typing import List, Dict, Any
import os
import sys

from plotly import data

sys.path.append(os.path.abspath("../.."))

from src.document.base import Document
from src.document.loaders.mistral_ocr import MistralOCRPDFLoader
from src.processing.md_splitter import MarkdownDocumentPreprocessor
from src.embedding.mpnet_embedder import MpnetEmbedder
from src.vector_db.weaviate_sections import WeaviateSectionChunksDB

from src.vector_db.weaviate_images import WeaviateImageDB
from src.vector_db.weaviate_tables import WeaviateTableDB
from src.vector_db.weaviate_summaries import WeaviateSummaryDB
from src.vector_db.weaviate_metadata import WeaviateMetadataDB

from src.generation.contextual_generator import ContextualGenerator

from src.document.base import Document
from src.document.loaders.mistral_ocr import MistralOCRPDFLoader
from src.pipelines.base import RagPipeline
from src.storage.document_storage import DocumentStorage
from src.checkpoint.checkpoint_db import CheckpointDB
import structlog
import json
from azure.storage.blob import BlobClient
from utils.citation_utils import group_chunks_by_paper
from utils.response_postprocessing import postprocess_answer
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, generate_container_sas, ContainerSasPermissions
from utils.load_pdf_from_blob import read_pdf_safely
from config.llm_selection import ResolvedLlmSelection
from datetime import datetime, timedelta
# Configure logging
logger = structlog.get_logger()

MAX_UPLOADED_PDF_CHARS = 50000
MAX_TOTAL_UPLOADED_PDF_CHARS = 100000

class BosnaRagPipeline(RagPipeline):

    preprocessor: MarkdownDocumentPreprocessor
    loader: MistralOCRPDFLoader

    section_db: WeaviateSectionChunksDB
    table_db: WeaviateTableDB
    image_db: WeaviateImageDB
    summary_db: WeaviateSummaryDB
    metadata_db: WeaviateMetadataDB

    llm: ContextualGenerator

    storage: DocumentStorage
    checkpoint_db: CheckpointDB

    def __init__(
        self,
        llm_selection: ResolvedLlmSelection | None = None,
        collection_name: str | None = None,
    ):
        super().__init__()
        self.llm_selection = llm_selection
        self.preprocessor = MarkdownDocumentPreprocessor()
        self.loader = MistralOCRPDFLoader()
        self.embedder = MpnetEmbedder()
        self.summary_db = WeaviateSummaryDB(collection_name=collection_name)
        self.section_db = WeaviateSectionChunksDB(collection_name=collection_name)
        self.table_db = WeaviateTableDB(collection_name=collection_name)
        self.image_db = WeaviateImageDB(collection_name=collection_name)
        
        self.llm = ContextualGenerator(llm_selection=llm_selection)
        self.metadata_db = WeaviateMetadataDB(collection_name=collection_name)
        self.storage = DocumentStorage()
        self.checkpoint_db = CheckpointDB()
        
        credential = DefaultAzureCredential()

        blob_service = BlobServiceClient.from_connection_string(os.getenv("AZURE_BLOB_CONNECTION_STRING"))

        container = blob_service.get_container_client(
            os.getenv("AZURE_PAPER_CONTAINER")
        )

        blob = container.get_blob_client(
            os.getenv("AZURE_PAPER_JSON")
        )

        logger.info(
            "Loading paper map",
            container=os.getenv("AZURE_PAPER_CONTAINER"),
            file=os.getenv("AZURE_PAPER_JSON")
        )

        data = blob.download_blob().readall()

        self.paper_map = json.loads(data)

    def generate_container_sas_token(self, container_name: str) -> str:
        account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
        account_key = os.getenv("AZURE_STORAGE_KEY")

        sas = generate_container_sas(
            account_name=account_name,
            container_name=container_name,
            account_key=account_key,
            permission=ContainerSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1)
        )

        return sas

    def add_documents(self, documents: List[Document]) -> None:
        documents = self.preprocessor.preprocess_batch(documents)

        self.summary_db.add(documents)
        self.section_db.add(documents)
        self.table_db.add(documents)
        self.image_db.add(documents)
        self.metadata_db.add(documents)



    def add_documents_from_folder(self, folder_path: str) -> None:
        docs = []

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if not (os.path.isfile(file_path) and filename.endswith('.pdf')):
                continue

            doc = self.loader.extract_text_from_pdf(file_path)

            self.storage.store_document(doc)

            self.add_document(doc)

    def add_document(self, document: Document) -> None:
        self.add_documents([document])
    def retrieve(self, query: str, experiment_id: str, top_k: int = 3) -> List[Any]:
        embed = self.embedder.embed_query(query)
        sum_res = self.summary_db.search_embedding(query_embedding=embed, k=top_k)
        if not sum_res:
            return []
        paper_ids = {res.get("paper_id") for res in sum_res if res.get("paper_id")}
        
        filter = {
            "operator": "Or",
            "operands": [
                {"path": ["paper_id"], "operator": "Equal", "valueText": pid}
                for pid in paper_ids
            ]
        } if len(paper_ids) > 1 else {
            "path": ["paper_id"], "operator": "Equal", "valueText": list(paper_ids)[0]
        } if paper_ids else None

        sec_res = self.section_db.search_embedding(query_embedding=embed, k=5, filter=filter)
        tab_res = self.table_db.search_embedding(query_embedding=embed, k=1, filter=filter)
        img_res = self.image_db.search_embedding(query_embedding=embed, k=1, filter=filter)

        return sum_res + sec_res + tab_res + img_res
        
    def retrieveBM25(self, query: str, experiment_id: str, top_k: int = 3, similarity: str = "bm25") -> List[Any]:
        sum_res = self.summary_db.BM25search(query, k=top_k, similarity=similarity)
        sec_res = self.section_db.BM25search(query, k=5, similarity=similarity)
        tab_res = self.table_db.BM25search(query, k=5, similarity=similarity)
        img_res = self.image_db.BM25search(query, k=5, similarity=similarity)
        logger.info(f"BM25 retrieved: summaries={len(sum_res)}, sections={len(sec_res)}, tables={len(tab_res)}, images={len(img_res)}", experiment_id=experiment_id)
        
        return sec_res + tab_res + img_res + sum_res
        
    def generate_response(self, query: str) -> str:
        return self.llm.generate(query, context=[])
    
  

    def generate_enhanced_response(
        self,
        query: str,
        experiment_id: str,
        dataset_paths: list[str] | None = None
    ) -> Dict[str, Any]:

        chunks = self.retrieve_with_metadata(query, experiment_id=experiment_id)
        uploaded_chunks = self.load_uploaded_pdf_chunks(dataset_paths or [])
        chunks = uploaded_chunks + chunks

        chunks = group_chunks_by_paper(chunks)
        llm_res = self.llm.generate(query, context=chunks)
        sources = [
                {
                    "citation": i + 1,
                    **chunk
                }
                for i, chunk in enumerate(chunks)
            ]
        llm_res, sources = postprocess_answer(llm_res, sources)
        return {
            "answer": llm_res,
            "references": sources
        }

    def load_uploaded_pdf_chunks(self, dataset_paths: list[str]) -> list[dict]:
        chunks = []
        total_chars = 0

        for index, path in enumerate(dataset_paths):
            remaining_chars = MAX_TOTAL_UPLOADED_PDF_CHARS - total_chars
            if remaining_chars <= 0:
                break

            if not isinstance(path, str) or not path.strip():
                continue

            normalized_path = path.split("?", 1)[0].lower()
            if not normalized_path.endswith(".pdf"):
                continue

            try:
                text = read_pdf_safely(path, as_text=True)
            except Exception as e:
                raise ValueError(
                    "Uploaded PDF contains no extractable text or is invalid/corrupt"
                ) from e
            text = (text or "").strip()
            if not text:
                raise ValueError("Uploaded PDF contains no extractable text.")

            filename = os.path.basename(path.split("?", 1)[0])
            content = text[:min(MAX_UPLOADED_PDF_CHARS, remaining_chars)]
            total_chars += len(content)
            chunks.append({
                "content": content,
                "paper_id": f"uploaded_pdf_{index + 1}",
                "title": filename or "Uploaded PDF",
                "authors": None,
                "abstract": None,
                "doi": None,
                "doi_url": None,
                "paper_url": path,
            })

        return chunks
    
    def retrieve_with_metadata(self, query: str, experiment_id: str, top_k: int = 3) -> List[Dict[str, Any]]:

        context = self.retrieve(query, experiment_id=experiment_id, top_k=top_k)
        sas_token = self.generate_container_sas_token(os.getenv("AZURE_PAPER_CONTAINER"))
        extracted = []
        paper_ids = set()

        for chunk in context:
            pid = chunk.get("paper_id")
            if pid:
                paper_ids.add(pid)

        metadata_map = self.get_metadata_map(list(paper_ids))
        blob_map = self.get_blob_urls(list(paper_ids))
        for chunk in context:
            content = chunk.get("content")
            paper_id = chunk.get("paper_id")
            
            if isinstance(content, str) and content.strip():
                blob_url = blob_map.get(paper_id)
                meta = metadata_map.get(paper_id, {})
                azure_url = blob_url if blob_url else self.paper_map.get(paper_id)
                doi = meta.get("doi")
                extracted.append({
                    "content": content.strip(),
                    "paper_id": paper_id,
                    "title": meta.get("title"),
                    "authors": meta.get("authors"),
                    "abstract": meta.get("abstract"),
                    "doi": doi,
                    "doi_url": f"https://doi.org/{doi}" if doi else None, 
                    "paper_url": f"{azure_url}?{sas_token}"
                })

        return extracted
    
    def get_metadata_map(self, paper_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        results = self.metadata_db.fetch_by_paper_ids(paper_ids)

        metadata = {}

        for res in results:
            props = res.properties
            pid = props.get("paper_id")

            metadata[pid] = {
                "title": props.get("title"),
                "authors": props.get("authors"),
                "doi": props.get("doi"),
                "abstract": props.get("abstract")
            }

        return metadata
    
    def get_blob_urls(self, paper_ids: List[str]) -> Dict[str, str]:
        if not paper_ids:
            return {}
        
        conn = self.checkpoint_db._get_conn()
        
        try:
            cursor = conn.cursor()
        
            query = f'''
                SELECT id, blob_url FROM documents
                WHERE id IN ({','.join(['?'] * len(paper_ids))})
            '''

            cursor.execute(query, paper_ids)

            rows = cursor.fetchall()

            return {row[0]: row[1] for row in rows}
        
        finally:
            conn.close()