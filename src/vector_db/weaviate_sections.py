import logging
from typing import List
from src.document.base import Document
from config.global_config import CONFIG
from config.settings import settings
import weaviate
from dotenv import load_dotenv
from src.vector_db.WeaviateDB import WeaviateDB
import structlog

load_dotenv()
logger = structlog.get_logger()


class WeaviateSectionChunksDB(WeaviateDB):
    chunk_size = 1000
    chunk_overlap = 200

    def __init__(self, collection_name: str | None = None, embedder=None):
        super().__init__()
        logging.info("Initializing WeaviateSectionChunksDB...")
        default_collection = CONFIG.get('weaviate_sections', {}).get('collection_name', "SectionChunks")
        clean_name = self.sanitize_collection_name(collection_name)
        self.collection_name = f"SectionChunks_{clean_name}" if clean_name else default_collection
        logger.info(f"Using collection name: {self.collection_name}")
        self._setup_collection()

    def _setup_collection(self):
        try:
            schema = self.weaviate_client.schema.get()
            existing = [c['class'] for c in schema.get('classes', [])]
            if self.collection_name not in existing:
                self.weaviate_client.schema.create_class({
                    "class": self.collection_name,
                    "vectorizer": "none",
                    "properties": [
                        {"name": "content", "dataType": ["text"]},
                        {"name": "section_index", "dataType": ["int"]},
                        {"name": "total_sections", "dataType": ["int"]},
                        {"name": "chunk_index", "dataType": ["int"]},
                        {"name": "total_chunks", "dataType": ["int"]},
                        {"name": "paper_id", "dataType": ["text"]},
                    ]
                })
                logging.info("Created collection: %s", self.collection_name)
            else:
                logging.info("Collection exists: %s", self.collection_name)
        except Exception as e:
            logging.error("Error setting up collection: %s", e)
            raise

    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if len(chunk) >= self.chunk_size // 2:
                chunks.append(chunk)
        return chunks

    def _add_document(self, doc: Document) -> None:
        paper_id = doc.metadata.get("paper_id")

        if paper_id is None:
            raise AssertionError("Paper requires 'paper_id'")
        logger.info("Started adding chunks")
        
        with self.weaviate_client.batch as batch:
            batch.batch_size = 100 

            for i, section in enumerate(doc.content['sections']):
                sec_chunks = self._chunk_text(section)
                vectors = self.model.encode(sec_chunks, batch_size=32, show_progress_bar=False)

                for j, (chunk, vector) in enumerate(zip(sec_chunks, vectors)):
                    batch.add_data_object(
                        data_object={
                            "content": chunk,
                            "section_index": i,
                            "total_sections": len(doc.content['sections']),
                            "chunk_index": j,
                            "total_chunks": len(sec_chunks),
                            "paper_id": paper_id,
                        },
                        class_name=self.collection_name,
                        vector=vector.tolist()
                    )

        logger.info(f"Inserted document with paper_id: {paper_id}")
        

    def add(self, documents: List[Document]) -> None:
        for doc in documents:
            self._add_document(doc)

    def search(self, query: str, k: int = 5, filter=None) -> List:
        return self.search_embedding(query=query, k=k, filter=filter)

    def search_embedding(self, query_embedding=None, query: str = None, k: int = 5, filter=None) -> List:
        if query_embedding is None:
            raise ValueError("query_embedding is required for near_vector search")
        q = (
            self.weaviate_client.query
            .get(self.collection_name, ["content", "paper_id", "section_index", "chunk_index"])
            .with_near_vector({"vector": query_embedding})
            .with_limit(k)
            .with_additional(["vector"])
        )
        if filter:
            q = q.with_where(filter)
        result = q.do()
        return result.get("data", {}).get("Get", {}).get(self.collection_name, [])

    def BM25search(self, query: str, k: int = 5, similarity: str = None) -> List:
        if similarity == "bm25":
            result = (
                self.weaviate_client.query
                .get(self.collection_name, ["content", "paper_id"])
                .with_bm25(query=query)
                .with_limit(k)
                .do()
            )
        else:
            result = (
                self.weaviate_client.query
                .get(self.collection_name, ["content", "paper_id"])
                .with_near_text({"concepts": [query]})
                .with_limit(k)
                .do()
            )
        return result.get("data", {}).get("Get", {}).get(self.collection_name, [])

    def search_bm25(self, query: str, k: int = 5) -> List:
        result = (
            self.weaviate_client.query
            .get(self.collection_name, ["content", "paper_id"])
            .with_bm25(query=query)
            .with_limit(k)
            .do()
        )
        return result.get("data", {}).get("Get", {}).get(self.collection_name, [])

    def delete(self, where_filter=None) -> None:
        if where_filter is None:
            raise ValueError("Refusing to delete without a filter")
        self.weaviate_client.batch.delete_objects(
            class_name=self.collection_name,
            where=where_filter,
        )

    def delete_by_paper_id(self, paper_id: str) -> None:
        if not paper_id:
            raise ValueError("Invalid paper_id")
        self.delete({
            "path": ["paper_id"],
            "operator": "Equal",
            "valueText": paper_id,
        })

    def list_paper_ids(self) -> List[str]:
        result = (
            self.weaviate_client.query
            .get(self.collection_name, ["paper_id"])
            .with_limit(10000)
            .do()
        )
        objects = result.get("data", {}).get("Get", {}).get(self.collection_name, [])
        return sorted({o["paper_id"] for o in objects if o.get("paper_id")})