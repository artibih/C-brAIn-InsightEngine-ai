import logging
from typing import List
from src.document.base import Document
from src.generation.paper_summary_generator import PaperSummaryGenerator
from config.global_config import CONFIG
from config.settings import settings
import weaviate
from dotenv import load_dotenv
from src.vector_db.WeaviateDB import WeaviateDB
import structlog

load_dotenv()
logger = structlog.get_logger()

class WeaviateSummaryDB(WeaviateDB):

    def __init__(self, collection_name: str | None = None, embedder=None):
        super().__init__()
        logger.info("Initializing WeaviateSummaryDB")
        default_collection = CONFIG.get('weaviate_summaries', {}).get('collection_name', "Summary")
        clean_name = self.sanitize_collection_name(collection_name)
        self.collection_name = f"Summary_{clean_name}" if clean_name else default_collection
        logger.info(f"Using collection name: {self.collection_name}")
        self.summarizer = PaperSummaryGenerator()
        self._setup_collection()

    def _setup_collection(self):
        try:
            schema = self.weaviate_client.schema.get()
            existing = [c['class'] for c in schema.get('classes', [])]
            if self.collection_name not in existing:
                logger.info(f"Creating new collection: {self.collection_name}")
                self.weaviate_client.schema.create_class({
                    "class": self.collection_name,
                    "vectorizer": "none",
                    "properties": [
                        {"name": "content", "dataType": ["text"]},
                        {"name": "paper_id", "dataType": ["text"]},
                    ]
                })
            else:
                logger.info(f"Collection {self.collection_name} already exists.")
        except Exception as e:
            logger.error(f"Error setting up collection: {e}")
            raise

    def _add_document(self, doc: Document) -> None:
        paper_id = doc.metadata.get("paper_id")

        if paper_id is None:
            raise AssertionError("Paper requires 'paper_id'")
        summary = self.summarizer.generate(doc.content['text'])
        try:
            vector = self.model.encode(summary).tolist()

            self.weaviate_client.data_object.create(
                data_object={
                    "content": summary,
                    "paper_id": paper_id
                },
                class_name=self.collection_name,
                vector=vector
            )
            logger.info(f"Inserted document with paper_id: {paper_id}")
        except Exception as e:
            logger.error(f"Failed to insert document: {e}")
            raise

    def add(self, documents: List[Document]) -> None:
        logger.info(f"Adding {len(documents)} documents")

        paper_ids = []
        summaries = []

        for doc in documents:
            paper_id = doc.metadata.get("paper_id")
            if paper_id is None:
                raise AssertionError("Paper requires 'paper_id'")

            summary = self.summarizer.generate(doc.content['text'])

            paper_ids.append(paper_id)
            summaries.append(summary)

        logger.info("Summaries generated")

        vectors = self.model.encode(
            summaries,
            batch_size=32,
            show_progress_bar=False
        )

        logger.info("Embeddings generated")

        with self.weaviate_client.batch as batch:
            batch.batch_size = 20  

            for paper_id, summary, vector in zip(paper_ids, summaries, vectors):
                batch.add_data_object(
                    data_object={
                        "content": summary,
                        "paper_id": paper_id
                    },
                    class_name=self.collection_name,
                    vector=vector.tolist()
                )

        logger.info("All documents inserted")

    def search(self, query: str, k: int = 5) -> List:
        return self.search_embedding(query=query, k=k)

    def search_embedding(self, query_embedding=None, query: str = None, k: int = 5) -> List:
        if query_embedding is None:
            raise ValueError("query_embedding is required for near_vector search")
        logger.info("Searching using near_text (REST)")
        try:
            result = (
                self.weaviate_client.query
                .get(self.collection_name, ["content", "paper_id"])
                .with_near_vector({"vector": query_embedding})
                .with_limit(k)
                .with_additional(["vector"])
                .do()
            )
            return result.get("data", {}).get("Get", {}).get(self.collection_name, [])
        except Exception as e:
            logger.error(f"near_text search failed: {e}")
            raise

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