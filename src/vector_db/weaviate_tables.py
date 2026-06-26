import logging
from typing import List
from src.document.base import Document
from src.generation.table_summary_generation import TableSummaryGenerator
from config.global_config import CONFIG
import weaviate
from dotenv import load_dotenv
from src.vector_db.WeaviateDB import WeaviateDB
import structlog

load_dotenv()
logger = structlog.get_logger()

class WeaviateTableDB(WeaviateDB):

    def __init__(self, collection_name: str | None = None, embedder=None):
        super().__init__()
        logger.info("Initializing WeaviateTableDB")
        default_collection = CONFIG.get('table_weaviate', {}).get('collection_name', "Tables")
        clean_name = self.sanitize_collection_name(collection_name)
        self.collection_name = f"Tables_{clean_name}" if clean_name else default_collection
        logger.info(f"Using collection name: {self.collection_name}")
        self.summarizer = TableSummaryGenerator()
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
                        {"name": "summary", "dataType": ["text"]},
                        {"name": "table_index", "dataType": ["int"]},
                        {"name": "total_tables", "dataType": ["int"]},
                        {"name": "paper_id", "dataType": ["text"]},
                    ]
                })
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists.")
        except Exception as e:
            logger.error(f"Error setting up collection: {e}")
            raise

    def _add_document(self, doc: Document) -> None:
        paper_id = doc.metadata.get("paper_id")

        if paper_id is None:
            raise AssertionError("Paper requires 'paper_id'")
        for i, table in enumerate(doc.content['tables']):
            summary = self.summarizer.generate(table)
            vector = self.model.encode(summary).tolist()
            try:
                self.weaviate_client.data_object.create(
                    data_object={
                        "content": table,
                        "summary": summary,
                        "table_index": i,
                        "total_tables": len(doc.content['tables']),
                        "paper_id": paper_id,
                    },
                    class_name=self.collection_name,
                    vector = vector
                )
                logger.info(f"Inserted table {i} from paper {paper_id}")
            except Exception as e:
                logger.error(f"Failed to insert table {i}: {e}")

    def add(self, documents: List[Document]) -> None:
        logger.info(f"Adding {len(documents)} documents")

        contents = []
        summaries = []
        metadata = []

        for doc in documents:
            paper_id = doc.metadata.get("paper_id")
            if paper_id is None:
                raise AssertionError("Paper requires 'paper_id'")

            tables = doc.content.get('tables', [])
            total_tables = len(tables)

            for i, table in enumerate(tables):
                summary = self.summarizer.generate(table)

                contents.append(table)
                summaries.append(summary)
                metadata.append({
                    "paper_id": paper_id,
                    "table_index": i,
                    "total_tables": total_tables
                })

        logger.info(f"Generated {len(summaries)} table summaries")

        vectors = self.model.encode(
            summaries,
            batch_size=32,
            show_progress_bar=False
        )

        logger.info("Embeddings generated")

        with self.weaviate_client.batch as batch:
            batch.batch_size = 50  

            for content, summary, meta, vector in zip(contents, summaries, metadata, vectors):
                batch.add_data_object(
                    data_object={
                        "content": content,
                        "summary": summary,
                        "table_index": meta["table_index"],
                        "total_tables": meta["total_tables"],
                        "paper_id": meta["paper_id"],
                    },
                    class_name=self.collection_name,
                    vector=vector.tolist()
                )

        logger.info("All tables inserted")

    def search(self, query: str, k: int = 5, filter=None) -> List:
        return self.search_embedding(query=query, k=k, filter=filter)

    def search_embedding(self, query_embedding=None, query: str = None, k: int = 5, filter=None) -> List:
        if query_embedding is None:
            raise ValueError("query_embedding is required for near_vector search")
        q = (
            self.weaviate_client.query
            .get(self.collection_name, ["content", "summary", "paper_id", "table_index"])
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