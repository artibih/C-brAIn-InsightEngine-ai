import logging
from typing import List, Optional
from src.document.base import Document
from config.global_config import CONFIG
import weaviate
from dotenv import load_dotenv
from src.vector_db.WeaviateDB import WeaviateDB
import structlog

load_dotenv()
logger = structlog.get_logger()

class WeaviateMetadataDB(WeaviateDB):

    def __init__(self, collection_name: str | None = None):
        super().__init__()
        logging.info("Initializing WeaviateMetadataDB...")
        default_collection = CONFIG.get('weaviate_metadata', {}).get('collection_name', "ResearchPaperMetadata_v1")
        clean_name = self.sanitize_collection_name(collection_name)
        self.collection_name = f"PaperMetadata_{clean_name}" if clean_name else default_collection
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
                        {"name": "paper_id", "dataType": ["text"]},
                        {"name": "doi", "dataType": ["text"]},
                        {"name": "title", "dataType": ["text"]},
                        {"name": "authors", "dataType": ["text"]},
                        {"name": "abstract", "dataType": ["text"]},
                    ]
                })
                logging.info("Collection '%s' created.", self.collection_name)
            else:
                logging.info("Collection '%s' found.", self.collection_name)
        except Exception as e:
            logging.error("Error setting up collection: %s", e)
            raise

    def add(self, documents: List[Document]) -> List:
        results = []
        for doc in documents:
            try:
                data_object={
                        "paper_id": doc.id,
                        "doi": doc.doi if doc.doi else None,
                        "title": getattr(doc, "title", ""),
                        "authors": getattr(doc, "authors", ""),
                        "abstract": getattr(doc, "abstract", ""),
                    }
                
                response = self.weaviate_client.data_object.create(
                    data_object=data_object,
                    class_name=self.collection_name,
                )
                results.append(response)
            except Exception as e:
                logging.error("Metadata insertion failed: %s", e)
        return results

    def search(self, paper_id: str) -> Optional[str]:
        try:
            result = (
                self.weaviate_client.query
                .get(self.collection_name, ["doi"])
                .with_where({
                    "path": ["paper_id"],
                    "operator": "Equal",
                    "valueText": paper_id,
                })
                .with_limit(1)
                .do()
            )
            objects = result.get("data", {}).get("Get", {}).get(self.collection_name, [])
            return objects[0].get("doi") if objects else None
        except Exception as e:
            logging.error("Search failed: %s", e)
            return None

    def fetch_by_paper_ids(self, paper_ids: List[str]) -> List:
        if not paper_ids:
            return []
        operands = [
            {"path": ["paper_id"], "operator": "Equal", "valueText": pid}
            for pid in paper_ids
        ]
        where_filter = {"operator": "Or", "operands": operands} if len(operands) > 1 else operands[0]
        result = (
            self.weaviate_client.query
            .get(self.collection_name, ["paper_id", "doi", "title", "authors", "abstract"])
            .with_where(where_filter)
            .with_limit(1000)
            .do()
        )
        objects = result.get("data", {}).get("Get", {}).get(self.collection_name, [])
        return [type('obj', (object,), {'properties': o})() for o in objects]

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