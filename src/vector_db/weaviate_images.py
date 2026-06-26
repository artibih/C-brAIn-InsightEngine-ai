import logging
from typing import List
from src.document.base import Document
from src.generation.image_summary_generator import ImageSummaryGenerator
from config.global_config import CONFIG
import weaviate
from dotenv import load_dotenv
from src.vector_db.WeaviateDB import WeaviateDB
import structlog

load_dotenv()
logger = structlog.get_logger()

class WeaviateImageDB(WeaviateDB):

    def __init__(self, collection_name: str | None = None, embedder=None):
        super().__init__()
        logging.info("Initializing WeaviateImageDB")
        
        default_collection = CONFIG.get('weaviate_images', {}).get('collection_name', "Images")
        clean_name = self.sanitize_collection_name(collection_name)
        self.collection_name = f"Images_{clean_name}" if clean_name else default_collection
        
        logger.info(f"Using collection name: {self.collection_name}")
        self.summarizer = ImageSummaryGenerator()
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
                        {"name": "image_index", "dataType": ["int"]},
                        {"name": "total_images", "dataType": ["int"]},
                        {"name": "paper_id", "dataType": ["text"]},
                    ]
                })
            else:
                logging.info(f"Collection {self.collection_name} already exists.")
        except Exception as e:
            logging.error(f"Error setting up collection: {e}")
            raise

    @staticmethod
    def _is_image_url(s: str) -> bool:
        if not s or not isinstance(s, str):
            return False
        s = s.strip()
        return s.startswith("http://") or s.startswith("https://") or s.startswith("data:image/")

    def _image_to_summary(self, image: str) -> str:
        if not image or not isinstance(image, str):
            return ""
        image = image.strip()
        if self._is_image_url(image):
            try:
                return self.summarizer.generate(image)
            except Exception as e:
                logging.warning("Vision API failed: %s", e)
                return (image[:1000] + "...") if len(image) > 1000 else image
        return image

    def _add_document(self, doc: Document) -> None:
        paper_id = doc.metadata.get("paper_id")
        if paper_id is None:
            raise AssertionError("Paper requires 'paper_id'")

        images = doc.content.get("images", [])
        if not isinstance(images, list):
            images = [images] if images else []

        summaries = []
        metadata = []

        for i, image in enumerate(images):
            summary = self._image_to_summary(image)
            if not summary:
                continue

            summaries.append(summary)
            metadata.append({
                "paper_id": paper_id,
                "image_index": i,
                "total_images": len(images)
            })

        if not summaries:
            return

        vectors = self.model.encode(
            summaries,
            batch_size=32,
            show_progress_bar=False
        )

        with self.weaviate_client.batch as batch:
            batch.batch_size = 50

            for summary, meta, vector in zip(summaries, metadata, vectors):
                batch.add_data_object(
                    data_object={
                        "content": summary,
                        "image_index": meta["image_index"],
                        "total_images": meta["total_images"],
                        "paper_id": meta["paper_id"],
                    },
                    class_name=self.collection_name,
                    vector=vector.tolist()
                )

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
            .get(self.collection_name, ["content", "paper_id", "image_index"])
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