import logging
from typing import List, Dict, Any
from src.document.base import Document
from src.vector_db.base import VectorDB
from config.global_config import CONFIG
import weaviate
from dotenv import load_dotenv
import os
from config.settings import settings 
from sentence_transformers import SentenceTransformer
import re
import threading
load_dotenv()  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_weaviate_client_cache = {}

class WeaviateDB(VectorDB):
    _model = None
    _model_lock = threading.Lock()
    def __init__(self):
        super().__init__()
        
        logging.info("Initializing WeaviateChunkedDB...")
        self.weaviate_config = CONFIG.get('weaviate_db', {})
        self.host = self.weaviate_config.get('host', "localhost")
        self.weaviate_client = self.connect()
        

    def get_weaviate_client(self):
            try:
                if self.host not in _weaviate_client_cache:
                    if self.host != "localhost":
                        logging.info("Connecting to remote Weaviate at %s", settings.weaviate_host)
                        url = f"https://{settings.weaviate_host}"
                        
                        client = weaviate.Client(url)
                        print("✅ Connected to Weaviate.")
                    else:
                        logging.info("Connecting to local Weaviate at %s", self.host)
                        port = int(os.environ.get('WEAVIATE_PORT', '8080'))
                        client = weaviate.Client(f"http://{self.host}:{port}")
                        logging.info("Successfully connected to Weaviate.")

                    _weaviate_client_cache[self.host] = client
                    schema = client.schema.get()
                    classes = [c['class'] for c in schema.get('classes', [])]
                    print("📦 Collections available:")
                    for name in classes:
                        print(f" - {name}")
                return _weaviate_client_cache[self.host]
            except Exception:
                logging.exception("Error connecting to Weaviate")
                raise
        
    def connect(self):
        client = self.get_weaviate_client()
        return client
            
    

    def add(self, documents: List[Document]) -> None:
        """Add documents and their embeddings to the database"""
        pass
        
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        pass

    @property
    def model(self):
        if WeaviateDB._model is None:
            with WeaviateDB._model_lock:
                if WeaviateDB._model is None:
                    logging.info("Loading SentenceTransformer model (first time only)...")
                    WeaviateDB._model = SentenceTransformer(
                        "sentence-transformers/all-mpnet-base-v2"
                    )
        return WeaviateDB._model
           
    def delete(self, filter: None) -> None:
        """
        Delete documents matching the given Weaviate Filter.

        Example:
            Filter.by_property("document_id").equal("doc_123")
        """
        try:
            if not filter:
                raise ValueError("Refusing to delete without a filter")

            collection_name = self.weaviate_config.get("collection_name")

            if not collection_name:
                raise ValueError("Weaviate collection_name not configured")

            collection = self.weaviate_client.collections.get(collection_name)

            logging.info("Deleting objects from Weaviate with filter: %s", filter)

            result = collection.data.delete_many(
                where=filter
            )

            logging.info("Delete result: %s", result)

        except Exception as e:
            logging.error("Failed to delete objects from Weaviate: %s", str(e))
            raise

    def sanitize_collection_name(self, name: str | None) -> str | None:
        if not name:
            return None

        cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")

        if cleaned and not cleaned[0].isalpha():
            cleaned = f"A{cleaned}"

        return cleaned
    

            