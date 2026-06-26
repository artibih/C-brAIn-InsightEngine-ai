import os
import requests
import json
import time
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Union
import numpy as np
from src.embedding.base import Embedder
from src.document.base import Document
import yaml
from sentence_transformers import SentenceTransformer

class MpnetEmbedder(Embedder):
    max_retries = 3
    retry_delay = 1

    def __init__(self):
        super().__init__()
        load_dotenv()
        
        self.model = None
        try:
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda')
        except:
            pass
        
        if self.model is None:
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    def embed_documents(self, documents: List[Document]) -> List[np.ndarray]:
        return [self.embed_query(doc.content['text']) for doc in documents]
        
    def embed_query(self, query: str) -> np.ndarray:
        embeddings = self.model.encode(query)
        return embeddings

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        res = self.model.encode(queries)
        return res
