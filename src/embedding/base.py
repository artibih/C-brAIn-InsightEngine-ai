from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
from src.document.base import Document

class Embedder(ABC):
    @abstractmethod
    def embed_documents(self, documents: List[Document]) -> List[np.ndarray]:
        """Create embeddings for documents"""
        pass
        
    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Create embedding for a query"""
        pass

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        res = []
        for query in queries:
            res.append(self.embed_query(query))
        return res