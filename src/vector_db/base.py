from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

from src.document.base import Document

class VectorDB(ABC):
    @abstractmethod
    def add(self, documents: List[Document]) -> None:
        """Add documents and their embeddings to the database"""
        pass
        
    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        pass

        
    @abstractmethod
    def delete(self, filter: Dict[str, Any]) -> None:
        """Delete documents matching the filter"""
        pass