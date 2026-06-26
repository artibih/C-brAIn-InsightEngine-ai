from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from src.document.base import Document

class DocumentPreprocessor(ABC):  
    @abstractmethod
    def preprocess(self, document: Document) -> Document:
        pass
    
    def preprocess_batch(self, documents: List[Document]) -> List[Document]:
        """
        Preprocess multiple docuements
        """
        res = []
        for document in documents:
            res.append(self.preprocess(document))
        return res
