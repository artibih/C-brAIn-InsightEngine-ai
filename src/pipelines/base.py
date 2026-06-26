from abc import ABC, abstractmethod
from typing import List, Any, Dict
from src.document.base import Document

class RagPipeline(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        pass

    def add_document(self, document: Document) -> None:
        pass
        
    @abstractmethod
    def retrieve(self, query: str) -> List[Any]:
        pass
        
    @abstractmethod
    def generate_response(self, query: str) -> str:
        pass

    @abstractmethod
    def generate_enhanced_response(self, query:str, experiment_id: str) -> Dict[str, Any]:
        pass