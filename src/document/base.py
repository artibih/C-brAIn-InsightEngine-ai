from typing import Dict, Any, Optional
import os
import uuid

class Document:
    def __init__(self, text: str, metadata: Dict[str, Any] = None):
        self.content = {
            'text' : text
        }
        self.metadata = metadata.copy() if metadata else {}

        self.metadata.setdefault("paper_id", str(uuid.uuid4()))
        self.metadata.setdefault("doi", None)
        self.metadata.setdefault("title", None)
        self.metadata.setdefault("authors", None)
        self.metadata.setdefault("abstract", None)
        self.metadata.setdefault("original_path", None)
    
    @property
    def id(self):
        # Getter for paper_id
        return self.metadata['paper_id']
    
    @property
    def original_path(self):
        # Getter for paper_id
        return self.metadata['original_path']
    
    @property
    def doi(self)-> Optional[str]:
        # Getter for doi
        return self.metadata.get("doi") or None
    

    @doi.setter
    def doi(self, value: Optional[str]):
        if not value:
            self.metadata["doi"] = None
        else:
            self.metadata["doi"] = str(value)
    
    @property
    def title(self) -> Optional[str]:
        return self.metadata.get("title")
    
    @title.setter
    def title(self, value: Optional[str]):
        self.metadata["title"] = value or None
    
    @property
    def authors(self) -> Optional[str]:
        return self.metadata.get("authors")

    @authors.setter
    def authors(self, value: Optional[str]):
        self.metadata["authors"] = value or None

    @property
    def abstract(self) -> Optional[str]:
        return self.metadata.get("abstract")

    @abstract.setter
    def abstract(self, value: Optional[str]):
        self.metadata["abstract"] = value or None

    @property
    def original_path(self) -> Optional[str]:
        return self.metadata.get("original_path")
    

    @property
    def text(self):
        # Getter
        return self.content['text']

    @text.setter
    def text(self, value):
        # Setter
        if not value:
            raise ValueError("Name cannot be empty")
        self.content['text'] = value
    
    @property
    def delete_filter(self):
        """
        Standard delete filter for this document in Weaviate
        """
        return {
            "paper_id": self.metadata["paper_id"]
        }

    
    def __repr__(self):
        return f"Document(content={self.content['text'][:50]}..., metadata={self.metadata})"

        
