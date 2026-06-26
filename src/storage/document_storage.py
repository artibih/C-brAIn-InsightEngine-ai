import os
import shutil
from config.global_config import CONFIG
from typing import List

from src.document.base import Document

class DocumentStorage:
    def __init__(self):
        self.storage_path = CONFIG.get("root_document_storage_path")
        if not self.storage_path:
            raise ValueError("root_document_storage_path must be configured")
        
    def store_document(self, document: Document) -> Document: 
        paper_id = document.metadata.get("paper_id")

        if not paper_id:
            raise ValueError("Document metadata must contain 'paper_id'")
        
        root = os.path.abspath(self.storage_path)
        doc_dir = os.path.abspath(os.path.join(root, paper_id))
        if not doc_dir.startswith(root + os.sep):
            raise ValueError("Invalid paper_id; path traversal detected.")
        
        os.makedirs(doc_dir, exist_ok=True)

        text = document.text
        with open(os.path.join(doc_dir, "mistral.md"), 'w', encoding='utf-8') as md_file:
            md_file.write(text)
        
        original_path = document.metadata.get("original_path", "")
        with open(os.path.join(doc_dir, "path.txt"), 'w', encoding='utf-8') as path_file:
            path_file.write(original_path)
        
        images = document.content.get("images", [])
        for i, img in enumerate(images):
            with open(os.path.join(doc_dir, f"img{i+1}.b64.txt"), 'w', encoding='utf-8') as out_file:
                out_file.write(img)
        print(f"Document with paper_id '{paper_id}' stored at:{doc_dir}")
        return document

    def store_documents(self, documents: List[Document]) -> List[Document]:
        for doc in documents:
            self.store_document(doc)
        return documents