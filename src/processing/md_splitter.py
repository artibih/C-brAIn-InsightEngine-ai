import re
import logging
from typing import List, Dict, Optional, Tuple
from src.processing.base import DocumentPreprocessor
from src.document.base import Document
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarkdownDocumentPreprocessor(DocumentPreprocessor): 

    def __init__(self):
        super().__init__()
        logger.debug("MarkdownDocumentPreprocessor initialized.")

    def extract_doi(self, text: str):
        doi_pattern = r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b'
        matches = re.findall(doi_pattern, text, flags=re.IGNORECASE)
        return matches[0] if matches else None


    def preprocess(self, document: Document) -> Document:
        logger.info("Preprocessing document with ID: %s", document.id)
        
        full_text = document.content['text']
        doi = self.extract_doi(full_text)
        existing_images = document.content.get("images", [])

        sections_text = []
        tables = []

        logger.debug("Removing references from document content.")
        full_text = self._remove_references(full_text)

        logger.debug("Splitting text into sections.")
        sections = self._split_text_to_sections(full_text)

        for section in sections:
            logger.debug("Processing section: %s", section[:50]) 

            section_tables, paragraphs = self._split_table_and_text(section)

            logger.debug("Found %d table(s) in section.", len(section_tables))
            for table in section_tables:
                tables.append(table)
            sections_text.append(paragraphs)

        document.content['sections'] = sections_text
        document.content['tables'] = tables
        document.content['images'] = existing_images

        if doi:
            document.doi = doi
        elif document.metadata.get("doi"):
            pass 
        else:
            document.doi = None

        logger.info("Preprocessing completed for document with ID: %s", document.id)
        return document

    def _remove_references(self, text: str) -> str:
        logger.debug("Removing references section from the text.")
        return re.sub(
            r'##\s*(References|Bibliography|Literature)\b.*', '', text, flags=re.IGNORECASE | re.DOTALL)

    def _split_text_to_sections(self, text: str) -> List[str]:
        logger.debug("Splitting text into sections based on headers.")
        
        pattern = r"(##\s+[^\n]+)"  
        
        parts = re.split(pattern, text)
        
        sections = []        
        for i in range(1, len(parts), 2):  
            section_name = parts[i].strip()
            section_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
            sections.append('\n\n'.join([section_name, section_text]))

        logger.debug("Split text into %d section(s).", len(sections))
        return sections

    def _split_table_and_text(self, text: str) -> Tuple[str, List[str]]:
        logger.debug("Splitting table(s) and text from the content.")

        text = re.sub(r'\|\|', '|\n|', text)

        regex = r"(\|.*\|)"
        
        tables = []
        paragraphs = ""
        table_lines = []
        
        for line in text.split('\n'):
            if re.match(regex, line):
                table_lines.append(line)
            else:
                if table_lines:
                    tables.append('\n'.join(table_lines))
                    table_lines = [] 
                paragraphs = '\n'.join([paragraphs, line])
        
        if table_lines:
            tables.append('\n'.join(table_lines))

        logger.debug("Found %d table(s) and processed paragraphs.", len(tables))
        return tables, paragraphs
