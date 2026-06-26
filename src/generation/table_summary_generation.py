import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI

from config.settings import settings
from src.document.base import Document
from src.generation.base import LLMResponseGenerator

from config.global_config import CONFIG

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TableSummaryGenerator(LLMResponseGenerator):
    prompt = """You are an expert in summarizing scientific tables concisely and accurately for RAG applications.
You will receive a table in Markdown as input.
Summarize its key insights, trends, and takeaways in at most 100 words and no more than 1000 characters.  
The summary will be embedded as a vector and used in a RAG pipeline.  

TABLE:
{table}

ANSWER:
"""

    def __init__(self):
        super().__init__()
        logger.debug("Initializing TableSummaryGenerator...")

        load_dotenv()
        
        self.api_url = CONFIG.get('table_summary_generator', {}).get('base_url', "https://api.openai.com/v1")
        self.model_name = CONFIG.get('table_summary_generator', {}).get('model', "gpt-4")

        self.api_key = os.getenv('AZURE_OPENAI_API_KEY')

        if not self.api_key:

            logger.error("AZURE_OPENAI_API_KEY is missing from environment variables.")
            raise ValueError("API key is required for Azure OpenAI")
        
        self.llm_client = AzureOpenAI(
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
            )
        logger.info("TableSummaryGenerator initialized successfully.")

    def generate(self, query: str, **kwargs) -> str:
        logger.debug(f"Generating summary for table with query: {query[:50]}...") 
        try:
            messages = [{
                "role": "user",
                "content": self.prompt.format(table=query)
            }]

            logger.info("Sending request to OpenAI API...")
            completion = self.llm_client.chat.completions.create(
                model=settings.azure_openai_chat_deployment,
                messages=messages,
                temperature=0.7,
                stream=False
            )

            summary = completion.choices[0].message.content
            logger.info("Received summary from OpenAI API.")
            return summary

        except Exception as e:
            logger.error(f"Error during OpenAI API request: {e}")
            raise

