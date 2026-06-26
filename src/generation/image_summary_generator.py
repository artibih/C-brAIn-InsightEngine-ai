import logging
import os
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
from src.generation.base import LLMResponseGenerator

from config.settings import settings
from config.global_config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageSummaryGenerator(LLMResponseGenerator):
    prompt = """You are an expert in summarizing images from scientific papers concisely and accurately for RAG applications.
You will receive an image as additional input.
Summarize its key insights, trends, and takeaways in at most 100 words and no more than 1000 characters.  
The summary will be embedded as a vector and used in a RAG pipeline.  

ANSWER:
"""

    def __init__(self):
        super().__init__()
        load_dotenv()

       
        self.api_url = CONFIG.get('image_summary_generator', {}).get('base_url', "https://api.openai.com/v1")
        self.model_name = CONFIG.get('image_summary_generator', {}).get('model', "gpt-4o-mini")

        self.api_key = os.getenv('AZURE_OPENAI_API_KEY')
        
        if not self.api_key:
            logger.error("AZURE_OPENAI_API_KEY environment variable is not set.")
            raise ValueError("API key is missing.")

        self.llm_client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        logger.info("OpenAI client initialized")

    def generate(self, query: str, **kwargs) -> str:
        logger.info(f"Generating summary for image URL: {query[:10]}")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": self.prompt},
                {"type": "image_url", "image_url": {"url": query}}
            ]
        }]

        try:
            logger.info("Sending request to OpenAI API")
            completion = self.llm_client.chat.completions.create(
                model=settings.azure_openai_chat_deployment,
                messages=messages,
                temperature=0.7,
                stream=False
            )
            logger.info("Received response from OpenAI API")
        except Exception as e:
            logger.error(f"Error during API request: {e}")
            raise

        summary = completion.choices[0].message.content
        logger.info(f"Summary generated: {summary}")
        return summary
