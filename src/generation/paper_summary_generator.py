import os
import logging
from config import settings
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
from src.document.base import Document
from src.generation.base import LLMResponseGenerator

from config.settings import settings
from config.global_config import CONFIG

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperSummaryGenerator(LLMResponseGenerator):
    prompt = """You are an expert in summarizing scientific papers concisely and accurately for RAG applications.
You will receive a paper in Markdown as additional input.
Summarize its key insights, trends, and takeaways in at most 100 words and no more than 1000 characters.  
The summary will be embedded as a vector and used in a RAG pipeline.  

The summary should:
1. Capture the main research question, methodology, and key findings
2. Highlight the most important contributions to the field
3. Be clear, concise, and academically rigorous

PAPER_TEXT:
{text}

ANSWER:
"""

    def __init__(self):
        super().__init__()
        load_dotenv()

        self.api_url = CONFIG.get('paper_summary_generator', {}).get('base_url', "https://api.openai.com/v1")
        self.model_name = CONFIG.get('paper_summary_generator', {}).get('model', "gpt-4")

        self.api_key = os.getenv('AZURE_OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("AZURE_OPENAI_API_KEY environment variable not set.")
            raise ValueError("API key is missing.")
        self.llm_client = AzureOpenAI(
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
            )
        
        logger.info("SummaryGenerator initialized.")

    def generate(self, query: str, **kwargs) -> str:
        logger.info("Generating summary for provided paper text.")
        messages = [{
            "role": "user",
            "content": self.prompt.format(text=query)
        }]

        try:
            completion = self.llm_client.chat.completions.create(
                model=settings.azure_openai_chat_deployment,
                messages=messages,
                temperature=0.7,
                stream=False
            )
            logger.info("Summary generated successfully.")
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error during OpenAI API call: {e}")
            raise
