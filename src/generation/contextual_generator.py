import logging

import os
from dotenv import load_dotenv

from src.generation.base import LLMResponseGenerator

from config.llm_config import ChatCompletionGateway
from config.llm_selection import ResolvedLlmSelection
from config.global_config import CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContextualGenerator(LLMResponseGenerator):
    prompt = """You are a research assistant with expertise in analyzing and summarizing scientific papers.

You will receive numbered research sources in the CONTEXT section.

Your task:
- Answer the QUESTION using ONLY the provided sources.
- Cite sources using square brackets like [1], [2].
- Every factual claim MUST include at least one citation.
- If multiple sources support a claim, cite them like [1][3].
- Do NOT invent citations.
- If the answer is not supported by the provided sources, respond with:
  "Not supported by provided sources."

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:
"""

    def __init__(self, llm_selection: ResolvedLlmSelection | None = None):
        super().__init__()
        self.llm_selection = llm_selection
        load_dotenv()

        self.api_url = CONFIG.get('contextual_generator', {}).get('base_url', "https://api.openai.com/v1")
        self.model_name = CONFIG.get('contextual_generator', {}).get('model', "gpt-4")
        self.api_key = os.getenv('AZURE_OPENAI_API_KEY')

        if not self.api_key:
            logger.error("AZURE_OPENAI_API_KEY not found in environment variables.")

        self.gateway = ChatCompletionGateway()
        logger.info("Chat completion gateway initialized.")

    def generate(self, query: str, **kwargs) -> str:
        logger.info("Generating response for query: %s", query)

        chunks = kwargs.get('context', [])
        if not chunks:
            logger.warning("No context provided for the query.")
        
        numbered_chunks = []
        for idx, chunk in enumerate(chunks, start=1):
            if isinstance(chunk, str):
                content = chunk.strip()
            elif isinstance(chunk, dict):
                raw_content = chunk.get("content", "")
                content = raw_content.strip() if isinstance(raw_content, str) else ""
            else:
                content = ""
            if content:
                numbered_chunks.append(f"[{idx}] {content}")
        context = "\n\n".join(numbered_chunks)

        logger.debug("Formatted context: %s", context[:200]) 

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": self.prompt.format(context_text=context, query=query)},
            ]
        }]

        try:
            answer = self.gateway.complete(
                messages=messages,
                temperature=0.7,
                workload="online",
                llm_selection=self.llm_selection,
            )
            logger.info("Response generated successfully.")
        except Exception as e:
            logger.error("Error generating response: %s", e)
            raise

        logger.debug("Generated answer: %s", answer[:200])  

        return answer
