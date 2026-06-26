import fitz
import json
import asyncio
from openai import AsyncAzureOpenAI
from config.settings import settings
import structlog
logger = structlog.get_logger(__name__)

def extract_first_pages(pdf_path, n_pages=2):
    doc = fitz.open(pdf_path)
    text = ""
    for i in range(min(n_pages, len(doc))):
        text += doc[i].get_text()


client_llm = AsyncAzureOpenAI(
    api_key=settings.azure_openai_api_key,
    azure_endpoint=settings.azure_openai_endpoint,
    api_version=settings.azure_openai_api_version
)

async def extract_metadata(text):
    prompt = f"""
Extract the metadata from the following academic paper text.

Return ONLY valid JSON. Do NOT wrap in markdown.
Do NOT use ```json or ``` blocks. No explanation.

Format:
{{
  "title": "...",
  
  "authors": "Author1, Author2, Author3",
  "abstract": "..."
}}

If abstract is missing, use null.

Text:
\"\"\"{text}\"\"\"
"""

    response = await client_llm.chat.completions.create(
        model=settings.azure_openai_deployment_online,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

def parse_llm_output(output_text):
    try:
        return json.loads(output_text)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse LLM output: {output_text}")
        return None

if __name__ == "__main__":
    pdf_path = "path-to-the-paper"
    output_json_path = "metadata_output.json"

    text = extract_first_pages(pdf_path)[:12000]

    llm_output = asyncio.run(extract_metadata(text))
    metadata = parse_llm_output(llm_output)

    if metadata:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Metadata saved to {output_json_path}")
    else:
        logger.error("Metadata extraction failed")