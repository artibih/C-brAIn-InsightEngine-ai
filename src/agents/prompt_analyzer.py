from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from apps.api.schemas.rag import PromptAnalyzeResponse
from config.llm_config import get_llm 
from src.prompts.prompt_analyzer_prompt import PROMPT_ANALYZER_SYSTEM_TEMPLATE

class PromptAnalyzer:
    def __init__(self):
        self.llm = get_llm(workload="online")
        self.parser = PydanticOutputParser(pydantic_object=PromptAnalyzeResponse)

    async def analyze(self, draft_query: str) -> PromptAnalyzeResponse:
        prompt = ChatPromptTemplate.from_template(PROMPT_ANALYZER_SYSTEM_TEMPLATE)
        
        chain = prompt | self.llm | self.parser
        
        return await chain.ainvoke({
            "draft_query": draft_query,
            "format_instructions": self.parser.get_format_instructions()
        })