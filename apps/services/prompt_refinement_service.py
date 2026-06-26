from src.agents.prompt_analyzer import PromptAnalyzer
from apps.api.schemas.rag import PromptAnalyzeResponse

class PromptRefinementService:
    def __init__(self):
        self.analyzer = PromptAnalyzer()

    async def analyze_draft(self, draft_query: str) -> PromptAnalyzeResponse:
        """
        Orchestrates the prompt analyzer agent and returns the structured response.
        """
        return await self.analyzer.analyze(draft_query)