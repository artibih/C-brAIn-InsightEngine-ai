from pydantic import BaseModel, Field
from typing import List, Optional

from config.llm_selection import LlmSelectionRequest

class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)

class RAGRetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    dataset_paths: Optional[List[str]] = None
    user_id: Optional[str] = None
    async_execution: bool = True
    llm_selection: Optional[LlmSelectionRequest] = None

class RAGChunk(BaseModel):
    text: str    
    source: str    
    chunk: int    
    score: Optional[float] = None
    
class RAGQueryResponse(BaseModel):
    answer: str    
    chunks: List[RAGChunk]

class PromptAnalyzeRequest(BaseModel):
    draft_query: str = Field(..., min_length=1)

class PromptAnalyzeResponse(BaseModel):
    suggestion: str