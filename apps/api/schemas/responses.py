# this is a script that defines response schemas for the API
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class TestPlanResponse(BaseModel):
    """Response schema for test plan details."""
    test_plan: str
    datasets_needed: List[str]
    
class HypothesisTestResponse(BaseModel):
    """Response schema for hypothesis test results."""
    hypothesis: str
    test_plan: str
    datasets_used: List[Dict[str, Any]]
    literature_used: List[Dict[str, Any]]
    analysis_results: Dict[str, Any]
    synthesis: str
    confidence_score: Optional[float]


class ErrorResponse(BaseModel):
    """Response schema for error messages."""
    error: str
    details: Optional[str]
class AsyncTestResponse(BaseModel):
    """Response schema for asynchronous hypothesis test submission."""
    task_id: str
    status_url: str