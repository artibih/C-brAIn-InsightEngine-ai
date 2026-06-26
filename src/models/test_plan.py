from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class DatasetRequirement(BaseModel):
    """Required dataset specification."""
    
    dataset_type: str 
    criteria: Dict[str, Any] 
    required_variables: List[str]
    minimum_sample_size: Optional[int] = None


class AnalysisStep(BaseModel):
    """Individual analysis step in test plan."""
    
    step_number: int
    description: str
    tool: str 
    parameters: Dict[str, Any]
    expected_output: str


class TestPlan(BaseModel):
    """Test plan for hypothesis validation."""
    
    id: str = Field(default_factory=lambda: f"plan_{datetime.utcnow().timestamp()}")
    hypothesis_id: str
    
    objective: str
    datasets_required: List[DatasetRequirement]
    methodology_checks: List[str]
    analysis_steps: List[AnalysisStep]
    validation_criteria: List[str]
    
    estimated_duration_minutes: Optional[int] = None
    status: str = "pending"  
    
    executed_steps: List[int] = Field(default_factory=list)
    findings: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)