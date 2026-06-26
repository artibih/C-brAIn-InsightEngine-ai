from typing import Any, Dict, List, Optional, Union, Literal

from pydantic import BaseModel, Field

from config.llm_selection import LlmSelectionRequest


class HypothesisTestRequest(BaseModel):
    """Request schema for hypothesis testing."""
    
    hypothesis: str = Field(..., description="Hypothesis to test")
    dataset_paths: Optional[list[str]] = Field(
        None,
        description="Path to the uploaded dataset (if any)"
    )
    user_id: Optional[str] = Field(None, description="User identifier")
    async_execution: bool = Field(
        default=True,
        description="Execute asynchronously for long-running analyses"
    )

    feedback: Optional[str] = Field(
        None,
        description="Scientist feedback from previous run"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Context from previous graph execution"
    )
    llm_selection: Optional[LlmSelectionRequest] = Field(
        None,
        description="Optional provider/model selection for this chat workflow"
    )

class ReviewParameters(BaseModel):
    """Optional MVP3 reviewer stream controls."""

    tone: Literal["constructive", "critical", "neutral"] = Field(
        default="neutral",
        description=(
            "Review tone. constructive = actionable improvement focus; "
            "critical = stricter risk surfacing; neutral = balanced evidence-first review."
        ),
    )
    depth: Literal["summary", "detailed", "comprehensive"] = Field(
        default="detailed",
        description=(
            "Review depth. summary = concise decisive concerns; detailed = current-style depth; "
            "comprehensive = richer reasoning within the same workflow limits."
        ),
    )
    persona: Literal[
        "editorial_focus",
        "methodological_focus",
        "scientific_rigor_focus",
    ] = Field(
        default="scientific_rigor_focus",
        description=(
            "Reviewer emphasis. editorial_focus = novelty/clarity/publication fit; "
            "methodological_focus = design/statistics/reproducibility/validity; "
            "scientific_rigor_focus = correctness/evidence grounding/unsupported claims."
        ),
    )


class ReviewerTestRequest(BaseModel):
    """Request schema for reviewer testing."""

    dataset_paths: Optional[List[str]] = Field(
        None,
        description="Paths to the uploaded file(s) for review (e.g., PDF paths)"
    )

    user_id: Optional[str] = Field(
        None,
        description="User identifier"
    )

    async_execution: bool = Field(
        default=True,
        description="Execute asynchronously for long-running analyses"
    )

    feedback: Optional[str] = Field(
        None,
        description="User critique of reviewers"
    )

    previous_reviews: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(
        None,
        description="Previous reviewer outputs (dict with events or list of event items)"
    )

    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional execution context"
    )
    llm_selection: Optional[LlmSelectionRequest] = Field(
        None,
        description="Optional provider/model selection for this reviewer workflow"
    )
    review_parameters: ReviewParameters = Field(
        default_factory=ReviewParameters,
        description="Optional controls for review tone, depth, and reviewer emphasis"
    )