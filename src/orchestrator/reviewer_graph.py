from platform import processor

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any

from src.agents.reviewer_domain_expert import ReviewerDomainExpert
from src.agents.reviewer_journal_editor import ReviewerJournalEditor
from src.agents.reviewer_methodological import ReviewerMethodological
from src.agents.reviewer_benchmark_evidence import ReviewerBenchmarkEvidence
from src.agents.hybrid_query_agent import HybridQueryAgent
from src.agents.document_preprocessor import DocumentPreprocessor
from src.agents.review_synthesizer import ReviewSynthesizer
from utils.load_pdf_from_blob import read_pdf_safely
from config.llm_selection import ResolvedLlmSelection
import time
import structlog
logger = structlog.get_logger()

# -------------------------
# State definition
# -------------------------
class ReviewState(TypedDict):
    experiment_id: str
    files: List[str]
    sections: Dict[str, Any]
    feedback: str  
    previous_reviews: Dict[str, Any] 
    llm_selection: ResolvedLlmSelection | None
    review_parameters: Dict[str, Any]

    review_journal_editor: Dict[str, Any]
    review_domain_expert: Dict[str, Any]
    review_methodological: Dict[str, Any]
    review_benchmark_evidence: Dict[str, Any]

    final_review: Dict[str, Any]


# -------------------------
# Node wrappers
# -------------------------
async def split_node(state: ReviewState):

    experiment_id = state.get("experiment_id")
    processor = DocumentPreprocessor(llm_selection=state.get("llm_selection"))
    logger.info("processing_reviewer_files", files=state["files"])
    sections = await processor.process_files(state["files"])
    return {
        **state,
        "experiment_id": str(experiment_id),
        "sections": sections}


async def synthesis_node(state: ReviewState):
    reviews = [
        state["review_journal_editor"],
        state["review_domain_expert"],
        state["review_methodological"],
        state.get("review_benchmark_evidence"),
    ]
    final = await ReviewSynthesizer(
        llm_selection=state.get("llm_selection"),
    ).synthesize_reviews(
        reviews,
        experiment_id=state["experiment_id"],
        feedback=state.get("feedback"),
        review_parameters=state.get("review_parameters"),
    )
    return {"final_review": final}


# -------------------------
# Graph construction
# -------------------------
def create_review_workflow(hybrid_agent: HybridQueryAgent):
    # Closure-bound node so the benchmark reviewer can access the shared
    # HybridQueryAgent without a module-level singleton.
    async def reviewer_benchmark_evidence_node(state: ReviewState) -> ReviewState:
        experiment_id = state["experiment_id"]

        start = time.perf_counter()
        logger.info(
            "reviewer_benchmark_evidence_node_started",
            experiment_id=experiment_id,
        )

        reviewer = ReviewerBenchmarkEvidence(
            hybrid_agent=hybrid_agent,
            llm_selection=state.get("llm_selection"),
        )
        result = await reviewer.review(
            state["sections"],
            experiment_id=experiment_id,
            feedback=state.get("feedback"),
            previous_reviews=state.get("previous_reviews"),
            review_parameters=state.get("review_parameters"),
        )

        end = time.perf_counter()
        logger.info(
            "reviewer_benchmark_evidence_node_completed",
            experiment_id=experiment_id,
            duration=end - start,
        )

        return {
            "review_benchmark_evidence": {
                "agent": "reviewer_benchmark_evidence",
                "output": result,
            }
        }

    builder = StateGraph(ReviewState)

    # Nodes
    builder.add_node("split", split_node)
    builder.add_node("review_journal_editor", reviewer_journal_editor_node)
    builder.add_node("review_domain_expert", reviewer_domain_expert_node)
    builder.add_node("review_methodological", reviewer_methodological_node)
    builder.add_node("review_benchmark_evidence", reviewer_benchmark_evidence_node)
    builder.add_node("synthesis", synthesis_node)

    # Entry
    builder.set_entry_point("split")

    # Fan-out
    builder.add_edge("split", "review_journal_editor")
    builder.add_edge("split", "review_domain_expert")
    builder.add_edge("split", "review_methodological")
    builder.add_edge("split", "review_benchmark_evidence")

    # Fan-in (join)
    builder.add_edge("review_journal_editor", "synthesis")
    builder.add_edge("review_domain_expert", "synthesis")
    builder.add_edge("review_methodological", "synthesis")
    builder.add_edge("review_benchmark_evidence", "synthesis")

    # End
    builder.add_edge("synthesis", END)

    return builder.compile()


async def reviewer_domain_expert_node(state: ReviewState) -> ReviewState:
    experiment_id = state["experiment_id"]

    start = time.perf_counter()
    logger.info(
        "reviewer_domain_expert_started",
        experiment_id=experiment_id,
    )

    reviewer = ReviewerDomainExpert(llm_selection=state.get("llm_selection"))
    result = await reviewer.review(
        state["sections"],
        experiment_id=experiment_id,
        feedback=state.get("feedback"),
        previous_reviews=state.get("previous_reviews"),
        review_parameters=state.get("review_parameters"),
    )
    
    end = time.perf_counter()
    logger.info(
        "reviewer_domain_expert_completed",
        experiment_id=experiment_id,
        duration=end - start,
    )

    return {
        "review_domain_expert": {
            "agent": "reviewer_domain_expert",
            "output": result
        }
    }

async def reviewer_methodological_node(state: ReviewState) -> ReviewState:
    experiment_id = state["experiment_id"]

    start = time.perf_counter()
    logger.info(
        "reviewer_methodological_started",
        experiment_id=experiment_id,
    )

    reviewer = ReviewerMethodological(llm_selection=state.get("llm_selection"))
    result = await reviewer.review(
        state["sections"],
        experiment_id=experiment_id,
        feedback=state.get("feedback"),
        previous_reviews=state.get("previous_reviews"),
        review_parameters=state.get("review_parameters"),
    )

    end = time.perf_counter()
    logger.info(
        "reviewer_methodological_completed",
        experiment_id=experiment_id,
        duration=end - start,
    )

    return {
        "review_methodological": {
            "agent": "reviewer_methodological",
            "output": result
        }
    }

async def reviewer_journal_editor_node(state: ReviewState) -> ReviewState:
    experiment_id = state["experiment_id"]

    start = time.perf_counter()
    logger.info(
        "reviewer_journal_editor_started",
        experiment_id=experiment_id,
    )

    reviewer = ReviewerJournalEditor(llm_selection=state.get("llm_selection"))
    result = await reviewer.review(
        state["sections"],
        experiment_id=experiment_id,
        feedback=state.get("feedback"),
        previous_reviews=state.get("previous_reviews"),
        review_parameters=state.get("review_parameters"),
    )

    end = time.perf_counter()
    logger.info(
        "reviewer_journal_editor_completed",
        experiment_id=experiment_id,
        duration=end - start,
    )

    return {
        "review_journal_editor": {
            "agent": "reviewer_journal_editor",
            "output": result
        }
    }