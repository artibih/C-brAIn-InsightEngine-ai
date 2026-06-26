from typing import AsyncIterator, Dict, Any
import uuid
from apps.services.workflow_registry import workflow
from config.llm_selection import ResolvedLlmSelection


async def stream_hypothesis_graph(
    hypothesis: str,
    dataset_paths: list[str] = None,
    scientist_feedback: str = None,
    previous_context: dict = None,
    llm_selection: ResolvedLlmSelection | None = None,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Stream frontend-safe agent events.
    """
    uploaded_datasets = []
    
    if dataset_paths:
        for dataset_path in dataset_paths:
            uploaded_datasets.append({
                "id": uuid.uuid4().hex,
                "path": dataset_path,
                "format": "csv",
                "description": "User uploaded dataset",
            })

    initial_state = {
        "hypothesis": hypothesis,
        "uploaded_datasets": uploaded_datasets,
        "execution_logs": [],
        "messages": [],
        "iteration_count": 0,
        "critique_issues": [],
        "scientist_feedback": scientist_feedback,
        "previous_context": previous_context,
        "llm_selection": llm_selection,
        "extracted_samples": [],
        "hallucination_samples": []
    }

    async for event in workflow.astream(initial_state):
        for node_name, node_state in event.items():
            if node_name == "hypothesis_planner":
                yield project_planner(node_state)
            elif node_name == "knowledge_retriever":
                yield project_retriever(node_state)
            elif node_name == "statistical_executor":
                yield project_statistics(node_state)
            elif node_name == "critique_agent":
                yield project_critic(node_state)
            elif node_name == "synthesizer":
                yield project_synthesizer(node_state)
            elif node_name == "semantic_arbiter":
                yield project_claim_extractor(node_state)
            elif node_name == "nli_consensus":
                yield project_hallucination_detector(node_state)
          

async def run_until_planner(
    hypothesis: str,
    dataset_path: str = ""
) -> Dict[str, Any]:
    """
    Run graph until hypothesis_planner completes
    and return its frontend event.
    """
    async for event in stream_hypothesis_graph(hypothesis, [dataset_path] if dataset_path else None):
        if (
            event.get("type") == "agent_update"
            and event.get("agent") == "hypothesis_planner"
        ):
            return event


def project_planner(state: dict) -> dict:
    return {
        "type": "agent_update",
        "agent": "hypothesis_planner",
        "experiment_id": state.get("experiment_id"),
        "objective": state.get("objective"),
        "analysis_steps": [
            {
                "step_id": s["step_id"],
                "rationale": s["rationale"],
                "agent": s["agent"],
                "task": s["task"]
            }
            for s in state.get("analysis_steps", [])
        ],
        "datasets": [
            {
                "id": d["id"]
            }
            for d in state.get("dataset_schemas", [])
        ]
    }

def project_retriever(state: dict) -> dict:
    step_results = state.get("step_results", {})

    if not step_results:
        return {
            "type": "agent_update",
            "agent": "knowledge_retriever",
            "step_id": None,
            "retrieved_knowledge": []
        }

    step_id, result = next(iter(step_results.items()))

    return {
        "type": "agent_update",
        "agent": "knowledge_retriever",
        "step_id": step_id,
        "step_text": result.get("step_text", ""),
        "retrieved_knowledge": result.get("answer"),
        "sources": result.get("sources")
    }

def project_statistics(state: dict) -> dict:
    step_results = state.get("step_results", {})

    if not step_results:
        return {
            "type": "agent_update",
            "agent": "statistical_executor",
            "step_id": None,
            "statistics": {}
        }

    step_id, result = next(iter(step_results.items()))

    return {
        "type": "agent_update",
        "agent": "statistical_executor",
        "step_id": step_id,
        "step_text": result.get("step_text", ""),
        "generated_artifacts": result["output"].get("generated_artifacts", []),
        "structured_results": result["output"].get("structured_results", []),
        "test_description": result["output"].get("test_description", ""),
        "group_labels": result["output"].get("group_labels", []),
        "significance_threshold": result["output"].get("significance_threshold", 0.05),
        "n_compounds_tested": result["output"].get("n_compounds_tested", 0),
        "n_significant": result["output"].get("n_significant", 0)
    }

def project_critic(state: dict) -> dict:
    feedback = state.get("critic_feedback", {})

    return {
        "type": "agent_update",
        "agent": "critique_agent",
        "needs_revision": feedback.get("needs_revision", False),
        "priority_agent": feedback.get("priority_agent"),
        "issues": feedback.get("issues", []),
        "revision_instructions": feedback.get("revision_instructions", []),
        "strengths": feedback.get("strengths", []),
        "validation_summary": feedback.get("validation_summary", ""),
        "revise_flags": {
            "planner": feedback.get("revise_planner", False),
            "retrieval": feedback.get("revise retrieval", False),
            "statistics": feedback.get("revise statistics", False),
            "synthesis": feedback.get("revise_synthesis", False),
        }
    }

def project_synthesizer(state: dict) -> dict:
    synth_sections = state.get("synth_sections")
  
    return {
        "type": "agent_update",
        "agent": "synthesizer",
        "findings": synth_sections 
    }


def project_claim_extractor(state: dict) -> dict:
    claims = state.get("extracted_claims", [])

    return {
        "type": "agent_update",
        "agent": "claim_extractor",
        "total_claims": len(claims),
    }


def project_hallucination_detector(state: dict) -> dict:
    report = state.get("hallucination_report", {})
    summary = report.get("summary", {})

    return {
        "type": "agent_update",
        "agent": "hallucination_detector",
        "summary": {
            "total_claims": summary.get("total_claims", 0),
            "entailed": summary.get("entailed", 0),
            "contradicted": summary.get("contradicted", 0),
            "neutral": summary.get("neutral", 0),
            "hallucination_risk_score": summary.get("hallucination_risk_score", 0.0),
        },
        "verdicts": [
            {
                "claim_id": v.get("claim_id"),
                "claim_text": v.get("claim_text"),
                "verdict": v.get("verdict"),
                "confidence_score": v.get("confidence_score", 0.0),
                "matched_evidence": v.get("matched_evidence"),
                "reasoning": v.get("reasoning"),

                "justification": v.get("reasoning"), 
            }
            for v in report.get("verdicts", [])
        ],
    }
