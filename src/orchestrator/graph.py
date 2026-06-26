import pandas as pd
from src.agents.synthetizer import Synthesizer
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
import operator
import asyncio
from typing import TypedDict,  Annotated, Sequence, Dict, Any, Optional, List
from langgraph.types import Send
import time
from config.global_config import CONFIG
from config.settings import settings
from config.llm_selection import ResolvedLlmSelection
import uuid

from collections import defaultdict

from src.agents.graph_query_agent import GraphQueryAgent
from src.agents.hybrid_query_agent import HybridQueryAgent
from utils.load_csv_from_blob import read_csv_safely
from apps.services.citation_service import get_citation_service
from operator import or_, add
from utils.citation_utils import remap_inline_citations, extract_results, build_paper_index
import structlog
logger = structlog.get_logger(__name__)
dispatched_steps = defaultdict(list)


def clearable_accumulator(existing: list, new: list | str):
    if new == "CLEAR":
        return []
    if existing is None:
        return new
    return existing + new


class ResearchState(TypedDict):

    hypothesis: str
    scientist_feedback: Optional[str]
    previous_context: Optional[dict]
    llm_selection: Optional[ResolvedLlmSelection]

    background: Optional[str]
    experiment_id: str

    step_results: Annotated[Dict[str, Any], operator.or_]

    uploaded_datasets: list[dict]  
    dataset_schemas: list[dict]

    objective: str

    execution_flags: dict
 
    methodology_checks: list[str]
    analysis_steps: list[dict]
    validation_criteria: list[str]

    retrieved_literature: list[dict]
    citations: list[dict]
    citation_registry: Annotated[Dict[str, int], or_]
    citation_counter: Annotated[int, add]

    python_code: Optional[str]
    code_stdout: Optional[str]
    code_stderr: Optional[str]
    generated_artifacts: list[dict]
    code_attempts: int

    last_error_summary: Optional[str]

    synth_sections: list[dict]
    contradictions: list[dict]
    failure_modes: list[str]
    confidence_score: float


    revision_rounds: int
    critic_feedback: dict
    epistemic_status: str

    extracted_samples: Annotated[list[dict], clearable_accumulator]
    extracted_claims: List[dict]

    hallucination_samples: Annotated[list[dict], clearable_accumulator]
    hallucination_report: dict

    execution_logs: Annotated[list[str], add]
    messages: Annotated[Sequence[BaseMessage], add]


def route_after_planner(state: ResearchState):
    flags = state["execution_flags"]
    has_data = bool(state.get("uploaded_datasets"))

    retrieval = flags["requires_retrieval"]
    stats = flags["allow_python_execution"] and has_data

    if retrieval and stats:
        return "both"
    if retrieval:
        return "retrieval_only"
    if stats:
        return "statistics_only"
    return "neither"


def route_after_critic(state: ResearchState):

    sends = []
    critic = state["critic_feedback"]
    rounds = state.get("revision_rounds", 0)
    updated_results = dict(state.get("step_results", {}))
    hallucination = state.get("hallucination_report", {})
    
    if rounds >= 3:
        return "finalize"

    if critic.get("revise_retrieval"):
        for step in state["analysis_steps"]:
            if step["agent"] == "retrieval":
                updated_results.pop(step["step_id"], None)
                payload = build_payload(step, state)
                sends.append(Send("knowledge_retriever", payload))

    if critic.get("revise_statistics"):
        for step in state["analysis_steps"]:
            if step["agent"] == "statistics":
                updated_results.pop(step["step_id"], None)
                payload = build_payload(step, state)
                sends.append(Send("statistical_executor", payload))

    elif critic.get("revise_planner"):
        return "hypothesis_planner"

    elif critic.get("revise_synthesis") or hallucination.get("needs_revision"):
        return "synthesizer"

    elif not critic.get("needs_revision") and not hallucination.get("needs_revision"):
        return "finalize"
    
    if sends:
        return sends
    return "synthesizer"


def build_payload(step, state):
    raw_deps = step.get("depends_on", [])
    deps = raw_deps if isinstance(raw_deps, list) else []

    dependency_results = {
        dep: state["step_results"][dep]
        for dep in deps if dep in state.get("step_results", {})
    }

    return {
        "step": step,
        "experiment_id": state["experiment_id"],
        "uploaded_datasets": state["uploaded_datasets"],
        "validation_criteria": state.get("validation_criteria", []),
        "dependency_results": dependency_results,
        "llm_selection": state.get("llm_selection"),
    }


def route_dispatch(state: ResearchState):
    sends = dispatch_steps(state)
    if sends:
        return sends
    return "wait_for_all_steps"


def dispatch_steps(state: ResearchState):
    sends = []
    experiment_id = state["experiment_id"]

    for step in state["analysis_steps"]:
        step_id = step["step_id"]

        if step_id in dispatched_steps[experiment_id]:
            logger.info(f"Skipping step {step_id} because it has already been dispatched.", step_id=step_id, experiment_id=experiment_id)
            continue


        raw_deps = step.get("depends_on", [])

        if isinstance(raw_deps, str):
            deps = [d.strip() for d in raw_deps.split(",")]
        elif isinstance(raw_deps, list):
            if len(raw_deps) == 1 and "," in raw_deps[0]:
                deps = [d.strip() for d in raw_deps[0].split(",")]
            else:
                deps = raw_deps
        else:
            deps = []
            
        if not set(deps).issubset(state.get("step_results", {}).keys()):   
            logger.info(f"Skipping step {step_id} because dependencies are not met.", step_id=step_id, experiment_id=experiment_id)
            continue

        dependency_results = {
            dep: state["step_results"][dep]
            for dep in deps
        }
        experiment_id = state["experiment_id"]
        if experiment_id is None:
            return []

        if experiment_id not in dispatched_steps:
            dispatched_steps[experiment_id] = []
            
        payload = {
            "step": step,
            "experiment_id": experiment_id,
            "uploaded_datasets": state["uploaded_datasets"],
            "validation_criteria": state.get("validation_criteria", []),
            "dependency_results": dependency_results,
            "llm_selection": state.get("llm_selection"),
        }

        if step["agent"] == "retrieval":
            sends.append(Send("knowledge_retriever", payload))
            dispatched_steps[experiment_id].append(step_id)
        elif step["agent"] == "statistics":
            sends.append(Send("statistical_executor", payload))
            dispatched_steps[experiment_id].append(step_id)

    return sends


def dispatch_extractors(state: ResearchState):
    """
    Triggers N parallel runs of the claim extractor.
    We pass only the exact keys required by the extractor to isolate the state.
    """
    payload = {
        "experiment_id": state.get("experiment_id", "unknown"),
        "hypothesis": state.get("hypothesis", ""),
        "synth_sections": state.get("synth_sections", []),
        "llm_selection": state.get("llm_selection"),
    }

    n_extractors = getattr(settings, "number_of_claim_extractors", 3)

    return [Send("claim_extractor", payload) for _ in range(n_extractors)]


def dispatch_hallucination_detectors(state: ResearchState):
    """
    Triggers N parallel runs of the hallucination detector.
    The payload is strictly mapped to the HallucinationDetector signatures.
    """
    lit, stats = extract_results(state.get("step_results", {}))
    literature = state.get("retrieved_literature", []) or lit
    
    final_check = state.get("revision_rounds", 0) >= 2
    claims = state.get("extracted_claims", [])

    payload = {
        "experiment_id": state.get("experiment_id", "unknown"),
        "claims": claims,
        "retrieved_literature": literature,
        "statistical_results": stats,
        "final_check": final_check,
        "llm_selection": state.get("llm_selection"),
    }
    
    n_detectors = getattr(settings, "number_of_hallucination_detectors", 3)
    
    if not claims:
        return [Send("hallucination_detector", payload)]
        
    return [Send("hallucination_detector", payload) for _ in range(n_detectors)]


def create_workflow(hybrid_agent):
    """Create the main LangGraph workflow for hypothesis testing."""
    
    workflow = StateGraph(ResearchState)

    workflow.add_node("hypothesis_planner", hypothesis_planner_node)
    workflow.add_node("dispatch_steps", dispatch_steps)
    async def knowledge_retriever_wrapper(state: ResearchState):
        return await knowledge_retriever_node(state, hybrid_agent)

    workflow.add_node("knowledge_retriever", knowledge_retriever_wrapper)   
    workflow.add_node("statistical_executor", statistical_executor_node)
    workflow.add_node("wait_for_all_steps", wait_for_all_steps)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("critique_agent", critique_agent_node)
    workflow.add_node("claim_extractor", claim_extractor_node)
    workflow.add_node("semantic_arbiter", semantic_arbiter_node)
    workflow.add_node("hallucination_detector", hallucination_detector_node)
    workflow.add_node("nli_consensus", nli_consensus_node)

    workflow.set_entry_point("hypothesis_planner")

    workflow.add_conditional_edges(
        "hypothesis_planner",
         route_dispatch,
        {
            "wait_for_all_steps": "wait_for_all_steps"
        }
    )

    workflow.add_conditional_edges(
        "knowledge_retriever",
        route_dispatch,
        {
            "wait_for_all_steps": "wait_for_all_steps"
        }
    )

    workflow.add_conditional_edges(
        "statistical_executor",
        route_dispatch,
        {
            "wait_for_all_steps": "wait_for_all_steps"
        }
    )

    workflow.add_conditional_edges(
        "wait_for_all_steps",
        barrier_router,
        {
            "synthesize": "synthesizer",
            "wait": "wait_for_all_steps"
        }
    )

    workflow.add_conditional_edges(
        "synthesizer",
        dispatch_extractors,
        ["claim_extractor"]
    )

    workflow.add_edge("claim_extractor", "semantic_arbiter")

    workflow.add_conditional_edges(
        "semantic_arbiter",
        dispatch_hallucination_detectors,
        ["hallucination_detector"]
    )

    workflow.add_edge("hallucination_detector", "nli_consensus")

    workflow.add_edge("nli_consensus", "critique_agent")

    workflow.add_conditional_edges(
        "critique_agent",
        route_after_critic,
        {
            "knowledge_retriever": "knowledge_retriever",
            "statistical_executor": "statistical_executor",
            "hypothesis_planner": "hypothesis_planner",
            "synthesizer": "synthesizer",
            "finalize": END,
        }
    )

    return workflow.compile()


def barrier_router(state: ResearchState):
    expected = {s["step_id"] for s in state["analysis_steps"]}
    completed = set(state.get("step_results", {}).keys())

    if expected.issubset(completed):
        return "synthesize"
    return "wait"


def hypothesis_planner_node(state: ResearchState) -> ResearchState:
    from src.agents.hypothesis_planner import HypothesisPlanner
    
    experiment_id = state.get("experiment_id") or str(uuid.uuid4())

    logger.info(
        "hypothesis_planner_node_started",
        experiment_id=experiment_id
    )
    planner = HypothesisPlanner(llm_selection=state.get("llm_selection"))
    logger.info(
        "hypothesis_planner_initialized",
        experiment_id=experiment_id
    )
    datasets = state.get("uploaded_datasets", [])
    profiled = []

    global dispatched_steps
    dispatched_steps[experiment_id] = []

    for dataset in datasets:
        path = dataset["path"]
        df = read_csv_safely(path, nrows=200)

        profile = {
            "id": dataset.get("id", "primary_dataset"),
            "path": path,
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numeric_columns": list(df.select_dtypes(include="number").columns),
            "categorical_columns": list(df.select_dtypes(include="object").columns),
        }

        profiled.append(profile)

    plan = planner.create_plan(
        hypothesis=state["hypothesis"],
        scientist_feedback=state.get("scientist_feedback"),
        previous_context=state.get("previous_context"), 
        dataset_schemas=profiled,
        critic_feedback=state.get("critic_feedback", {}),
        experiment_id=experiment_id
    )
    
    logger.info(
        "hypothesis_planner_node_completed",
            experiment_id=experiment_id,
            objective=plan.get("objective"),
            execution_flags=plan.get("execution_flags", {}),
            num_analysis_steps=len(plan.get("analysis_steps", [])), 
    )
    
    return {
        **state,
        "experiment_id": str(experiment_id),
        "dataset_schemas": profiled,
        "objective": plan["objective"],
        "execution_flags": plan["execution_flags"],
        "methodology_checks": plan["methodology_checks"],
        "analysis_steps": plan["analysis_steps"],
        "validation_criteria": plan["validation_criteria"],
        "execution_logs": state.get("execution_logs", []) + ["Planner complete"]
    }


async def knowledge_retriever_node(state: ResearchState, hybrid_agent: HybridQueryAgent) -> ResearchState:
    """Retrieve relevant datasets and literature."""
    experiment_id = state["experiment_id"]

    step = state["step"]
    start = time.perf_counter()
    logger.info(
        "knowledge_retriever_node_started",
        experiment_id=experiment_id,
        step_id=step.get("step_id"),
        step_description=step.get("description", ""),
        start_time=start
    )
    result = await hybrid_agent.run(
        question=step.get("description", ""),
        experiment_id=experiment_id,
        llm_selection=state.get("llm_selection"),
    )
    service = get_citation_service(experiment_id)

    sources = result["sources"]

    local_to_pid = {
        src["citation"]: src["paper_id"]
        for src in sources if src.get("paper_id")
    }

    pid_to_global = await service.batch_get_ids(local_to_pid.values())

    local_to_global = {
        local: pid_to_global[pid]
        for local, pid in local_to_pid.items()
    }
    
    normalized_answer = remap_inline_citations(
        result["answer"],
        local_to_global
    )
    normalized_sources = []

    for src in sources:
        pid = src.get("paper_id")
        if not pid:
            continue

        src["citation"] = pid_to_global[pid]
        normalized_sources.append(src)
    
    end = time.perf_counter()
    logger.info(
        "knowledge_retriever_node_completed",
        experiment_id=experiment_id,
        step_id=step.get("step_id"),
        step_description=step.get("description", ""),
        end_time=end,
        duration=end-start
    )

    return {
            "step_results": {
                step["step_id"]: {
                    "agent": "retrieval",
                    "step_text": step.get("description", ""),
                    "answer": normalized_answer,
                    "sources": normalized_sources
                }
            }

        }

async def statistical_executor_node(state: ResearchState) -> ResearchState:
    """Execute statistical analyses."""
    experiment_id = state["experiment_id"]

    from src.agents.statistical_executor import StatisticalExecutor
    step = state["step"]

    start = time.perf_counter()
    logger.info(
        "statistical_executor_node_started",
        experiment_id=experiment_id,
        step_id=state["step"]["step_id"],
        step_description=state["step"].get("description", ""),
    )

    executor = StatisticalExecutor(
        experiment_id=experiment_id,
        step_id=step["step_id"],
        llm_selection=state.get("llm_selection"),
    )
    result = await executor.execute(step, state)

    end = time.perf_counter()
    logger.info(
        "statistical_executor_node_completed",
        experiment_id=experiment_id,
        step_id=step.get("step_id"),
        step_description=step.get("description", ""),
        end_time=end,
        duration=end-start
    )

    return {
        "step_results": {
            step["step_id"]: {
                "agent": "statistics",
                "step_text": step.get("description", ""),
                "output": result
            }
        }
    }


def wait_for_all_steps(state: ResearchState):
    expected = {s["step_id"] for s in state["analysis_steps"]}
    completed = set(state.get("step_results", {}).keys())
    
    if expected.issubset(completed):
        return {}
    return None


async def synthesizer_node(state: ResearchState) -> ResearchState:
    """Synthesize final claims."""
    experiment_id=state["experiment_id"]
    logger.info(
        "synthesizer_node_started",
        experiment_id=experiment_id,
    )
    synthesizer = Synthesizer(
        experiment_id=experiment_id,
        llm_selection=state.get("llm_selection"),
    )
    logger.info(
        "synthesizer_initialized",
        experiment_id=experiment_id,
        )
    lit, stats = extract_results(state.get("step_results", {}))
    
    synthesis = await synthesizer.synthesize(
        hypothesis=state["hypothesis"],
        test_plan=state.get("analysis_steps", {}),
        results=stats,
        methodology=state.get("methodology_checks", []),
        literature=lit,
        critic_feedback=state.get("critic_feedback", None),
        extracted_claims=state.get("extracted_claims", []),
        hallucination_feedback=state.get("hallucination_report", None),
    )
    paper_index = build_paper_index(state.get("step_results", {}))
    def hydrate(citation):
        if not citation:
            return None

        if isinstance(citation, dict):
            paper_id = citation.get("paper_id")
        else:
            paper_id = citation 

        if not paper_id:
            return None

        source = paper_index.get(paper_id)

        if not source:
            return {
                "paper_id": paper_id,
                "error": "Paper not found in retrieved sources"
            }
        doi = source.get("doi")
        return {
            "citation_number": source.get("citation"),
            "paper_id": source.get("paper_id"), 
            "doi": doi,
            "doi_url": f"https://doi.org/{doi}" if doi else None,
            "title": source.get("title"),
            "authors": source.get("authors"),
            "abstract": source.get("abstract"),
            "paper_url": source.get("paper_url"),
        }    
    sections = synthesis.get("sections", {}) or {}

    sections_formatted = []

    for section in sections.values():
        if isinstance(section, dict):

            citation = hydrate(section.get("paper_id"))

            sections_formatted.append({
                "citation": citation,
                "detail": section.get("detail")
            })

    logger.info(
        "synthesizer_node_completed",
        experiment_id=experiment_id,
        contradictions=synthesis.get("contradictions", []),
        failure_modes=synthesis.get("failure_modes", []),
        confidence_score=synthesis.get("confidence_score", 0.0)
    )
    return {
        "synth_sections": sections_formatted,
        "retrieved_literature": lit,
        "contradictions": synthesis.get("contradictions", []),
        "failure_modes": synthesis.get("failure_modes", []),
        "confidence_score": synthesis.get("confidence_score", 0.0),
        "execution_logs": state.get("execution_logs", []) + ["Synthesizer done"],
    }


async def critique_agent_node(state: ResearchState) -> ResearchState:
    """Critique the synthesis output."""
    experiment_id = state["experiment_id"]
    logger.info(
        "critique_agent_node_started",
        experiment_id=experiment_id,
    )
    from src.agents.critique_agent import CriticAgent

    critic = CriticAgent(
        experiment_id=experiment_id,
        llm_selection=state.get("llm_selection"),
    )
    feedback = await critic.critique(
        hypothesis=state["hypothesis"],
        synthesis = {
            "extracted_claims": state.get("extracted_claims", []),
            "contradictions": state.get("contradictions", []),
            "failure_modes": state.get("failure_modes", []),
            "confidence_score": state.get("confidence_score", 0.0)
        },
        literature=state.get("retrieved_literature", []),
        has_uploaded_data=bool(state.get("uploaded_datasets")),
        final_critique=state.get("revision_rounds", 0) >= 2
    )
    
    updated_results = dict(state.get("step_results", {}))

    if feedback.get("revise_retrieval"):
        for step in state["analysis_steps"]:
            if step["agent"] == "retrieval":
                updated_results.pop(step["step_id"], None)
    
    logger.info(
        "critique_agent_node_completed",
        experiment_id=experiment_id,
        needs_revision=feedback.get("needs_revision"),
        revise_retrieval=feedback.get("revise_retrieval"),
        revise_statistics=feedback.get("revise_statistics"),
        revise_synthesis=feedback.get("revise_synthesis"),
        epistemic_status=feedback.get("epistemic_status", "inconclusive")
    )

    return {
        "critic_feedback": feedback,
        "step_results": updated_results,
        "confidence_score": 0.0,
        "epistemic_status": feedback.get("epistemic_status", "inconclusive"),
        "revision_rounds": state.get("revision_rounds", 0) + 1,
        "execution_logs": state.get("execution_logs", []) + ["Critic evaluation complete"]
    }


async def claim_extractor_node(state: dict) -> dict:
    """
    Parallel Worker Node.
    NOTE: 'state' here is the precise payload from Send(), not the full ResearchState.
    """
    from src.agents.claim_extractor import ClaimExtractor

    experiment_id = state.get("experiment_id", "unknown")
    hypothesis = state.get("hypothesis", "")
    synth_sections = state.get("synth_sections", [])

    extractor = ClaimExtractor(
        experiment_id=experiment_id,
        llm_selection=state.get("llm_selection"),
    )

    result = await extractor.extract_claims(
        hypothesis=hypothesis,
        synth_sections=synth_sections
    )

    return {"extracted_samples": [result]}


async def semantic_arbiter_node(state: ResearchState) -> dict:
    """
    
    """
    from src.agents.semantic_arbiter import SemanticArbiter

    experiment_id = state.get("experiment_id", "unknown")
    arbiter = SemanticArbiter(
        experiment_id=experiment_id,
        llm_selection=state.get("llm_selection"),
    )

    samples = state.get("extracted_samples", [])

    result = await arbiter.arbitrate(samples)

    return {
        "extracted_claims": result.get("extracted_claims", []),
        "hallucination_samples": "CLEAR",
        "execution_logs": state.get("execution_logs", []) + [
            f"Claim extraction complete (Consensus reached from {len(samples)} parallel runs)"
        ]
    }


async def hallucination_detector_node(state: dict) -> dict:
    """
    Parallel Worker Node.
    NOTE: 'state' here is the strictly mapped payload from Send().
    """
    from src.agents.hallucination_detector import HallucinationDetector

    claims = state.get("claims", [])
    
    if not claims:
         return {"hallucination_samples": [{"verdicts": []}]}

    detector = HallucinationDetector(
        experiment_id=state.get("experiment_id", "unknown"),
        llm_selection=state.get("llm_selection"),
    )

    result = await detector.detect(
        claims=claims,
        retrieved_literature=state.get("retrieved_literature", []),
        statistical_results=state.get("statistical_results"),
        final_check=state.get("final_check", False)
    )
    
    return {"hallucination_samples": [result]}


async def nli_consensus_node(state: ResearchState) -> dict:
    from src.agents.nli_consensus_arbiter import NLIConsensusArbiter


    experiment_id = state.get("experiment_id", "unknown")
    hallucination_samples = state.get("hallucination_samples", [])
    extracted_claims = state.get("extracted_claims", [])
    arbiter = NLIConsensusArbiter(experiment_id=experiment_id)

    consensus_report = arbiter.arbitrate(
        hallucination_samples=hallucination_samples,
        official_claims=extracted_claims
    )

    return {
        "hallucination_report": consensus_report,
        "execution_logs": state.get("execution_logs", []) + [
            f"Hallucination detection complete (Consensus from {len(hallucination_samples)} parallel runs)"
        ]
    }
