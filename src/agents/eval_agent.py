import asyncio
import os
import instructor

import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from openai import AzureOpenAI
from pydantic import BaseModel, Field

from config.settings import settings
from src.agents.graph_query_agent import GraphQueryAgent
from src.agents.hybrid_query_agent import HybridQueryAgent
from src.prompts.eval_prompt import EVAL_SYSTEM_PROMPT
import instructor
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

class LlmEval(BaseModel):
    """
    LLM-based evaluation of candidate answer vs reference answer.

    factual_precision: factual precision rating (1..5)
    relevance: relevance rating (1..5)
    """

    factual_precision: float = Field(..., ge=1.0, le=5.0)
    relevance: float = Field(..., ge=1.0, le=5.0)
    rationale: str = Field(..., description="1-4 sentences explaining the scores.")


@dataclass(frozen=True)
class EvalRowResult:
    agent_answer: str
    factual_precision: float
    relevance: float


class EvalAgent:
    def __init__(self) -> None:

        logger.info(
            "eval_agent_initializing"
        )

        if not settings.azure_openai_api_key:
            logger.error(
                "eval_agent_missing_azure_openai_api_key"
            )
            raise ValueError("Missing required environment variable: AZURE_OPENAI_API_KEY")
        self.eval_client = instructor.patch(AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        ), mode=instructor.Mode.JSON)
        self.eval_model = os.environ.get("EVAL_JUDGE_MODEL",settings.azure_openai_deployment_eval)

        graph_agent = GraphQueryAgent(
            neo4j_uri=settings.neo4j_uri,
            neo4j_user=settings.neo4j_user,
            neo4j_password=settings.neo4j_password,
            azure_openai_api_key=settings.azure_openai_api_key,
            default_limit=settings.graph_query_limit
        )
        self.hybrid = HybridQueryAgent(graph_agent)

        logger.info(
            "eval_agent_initialized",
            eval_model=self.eval_model,
            graph_model=settings.graph_query_model,
        )

    def llm_judge(self, question: str, real_answer: str, agent_answer: str) -> LlmEval:
        logger.info(
            "eval_llm_judge_started",
            question_length=len(question or ""),
            reference_length=len(real_answer or ""),
            candidate_length=len(agent_answer or ""),
            model=self.eval_model,
        )
        user_prompt = (
            f"QUESTION:\n{question}\n\n"
            f"REFERENCE_ANSWER:\n{real_answer}\n\n"
            f"CANDIDATE_ANSWER:\n{agent_answer}\n"
        )

        try:
            verdict = self.eval_client.chat.completions.create(
                model=self.eval_model,
                response_model=LlmEval,
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )

            logger.info(
                "eval_llm_judge_completed",
                factual_precision=verdict.factual_precision,
                relevance=verdict.relevance,
            )
            return verdict

        except Exception as e:
            logger.exception(
                "eval_llm_judge_failed",
                error=str(e),
                model=self.eval_model,
            )
            raise

    async def evaluate(self, question: str, real_answer: str) -> EvalRowResult:
        logger.info(
            "eval_evaluate_started",
            question_length=len(question or ""),
            reference_length=len(real_answer or ""),
        )
        try:
            agent_answer = await self.hybrid.run(question)

            logger.info(
                "eval_hybrid_answer_generated",
                answer_length=len(agent_answer or ""),
            )
        except Exception as e:
            logger.exception(
                "eval_hybrid_generation_failed",
                error=str(e),
            )
            return EvalRowResult(
                agent_answer=f"(error) {e}",
                factual_precision=1.0,
                relevance=1.0,
            )
        
        try:
            verdict = self.llm_judge(question, real_answer, agent_answer)
            result = EvalRowResult(
                agent_answer=agent_answer,
                factual_precision=verdict.factual_precision,
                relevance=verdict.relevance,
            )

            logger.info(
                "eval_evaluate_completed",
                factual_precision=result.factual_precision,
                relevance=result.relevance,
            )
            return result
        except Exception as e:
            logger.warning(f"LLM evaluation failed, using default scores (factual_precision=1.0, relevance=1.0): {e}")
            logger.warning(
                "eval_llm_evaluation_failed_using_default_scores",
                error=str(e),
                factual_precision=1.0,
                relevance=1.0,
            )
            return EvalRowResult(
                agent_answer=agent_answer,
                factual_precision=1.0,
                relevance=1.0,
            )

def normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure stable dtypes when writing to CSV.
    """
    for col in ("questions", "real_answer", "agent_answer"):
        if col in df.columns:
            df[col] = df[col].astype("string")

    for col in ("factual_precision", "relevance"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")

    return df

def write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out_path)

async def main(argv: list[str] | None = None) -> None:
    in_path = Path("src/graph/evalset/Evalset.csv")
    out_path = Path("src/graph/evalset/Evalset_output.csv")

    df = pd.read_csv(in_path)
    df = normalize_dtypes(df)

    agent = EvalAgent()

    for idx in range(0, len(df)):
        question = df.at[idx, "questions"] if "questions" in df.columns else ""
        real_answer = df.at[idx, "real_answer"] if "real_answer" in df.columns else ""

        logger.info(f"Processing row {idx}")
        
        try:
            response = await agent.evaluate(str(question), str(real_answer))
        except Exception as e:
            logger.exception(f"Row {idx} failed: {e}")
            df.at[idx, "agent_answer"] = f"(error) {e}"
            df.at[idx, "factual_precision"] = 1.0
            df.at[idx, "relevance"] = 1.0
            write_csv(df, out_path)
            continue

        df.at[idx, "agent_answer"] = response.agent_answer
        df.at[idx, "factual_precision"] = response.factual_precision
        df.at[idx, "relevance"] = response.relevance

        write_csv(df, out_path)
        logger.info(f"Row {idx} completed: factual_precision={response.factual_precision:.3f}, relevance={response.relevance:.3f}")
        
    logger.info(
    "eval_main_completed",
    output_path=str(out_path),
    )
