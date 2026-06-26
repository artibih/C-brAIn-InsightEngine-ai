import re

from src.pipelines.rag_pipeline import BosnaRagPipeline

from config.llm_config import ChatCompletionGateway
from config.llm_selection import ResolvedLlmSelection
from config.settings import settings
from src.agents.graph_query_agent import GraphQueryAgent
from src.prompts.hybrid_query_prompt import build_hybrid_system_prompt
import structlog

logger = structlog.get_logger(__name__)
from utils.citation_utils import group_chunks_by_paper
from utils.response_postprocessing import postprocess_answer

class HybridQueryAgent:
    """
    Answers a question by combining:
    - Graph query (GraphQueryAgent -> Markdown table)
    - RAG retrieval (Question + table -> vector search -> chunks)
    - LLM synthesis (role + schema + table + chunks -> answer)
    """

    def __init__(
        self,
        graph_agent: GraphQueryAgent,
        *,
        rag_base_url: str | None = None,
    ) -> None:
        """
        Args:
            graph_agent: Existing GraphQueryAgent (used for Cypher -> Markdown table and schema).
            rag_base_url: Base URL of the RAG API (defaults to settings.rag_base_url).
        """
        self.graph_agent = graph_agent
        self.rag_base_url = (rag_base_url or settings.rag_base_url).rstrip("/")
        self.rag_top_k = settings.hybrid_rag_top_k
        
        self.chat_gateway = ChatCompletionGateway()
        self.rag = BosnaRagPipeline()
        logger.info(
            "hybrid_agent_initialized",
            deployment=settings.azure_openai_deployment_online,
            rag_base_url=self.rag_base_url,
            rag_top_k=self.rag_top_k,
        )
    
    async def run(
        self,
        question: str,
        experiment_id: str,
        llm_selection: ResolvedLlmSelection | None = None,
    ) -> dict:
        """
        End-to-end: graph table + RAG chunks -> LLM -> answer.
        """
        logger.info(
            "hybrid_run_started",
            experiment_id=experiment_id,
            question_length=len(question or ""),
        )

        if not question or not question.strip():
            logger.warning(
                "hybrid_run_empty_question",
                experiment_id=experiment_id,
            )
            return {
                "answer": "Please provide a non-empty question.", 
                "sources": []
            }

        try:
            markdown_table = await self.graph_agent.run(question, experiment_id=experiment_id)
            logger.info(
                "hybrid_graph_step_completed",
                experiment_id=experiment_id,
                table_length=len(markdown_table or ""),
                table_preview=(markdown_table or "")[:1000],
            )
        except Exception as e:
            logger.exception(
                "hybrid_graph_step_failed",
                experiment_id=experiment_id,
                error=str(e),
            )
            markdown_table = f"Graph step failed: {e}"

        rag_query = question.strip()  
        

        try:
            logger.info(
                "hybrid_rag_retrieval_started",
                experiment_id=experiment_id,
                rag_query_length=len(rag_query),
            )

            chunks = self.rag.retrieve_with_metadata(rag_query, experiment_id=experiment_id)

            logger.info(
                "hybrid_rag_retrieval_completed",
                experiment_id=experiment_id,
                chunk_count=len(chunks or []),
            )
        except Exception as e:
            logger.warning(
                "hybrid_rag_retrieval_failed",
                experiment_id=experiment_id,
                error=str(e),
            )
            chunks = []
        
        try:
            schema = await self.graph_agent.get_schema()
            logger.info(
                "hybrid_schema_loaded",
                experiment_id=experiment_id,
                schema_length=len(schema or ""),
            )
        except Exception as e:
            logger.exception(
                "hybrid_schema_load_failed",
                experiment_id=experiment_id,
                error=str(e),
            )
            schema = ""

        chunks = group_chunks_by_paper(chunks)

        answer = await self.answer_with_llm(
            question,
            schema,
            markdown_table,
            chunks,
            experiment_id=experiment_id,
            llm_selection=llm_selection,
        )
        sources = self.format_sources_with_links(chunks, experiment_id=experiment_id)
    
        answer, filtered_sources = postprocess_answer(answer, sources)
        logger.info(
            "hybrid_run_completed",
            experiment_id=experiment_id,
            answer_length=len(answer or ""),
            source_count=len(filtered_sources),
        )
        return {
            "answer": answer,
            "sources": filtered_sources
        }

    async def answer_with_llm(
        self,
        question: str,
        schema: str,
        markdown_table: str,
        chunks: list[dict],
        experiment_id: str,
        llm_selection: ResolvedLlmSelection | None = None,
    ) -> str:
        """Build prompt (role + schema + table + chunks) and call LLM; return answer text."""
        system_prompt = build_hybrid_system_prompt(schema, markdown_table, chunks)
        logger.info(
            "hybrid_final_answer_generation_started",
            experiment_id=experiment_id,
            model_key=llm_selection.model_key if llm_selection else settings.default_model,
            provider=llm_selection.provider if llm_selection else settings.default_llm_provider,
            chunks=len(chunks or []),
            table_length=len(markdown_table or ""),
            schema_length=len(schema or ""),
            system_prompt_length=len(system_prompt or ""),
        )

        try:
            answer = await self.chat_gateway.acomplete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=0,
                workload="online",
                llm_selection=llm_selection,
            )

            answer = answer.strip() or "(No answer generated.)"

            logger.info(
                "hybrid_final_answer_generation_completed",
                experiment_id=experiment_id,
                answer_length=len(answer or ""),
            )

            return answer

        except Exception as e:
            logger.exception(
                "hybrid_final_answer_generation_failed",
                experiment_id=experiment_id,
                error=str(e),
                model_key=llm_selection.model_key if llm_selection else settings.default_model,
            )
            raise
        

    def format_sources_with_links(self, chunks: list[dict], experiment_id: str) -> list[dict]:
        logger.info(
            "hybrid_format_sources_started",
            experiment_id=experiment_id,
            chunk_count=len(chunks or []),
        )
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            formatted.append({
                "citation": i,
                "doi": chunk.get("doi"),  
                "url": chunk.get("doi_url"), 
                "paper_id": chunk.get("paper_id"),     
                "content": chunk.get("content"),
                "title": chunk.get("title"),
                "authors": chunk.get("authors"), 
                "abstract":chunk.get("abstract"),
                "paper_url": chunk.get("paper_url")
            })
        logger.info(
            "hybrid_format_sources_completed",
            experiment_id=experiment_id,
            source_count=len(formatted),
        )
        return formatted
