from src.agents.graph_query_agent import GraphQueryAgent
from src.agents.hybrid_query_agent import HybridQueryAgent
from src.orchestrator.graph import create_workflow
from src.orchestrator.reviewer_graph import create_review_workflow
from config.settings import settings

graph_agent = GraphQueryAgent(
    neo4j_uri=settings.neo4j_uri,
    neo4j_user=settings.neo4j_user,
    neo4j_password=settings.neo4j_password,
    azure_openai_api_key=settings.azure_openai_api_key,
    default_limit=settings.graph_query_limit,
)

hybrid_agent = HybridQueryAgent(graph_agent)

workflow = create_workflow(hybrid_agent=hybrid_agent)
review_workflow = create_review_workflow(hybrid_agent=hybrid_agent)
