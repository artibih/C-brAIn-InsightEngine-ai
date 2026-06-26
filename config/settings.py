from pydantic_settings import BaseSettings
from typing import Optional
from pydantic import ConfigDict

class Settings(BaseSettings):
    """Application configuration settings."""

    # =========================
    # LLM Configuration
    # =========================
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None

    default_llm_provider: str = "openai"
    default_model: str = "gpt-4.1"

    # =========================
    # Code Execution Settings
    # =========================
    max_code_attempts: int = 4
    # artifact_storage_path: str = "./artifacts"
    AZURE_STORAGE_CONNECTION_STRING: Optional[str] = None
    artifact_storage_path: str = "/tmp/artifacts"  # Local fallback for artifacts
    # =========================
    # Weaviate (Local or Remote)
    # =========================
    weaviate_host: str = "weaviate-local"
    weaviate_port: int = 8080
    weaviate_scheme: str = "http"   # http | https

    weaviate_grpc_host: Optional[str] = None
    weaviate_grpc_port: Optional[int] = 50051
    weaviate_grpc_secure: bool = False

    weaviate_api_key: Optional[str] = None
    weaviate_timeout: int = 60

    @property
    def weaviate_url(self) -> str:
        """Derived HTTP URL (read-only)."""
        return f"{self.weaviate_scheme}://{self.weaviate_host}:{self.weaviate_port}"

    # =========================
    # Neo4j
    # =========================
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # =========================
    # API
    # =========================
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    api_workers: int = 4

    # =========================
    # Metrics
    # =========================
    max_response_time_simple: int = 5
    max_response_time_complex: int = 3600
    citation_hallucination_threshold: float = 1e-8
    citation_irrelevance_threshold: float = 1e-3

    # =========================
    # Code Execution
    # =========================
    code_execution_timeout: int = 300
    max_code_execution_memory: int = 2048

    # =========================
    # Logging & Env
    # =========================
    log_level: str = "INFO"
    log_format: str = "json"
    log_string_format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    environment: str = "development"

    # =========================
    # Weaviate
    # =========================
    # weaviate_url: str = "http://localhost:8080"
    weaviate_class: str = "Document"

    # =========================
    # Azure OpenAI (optional if using standard OpenAI for RAG via OPENAI_API_KEY)
    # =========================
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_api_version: Optional[str] = "2024-10-21"
    azure_openai_embedding_deployment: Optional[str] = None
    azure_openai_chat_deployment: Optional[str] = None
    azure_storage_account: Optional[str] = None
    azure_storage_key: Optional[str] = None
    azure_blob_url: Optional[str] = None
    azure_blob_container: Optional[str] = None
    azure_paper_json: Optional[str] = None

    azure_openai_deployment_online: str = "llm-online"
    azure_openai_deployment_background: str = "llm-background"
    azure_openai_deployment_graph: str = "llm-graph"
    azure_openai_deployment_eval: str = "llm-eval"
    azure_blob_url: Optional[str] = None
    azure_blob_container: Optional[str] = None
    azure_paper_container: Optional[str] = None
    azure_blob_url_paper: Optional[str] = None
    azure_blob_connection_string: Optional[str] = None
    azure_paper_container: Optional[str] = None

    # =========================
    # Hallucination Detection
    # =========================
    number_of_claim_extractors: Optional[int] = 3
    number_of_hallucination_detectors: Optional[int] = 3

    # =========================
    # Hybrid / Graph-RAG agent
    # =========================
    rag_base_url: str = "http://localhost:8000"
    hybrid_answer_model: Optional[str] = "gpt-4o-mini"
    hybrid_rag_top_k: Optional[int] = 5
    graph_query_limit: int = 25
    azure_openai_resource_name: Optional[str] = None  # e.g. "res-foundry-387"
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================
    # AZURE SQL
    # =========================
    db_server: Optional[str] = None
    db_name: str = "checkpoint"
    db_user: Optional[str] = None
    db_password: Optional[str] = None
    db_odbc_driver: str = "ODBC Driver 18 for SQL Server"
    


    def use_openai_for_rag(self) -> bool:
        """True when RAG should use standard OpenAI (OPENAI_API_KEY) instead of Azure."""
        return bool(self.openai_api_key and not self.azure_openai_api_key)


settings = Settings()
