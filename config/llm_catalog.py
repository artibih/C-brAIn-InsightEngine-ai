from enum import StrEnum

from pydantic import BaseModel, Field

from config.settings import settings


class ConversationMode(StrEnum):
    RAG = "rag"
    CHAT = "chat"
    REVIEWER = "reviewer"


class LlmRuntime(StrEnum):
    AZURE_OPENAI = "azure_openai"
    AZURE_FOUNDRY_OPENAI_V1 = "azure_foundry_openai_v1"


class LlmModelOption(BaseModel):
    model_key: str
    provider: str
    display_name: str
    runtime: LlmRuntime
    enabled: bool = False
    model_name: str | None = None
    deployment_by_workload: dict[str, str] = Field(default_factory=dict)
    endpoint_env: str | None = None
    api_key_env: str | None = None
    notes: str | None = None


CATALOG: dict[str, LlmModelOption] = {
    "azure-openai-gpt-4.1": LlmModelOption(
        model_key="azure-openai-gpt-4.1",
        provider="openai",
        display_name="OpenAI GPT-4.1",
        runtime=LlmRuntime.AZURE_OPENAI,
        enabled=True,
        model_name="gpt-4.1",
        deployment_by_workload={
            "online": settings.azure_openai_deployment_online,
            "background": settings.azure_openai_deployment_background,
            "graph": settings.azure_openai_deployment_graph,
            "eval": settings.azure_openai_deployment_eval,
        },
    ),
    "Mistral-Large-3": LlmModelOption(
        model_key="Mistral-Large-3",
        provider="mistral",
        display_name="Mistral Large 3",
        runtime=LlmRuntime.AZURE_FOUNDRY_OPENAI_V1,
        enabled=True,
        model_name="Mistral-Large-3",
        endpoint_env="AZURE_OPENAI_ENDPOINT",
        api_key_env="AZURE_OPENAI_API_KEY",
    ),
    "Llama-4-Maverick-17B-128E-Instruct-FP8": LlmModelOption(
        model_key="Llama-4-Maverick-17B-128E-Instruct-FP8",
        provider="llama",
        display_name="Llama 4 Maverick 17B 128E Instruct FP8",
        runtime=LlmRuntime.AZURE_FOUNDRY_OPENAI_V1,
        enabled=True,
        model_name="Llama-4-Maverick-17B-128E-Instruct-FP8",
        endpoint_env="AZURE_OPENAI_ENDPOINT",
        api_key_env="AZURE_OPENAI_API_KEY",
    ),
    "DeepSeek-V4-Pro": LlmModelOption(
        model_key="DeepSeek-V4-Pro",
        provider="deepseek",
        display_name="DeepSeek V4 Pro",
        runtime=LlmRuntime.AZURE_FOUNDRY_OPENAI_V1,
        enabled=True,
        model_name="DeepSeek-V4-Pro",
        endpoint_env="AZURE_OPENAI_ENDPOINT",
        api_key_env="AZURE_OPENAI_API_KEY",
    ),
}


MODE_DEFAULTS: dict[str, str] = {
    ConversationMode.RAG.value: "azure-openai-gpt-4.1",
    ConversationMode.CHAT.value: "azure-openai-gpt-4.1",
    ConversationMode.REVIEWER.value: "azure-openai-gpt-4.1",
}
