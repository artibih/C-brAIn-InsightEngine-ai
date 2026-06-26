import os
from typing import Any

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from config.llm_catalog import LlmRuntime
from config.llm_selection import ResolvedLlmSelection
from config.settings import settings

logger = structlog.get_logger(__name__)


def _resolve_azure_deployment(workload: str | None) -> str:
    workload = workload or "online"

    mapping = {
        "online": settings.azure_openai_deployment_online,
        "background": settings.azure_openai_deployment_background,
        "graph": settings.azure_openai_deployment_graph,
        "eval": settings.azure_openai_deployment_eval,
    }

    try:
        return mapping[workload]
    except KeyError:
        raise ValueError(f"Unsupported Azure workload: {workload}")


def _get_required_setting(name: str | None, *, label: str) -> str:
    if not name:
        raise ValueError(f"Missing {label} setting name")

    attr_name = name.lower()
    value = os.getenv(name) or getattr(settings, attr_name, None)
    if not value:
        raise ValueError(f"Missing required LLM configuration: {name}")
    return value


def _get_openai_v1_base_url(name: str | None) -> str:
    base_url = _get_required_setting(name, label="OpenAI-compatible base URL").rstrip("/")
    if base_url.endswith("/openai/v1"):
        return base_url
    return f"{base_url}/openai/v1"


def _resolve_selection_deployment(llm_selection: ResolvedLlmSelection, workload: str) -> str:
    deployment = llm_selection.option.deployment_by_workload.get(workload)
    if not deployment:
        raise ValueError(
            f"LLM model {llm_selection.model_key} does not support workload: {workload}"
        )
    return deployment


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0,
    workload: str = "online",
    llm_selection: ResolvedLlmSelection | None = None,
):
    """
    Get a LangChain chat model.

    When llm_selection is omitted, this preserves the legacy provider/model
    behavior and Azure deployment mapping.
    """
    if llm_selection is not None:
        option = llm_selection.option
        logger.info(
            "llm_factory_selection_requested",
            mode=llm_selection.mode,
            provider=llm_selection.provider,
            model_key=llm_selection.model_key,
            runtime=option.runtime.value,
            workload=workload,
        )

        if option.runtime == LlmRuntime.AZURE_OPENAI:
            deployment = _resolve_selection_deployment(llm_selection, workload)
            logger.info(
                "llm_factory_azure_openai_selected",
                mode=llm_selection.mode,
                model_key=llm_selection.model_key,
                workload=workload,
                deployment=deployment,
            )
            return AzureChatOpenAI(
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
                azure_deployment=deployment,
                temperature=temperature,
            )

        if option.runtime == LlmRuntime.AZURE_FOUNDRY_OPENAI_V1:
            logger.info(
                "llm_factory_openai_compatible_client_configured",
                mode=llm_selection.mode,
                provider=llm_selection.provider,
                model_key=llm_selection.model_key,
                model_name=option.model_name,
                runtime=option.runtime.value,
                workload=workload,
                base_url_env=option.endpoint_env,
                api_key_env=option.api_key_env,
            )
            return ChatOpenAI(
                api_key=_get_required_setting(option.api_key_env, label="OpenAI-compatible API key"),
                base_url=_get_openai_v1_base_url(option.endpoint_env),
                model=option.model_name,
                temperature=temperature,
            )

        raise ValueError(f"Unsupported LLM runtime: {option.runtime}")

    provider = provider or settings.default_llm_provider
    model = model or settings.default_model

    if provider == "openai":
        deployment = _resolve_azure_deployment(workload)

        return AzureChatOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            azure_deployment=deployment,
            temperature=temperature,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=model or "claude-sonnet-4-20250514",
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


class ChatCompletionGateway:
    """Provider-aware gateway for direct chat-completions call sites."""

    def _legacy_azure_client(self) -> AzureOpenAI:
        return AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )

    def _legacy_async_azure_client(self) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )

    def _openai_compatible_client(self, llm_selection: ResolvedLlmSelection) -> OpenAI:
        option = llm_selection.option
        logger.info(
            "chat_completion_gateway_openai_compatible_client_configured",
            mode=llm_selection.mode,
            provider=llm_selection.provider,
            model_key=llm_selection.model_key,
            model_name=option.model_name,
            runtime=option.runtime.value,
            base_url_env=option.endpoint_env,
            api_key_env=option.api_key_env,
        )
        return OpenAI(
            api_key=_get_required_setting(option.api_key_env, label="OpenAI-compatible API key"),
            base_url=_get_openai_v1_base_url(option.endpoint_env),
        )

    def _async_openai_compatible_client(self, llm_selection: ResolvedLlmSelection) -> AsyncOpenAI:
        option = llm_selection.option
        logger.info(
            "chat_completion_gateway_async_openai_compatible_client_configured",
            mode=llm_selection.mode,
            provider=llm_selection.provider,
            model_key=llm_selection.model_key,
            model_name=option.model_name,
            runtime=option.runtime.value,
            base_url_env=option.endpoint_env,
            api_key_env=option.api_key_env,
        )
        return AsyncOpenAI(
            api_key=_get_required_setting(option.api_key_env, label="OpenAI-compatible API key"),
            base_url=_get_openai_v1_base_url(option.endpoint_env),
        )

    def _model_for_selection(
        self,
        workload: str,
        llm_selection: ResolvedLlmSelection | None,
    ) -> str:
        if llm_selection is None:
            return _resolve_azure_deployment(workload)

        option = llm_selection.option
        if option.runtime == LlmRuntime.AZURE_OPENAI:
            return _resolve_selection_deployment(llm_selection, workload)

        if not option.model_name:
            raise ValueError(f"LLM model {llm_selection.model_key} is missing model_name")
        return option.model_name

    def complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        workload: str,
        llm_selection: ResolvedLlmSelection | None = None,
    ) -> str:
        model = self._model_for_selection(workload, llm_selection)
        runtime = llm_selection.option.runtime if llm_selection else LlmRuntime.AZURE_OPENAI

        logger.info(
            "chat_completion_gateway_complete_started",
            mode=llm_selection.mode if llm_selection else None,
            provider=llm_selection.provider if llm_selection else settings.default_llm_provider,
            model_key=llm_selection.model_key if llm_selection else settings.default_model,
            runtime=runtime.value,
            workload=workload,
            model=model,
        )

        if runtime == LlmRuntime.AZURE_OPENAI:
            client = self._legacy_azure_client()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False,
            )
            return response.choices[0].message.content or ""

        if runtime == LlmRuntime.AZURE_FOUNDRY_OPENAI_V1:
            client = self._openai_compatible_client(llm_selection)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False,
            )
            return response.choices[0].message.content or ""

        llm = get_llm(temperature=temperature, workload=workload, llm_selection=llm_selection)
        response = llm.invoke(messages)
        return getattr(response, "content", "") or ""

    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        workload: str,
        llm_selection: ResolvedLlmSelection | None = None,
    ) -> str:
        model = self._model_for_selection(workload, llm_selection)
        runtime = llm_selection.option.runtime if llm_selection else LlmRuntime.AZURE_OPENAI

        logger.info(
            "chat_completion_gateway_acomplete_started",
            mode=llm_selection.mode if llm_selection else None,
            provider=llm_selection.provider if llm_selection else settings.default_llm_provider,
            model_key=llm_selection.model_key if llm_selection else settings.default_model,
            runtime=runtime.value,
            workload=workload,
            model=model,
        )

        if runtime == LlmRuntime.AZURE_OPENAI:
            client = self._legacy_async_azure_client()
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""

        if runtime == LlmRuntime.AZURE_FOUNDRY_OPENAI_V1:
            client = self._async_openai_compatible_client(llm_selection)
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""

        llm = get_llm(temperature=temperature, workload=workload, llm_selection=llm_selection)
        response = await llm.ainvoke(messages)
        return getattr(response, "content", "") or ""