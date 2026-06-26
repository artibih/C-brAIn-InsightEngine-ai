import os

import structlog
from fastapi import HTTPException
from pydantic import BaseModel

from config.llm_catalog import CATALOG, MODE_DEFAULTS, ConversationMode, LlmModelOption
from config.settings import settings

logger = structlog.get_logger(__name__)


class LlmSelectionRequest(BaseModel):
    provider: str | None = None
    model_key: str | None = None


class ResolvedLlmSelection(BaseModel):
    mode: str
    model_key: str
    provider: str
    option: LlmModelOption


def _get_env_value(name: str | None) -> str | None:
    if not name:
        return None
    return os.getenv(name) or getattr(settings, name.lower(), None)


def _validate_enabled_option(
    *,
    mode: str,
    model_key: str,
    provider: str,
    option: LlmModelOption,
) -> None:
    missing = [
        env_name
        for env_name in (option.endpoint_env, option.api_key_env)
        if env_name and not _get_env_value(env_name)
    ]
    if missing:
        logger.warning(
            "llm_selection_missing_configuration",
            mode=mode,
            provider=provider,
            model_key=model_key,
            runtime=option.runtime.value,
            endpoint_env=option.endpoint_env,
            api_key_env=option.api_key_env,
            missing_env=missing,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"LLM model '{model_key}' ({provider}) is missing required "
                f"configuration: {', '.join(missing)}"
            ),
        )


def resolve_llm_selection(
    mode: str,
    requested: LlmSelectionRequest | None,
) -> ResolvedLlmSelection:
    try:
        normalized_mode = ConversationMode(mode).value
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unsupported conversation mode: {mode}") from exc

    model_key = requested.model_key if requested and requested.model_key else MODE_DEFAULTS[normalized_mode]
    option = CATALOG.get(model_key)
    if option is None:
        raise HTTPException(status_code=400, detail=f"Unsupported LLM model_key: {model_key}")

    if requested and requested.provider and requested.provider != option.provider:
        raise HTTPException(status_code=400, detail="LLM provider does not match model_key")

    if not option.enabled:
        raise HTTPException(status_code=400, detail=f"LLM model is disabled: {model_key}")

    logger.info(
        "llm_selection_candidate_resolved",
        mode=normalized_mode,
        requested_provider=requested.provider if requested else None,
        requested_model_key=requested.model_key if requested else None,
        provider=option.provider,
        model_key=model_key,
        display_name=option.display_name,
        runtime=option.runtime.value,
        endpoint_env=option.endpoint_env,
        api_key_env=option.api_key_env,
    )

    _validate_enabled_option(
        mode=normalized_mode,
        model_key=model_key,
        provider=option.provider,
        option=option,
    )

    resolved = ResolvedLlmSelection(
        mode=normalized_mode,
        model_key=model_key,
        provider=option.provider,
        option=option,
    )
    logger.info(
        "llm_selection_resolved",
        mode=resolved.mode,
        provider=resolved.provider,
        model_key=resolved.model_key,
        runtime=resolved.option.runtime.value,
    )
    return resolved
