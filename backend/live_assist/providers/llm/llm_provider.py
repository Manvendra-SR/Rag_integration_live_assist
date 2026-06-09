"""Live Cycle LLM — Slot 1: query enrichment + answer generation.

Reads provider configuration from the resolved Slot 1 vars:
    LIVE_LLM_PROVIDER   groq | openai | openai_compatible | gemini
    LIVE_LLM_MODEL      model name
    LIVE_LLM_BASE_URL   required for openai_compatible
    LIVE_LLM_API_KEY    API key (falls back to GROQ_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY)

When slot-specific vars are not set, falls back to the legacy
LLM_PROVIDER / LLM_MODEL / GROQ_API_KEY / LLM_API_KEY values so existing
.env files continue to work without any changes.

For openai_compatible (vLLM / RunPod):
    ChatOpenAI(model=LIVE_LLM_MODEL, api_key=LIVE_LLM_API_KEY or "dummy", base_url=LIVE_LLM_BASE_URL)

For gemini:
    Routed through Gemini's OpenAI-compatible endpoint
    (https://generativelanguage.googleapis.com/v1beta/openai/)
    so no new SDK dependency is required.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional, Type

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

log = logging.getLogger(__name__)

# Gemini's OpenAI-compatible endpoint (no extra SDK needed)
_GEMINI_OPENAI_COMPAT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Groq's OpenAI-compatible endpoint
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LiveCycleLLM:
    """Provider-agnostic LLM for live-cycle query enrichment and answer generation.

    Instantiate once at module level (graph.py) and reuse across all calls.
    All provider details are read from the resolved config dict produced by
    Settings.workflow_config() — callers never touch env vars directly.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        # Read from Slot 1 resolved keys (set by Settings.workflow_config)
        self.provider = str(
            config.get("LIVE_LLM_PROVIDER")
            or config.get("LLM_PROVIDER")
            or "groq"
        ).lower()
        self.model_name = (
            config.get("LIVE_LLM_MODEL")
            or config.get("LLM_MODEL")
            or ""
        )
        self.temperature = float(
            config.get("LIVE_LLM_TEMPERATURE", config.get("TEMPERATURE", 0))
            if config.get("LIVE_LLM_TEMPERATURE", -1) >= 0
            else config.get("TEMPERATURE", 0)
        )
        self.structured_output_mode = str(
            config.get("LIVE_LLM_STRUCTURED_OUTPUT_MODE")
            or config.get("LLM_STRUCTURED_OUTPUT_MODE")
            or "json_prompt"
        ).lower()

        api_key, base_url = self._resolve_connection()

        log.info(
            f"[LiveCycleLLM] provider={self.provider} model={self.model_name} "
            f"base_url={base_url or '(default)'} "
            f"structured_output_mode={self.structured_output_mode}"
        )

        client_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "api_key": api_key,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self.llm = ChatOpenAI(**client_kwargs)

    # ------------------------------------------------------------------
    # Provider resolution
    # ------------------------------------------------------------------

    def _resolve_connection(self) -> tuple[str, str | None]:
        """Return (api_key, base_url) for the configured provider.

        For openai_compatible: api_key defaults to "dummy" so vLLM/RunPod
        endpoints that require no authentication still work.
        For gemini: routes through the OpenAI-compatible endpoint so we
        don't need a separate langchain-google-genai dependency.
        """
        config = self.config

        if self.provider == "groq":
            api_key = (
                config.get("LIVE_LLM_API_KEY")
                or config.get("GROQ_API_KEY")
                or ""
            )
            if not api_key:
                raise ValueError(
                    "No API key found for Groq Live LLM. "
                    "Set LIVE_LLM_API_KEY or GROQ_API_KEY."
                )
            return api_key, _GROQ_BASE_URL

        if self.provider == "openai":
            api_key = (
                config.get("LIVE_LLM_API_KEY")
                or config.get("OPENAI_API_KEY")
                or config.get("LLM_API_KEY")
                or ""
            )
            if not api_key:
                raise ValueError(
                    "No API key found for OpenAI Live LLM. "
                    "Set LIVE_LLM_API_KEY or OPENAI_API_KEY."
                )
            return api_key, None  # OpenAI uses default base URL

        if self.provider == "openai_compatible":
            # vLLM / RunPod / any OpenAI-compatible endpoint
            api_key = (
                config.get("LIVE_LLM_API_KEY")
                or config.get("LLM_API_KEY")
                or "dummy"  # vLLM doesn't need a real key
            )
            base_url = (
                config.get("LIVE_LLM_BASE_URL")
                or config.get("LLM_BASE_URL")
                or ""
            )
            if not base_url:
                raise ValueError(
                    "LIVE_LLM_BASE_URL (or LLM_BASE_URL) is required for "
                    "openai_compatible Live LLM provider."
                )
            return api_key, base_url

        if self.provider == "gemini":
            api_key = (
                config.get("LIVE_LLM_API_KEY")
                or config.get("GEMINI_API_KEY")
                or ""
            )
            if not api_key:
                raise ValueError(
                    "No API key found for Gemini Live LLM. "
                    "Set LIVE_LLM_API_KEY or GEMINI_API_KEY."
                )
            # Route through Gemini's OpenAI-compatible endpoint
            return api_key, _GEMINI_OPENAI_COMPAT_BASE_URL

        raise ValueError(
            f"Unsupported LIVE_LLM_PROVIDER '{self.provider}'. "
            "Use: groq | openai | openai_compatible | gemini"
        )

    # ------------------------------------------------------------------
    # Inference helpers (unchanged from original GroqLLM)
    # ------------------------------------------------------------------

    def _build_messages(self, system_prompt: str, user_prompt: str) -> list:
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

    def invoke_model(
        self,
        system_prompt: str,
        user_prompt: str,
        variables: Optional[dict[str, Any]] = None,
    ) -> str:
        del variables
        response = self.llm.invoke(self._build_messages(system_prompt, user_prompt))
        return StrOutputParser().invoke(response)

    def _extract_json_object(self, text: str) -> str:
        cleaned = (text or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model response.")
        return cleaned[start : end + 1]

    def _invoke_structured_output_fallback(
        self,
        schema: Type[BaseModel],
        system_prompt: str,
        user_prompt: str,
    ) -> Any:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        fallback_messages = self._build_messages(
            system_prompt,
            (
                f"{user_prompt}\n\n"
                "Return ONLY a valid JSON object matching this schema exactly.\n"
                f"{schema_json}"
            ),
        )
        raw_response = StrOutputParser().invoke(self.llm.invoke(fallback_messages))
        json_payload = self._extract_json_object(raw_response)
        return schema.model_validate_json(json_payload)

    def invoke_model_with_structured_output(
        self,
        schema: Type[BaseModel],
        system_prompt: str,
        user_prompt: str,
        variables: Optional[dict[str, Any]] = None,
    ) -> Any:
        del variables
        if self.structured_output_mode == "json_prompt":
            try:
                return self._invoke_structured_output_fallback(
                    schema=schema,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            except Exception:
                if schema.__name__ == "QueryResponse" and "answer" in schema.model_fields:
                    raw_response = self.invoke_model(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    )
                    return schema(answer=raw_response.strip())
                raise

        structured_llm = self.llm.with_structured_output(schema)
        if self.structured_output_mode == "native":
            response = structured_llm.invoke(self._build_messages(system_prompt, user_prompt))
            if response is None:
                raise ValueError("Structured output returned None.")
            return response

        if self.structured_output_mode != "native_fallback":
            raise ValueError(
                "Unsupported LIVE_LLM_STRUCTURED_OUTPUT_MODE. "
                "Use: json_prompt | native | native_fallback"
            )

        try:
            response = structured_llm.invoke(self._build_messages(system_prompt, user_prompt))
            if response is None:
                raise ValueError("Structured output returned None.")
            return response
        except Exception:
            if schema.__name__ == "QueryResponse" and "answer" in schema.model_fields:
                raw_response = self.invoke_model(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                return schema(answer=raw_response.strip())
            return self._invoke_structured_output_fallback(
                schema=schema,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
