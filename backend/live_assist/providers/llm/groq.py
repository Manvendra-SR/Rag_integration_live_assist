from __future__ import annotations

import json
from typing import Any, Optional, Type

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class GroqLLM:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.provider = str(config.get("LLM_PROVIDER", "groq") or "groq").lower()
        self.model_name = config.get("LLM_MODEL", "openai/gpt-oss-20b")
        self.temperature = config.get("TEMPERATURE", 0)
        self.structured_output_mode = str(
            config.get("LLM_STRUCTURED_OUTPUT_MODE", "json_prompt") or "json_prompt"
        ).lower()

        api_key, base_url = self._provider_connection(config)

        client_kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "api_key": api_key,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self.llm = ChatOpenAI(**client_kwargs)

    def _provider_connection(self, config: dict[str, Any]) -> tuple[str, str | None]:
        if self.provider == "groq":
            api_key = config.get("GROQ_API_KEY", "")
            if not api_key:
                raise ValueError("GROQ_API_KEY is required for the Groq LLM provider")
            return api_key, "https://api.groq.com/openai/v1"

        if self.provider == "openai":
            api_key = config.get("OPENAI_API_KEY", "") or config.get("LLM_API_KEY", "")
            if not api_key:
                raise ValueError("OPENAI_API_KEY or LLM_API_KEY is required for the OpenAI LLM provider")
            return api_key, None

        if self.provider == "openai_compatible":
            api_key = config.get("LLM_API_KEY", "")
            base_url = config.get("LLM_BASE_URL", "")
            if not api_key:
                raise ValueError("LLM_API_KEY is required for openai_compatible LLM provider")
            if not base_url:
                raise ValueError("LLM_BASE_URL is required for openai_compatible LLM provider")
            return api_key, base_url

        raise ValueError(
            "Unsupported LLM_PROVIDER. Use groq, openai, or openai_compatible."
        )

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
                "Unsupported LLM_STRUCTURED_OUTPUT_MODE. Use json_prompt, native, or native_fallback."
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
