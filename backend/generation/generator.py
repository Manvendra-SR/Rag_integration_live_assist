"""Strict JSON answer generation with cloud-to-local fallback."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "chunk_id": {"type": "string"},
                    "doc_id": {"type": "string"},
                    "source_filename": {"type": "string"},
                    "page": {"type": "integer"},
                    "chunk_index": {"type": "integer"},
                    "section": {"type": "string"},
                    "excerpt": {"type": "string"},
                },
                "required": [
                    "chunk_id", "doc_id", "source_filename", "page",
                    "chunk_index", "section", "excerpt",
                ],
            },
        },
    },
    "required": ["answer", "confidence", "citations"],
}


@dataclass
class GenerationResult:
    provider: str
    model: str
    answer_payload: dict[str, Any]
    raw_text: str
    fallback_reason: str | None = None


def _extract_json_text(raw_text: str) -> str:
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end >= start:
        cleaned = cleaned[start:end + 1]
    return cleaned.strip()


def _parse_answer_payload(raw_text: str) -> dict[str, Any]:
    payload = json.loads(_extract_json_text(raw_text))
    if not isinstance(payload, dict):
        raise ValueError("Model output is not a JSON object")
    if "answer" not in payload:
        raise ValueError("Model output missing answer")
    payload.setdefault("confidence", 0.0)
    payload.setdefault("citations", [])
    return payload


def _groq_available() -> bool:
    return bool(
        (os.getenv("GENERATION_PROVIDER", "groq").lower() == "groq")
        and os.getenv("GENERATION_MODEL")
        and os.getenv("GROQ_API_KEY")
    )


def _groq_base_url() -> str:
    base = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    if base.endswith("/chat/completions"):
        return base.removesuffix("/chat/completions")
    return base


def _generate_groq(system_prompt: str, user_prompt: str) -> GenerationResult:
    import requests
    model = os.getenv("GENERATION_MODEL", "llama-3.3-70b-versatile")
    url = f"{_groq_base_url()}/chat/completions"
    payload = {
        "model": model,
        "temperature": float(os.getenv("GENERATION_TEMPERATURE", "0.0")),
        "max_tokens": int(os.getenv("GENERATION_MAX_TOKENS", "900")),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    resp = requests.post(
        url,
        json=payload,
        headers={
            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
            "Content-Type": "application/json",
        },
        timeout=int(os.getenv("GENERATION_TIMEOUT_SECONDS", "180")),
    )
    resp.raise_for_status()
    data = resp.json()
    raw = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "{}"
    return GenerationResult("groq", model, _parse_answer_payload(raw), raw)


def _generate_ollama(system_prompt: str, user_prompt: str,
                     fallback_reason: str | None = None) -> GenerationResult:
    import requests
    model = os.getenv("LLM_LOCAL_MODEL", os.getenv("OLLAMA_MODEL", "llama3.1"))
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    resp = requests.post(
        f"{base}/api/chat",
        json={
            "model": model,
            "stream": False,
            "format": ANSWER_SCHEMA,
            "options": {
                "temperature": float(os.getenv("GENERATION_TEMPERATURE", "0.0")),
                "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "900")),
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "240")),
    )
    resp.raise_for_status()
    raw = (resp.json().get("message") or {}).get("content") or "{}"
    return GenerationResult("ollama", model, _parse_answer_payload(raw), raw,
                            fallback_reason=fallback_reason)


def generate_answer(system_prompt: str, user_prompt: str) -> GenerationResult:
    if _groq_available():
        try:
            return _generate_groq(system_prompt, user_prompt)
        except Exception as exc:
            return _generate_ollama(system_prompt, user_prompt,
                                    fallback_reason=f"groq failed: {type(exc).__name__}: {exc}")
    return _generate_ollama(system_prompt, user_prompt,
                            fallback_reason="groq skipped: GENERATION_PROVIDER/model/key not all present")
