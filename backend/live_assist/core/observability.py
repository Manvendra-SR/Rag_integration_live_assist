"""Structured startup observability for AI provider configuration.

Logs the resolved provider, model, base URL and masked API key for all three
workload slots at application startup so ops can quickly verify configuration
without exposing secrets in plain text.

Usage (call once in application startup, e.g. main.py or api/__init__.py):

    from live_assist.core.observability import log_provider_config
    log_provider_config()

Output example:
    ═══ Live Assist Provider Configuration ═══
      [Slot 1] Live LLM (query enrichment + answer generation)
        Provider:   groq
        Model:      llama-3.3-70b-versatile
        Base URL:   https://api.groq.com/openai/v1  (auto)
        API Key:    gsk_...HHOs
        Struct Out: json_prompt
      [Slot 2] Context LLM (ingestion contextual prefixes)
        Provider:   groq
        Model:      llama-3.3-70b-versatile
        Base URL:   (default)
        API Key:    gsk_...HHOs
      [Slot 2b] Context VLM (figure / table description)
        Provider:   gemini
        Model:      gemini-2.0-flash
        Base URL:   (default)
        API Key:    AIza...yxg
      [Slot 3] Embeddings (semantic search + cache + ingestion)
        Provider:   local
        Model:      BAAI/bge-m3
        Base URL:   (local)
        API Key:    (not set)
    ═══════════════════════════════════════════
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_BORDER = "═" * 45


def _mask(key: str | None) -> str:
    """Return a partially masked API key safe for log output."""
    if not key:
        return "(not set)"
    key = key.strip()
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


def _e(var: str, default: str = "") -> str:
    return (os.environ.get(var, default) or "").strip()


def log_provider_config() -> None:
    """Log all three AI workload slot configurations with masked API keys."""
    try:
        from live_assist.core.config import get_settings
        s = get_settings()
    except Exception as exc:
        log.warning(f"[observability] Could not load settings: {exc}")
        return

    log.info(_BORDER)
    log.info("Live Assist — AI Provider Configuration")
    log.info(_BORDER)

    # ── Slot 1: Live LLM ─────────────────────────────────────────────────
    provider1 = s.resolve_live_llm_provider()
    model1 = s.resolve_live_llm_model()
    base_url1 = s.resolve_live_llm_base_url()
    api_key1 = s.resolve_live_llm_api_key()
    smode1 = s.resolve_live_llm_structured_output_mode()
    temp1 = s.resolve_live_llm_temperature()

    # Compute effective base_url for display
    if not base_url1 and provider1 == "groq":
        display_url1 = "https://api.groq.com/openai/v1  (auto)"
    elif not base_url1 and provider1 == "gemini":
        display_url1 = "https://generativelanguage.googleapis.com/v1beta/openai/  (auto)"
    else:
        display_url1 = base_url1 or "(default)"

    log.info("  [Slot 1] Live LLM  (query enrichment + answer generation)")
    log.info(f"    Provider:   {provider1}")
    log.info(f"    Model:      {model1 or '(not set)'}")
    log.info(f"    Base URL:   {display_url1}")
    log.info(f"    API Key:    {_mask(api_key1)}")
    log.info(f"    Temp:       {temp1}   Struct-out: {smode1}")

    # ── Slot 2: Context LLM ───────────────────────────────────────────────
    provider2 = s.resolve_context_llm_provider()
    model2 = s.resolve_context_llm_model()
    base_url2 = s.resolve_context_llm_base_url()
    api_key2 = s.resolve_context_llm_api_key()

    if not base_url2 and provider2 == "groq":
        display_url2 = "https://api.groq.com/openai/v1  (auto)"
    else:
        display_url2 = base_url2 or "(default)"

    log.info("  [Slot 2] Context LLM  (ingestion contextual prefixes)")
    log.info(f"    Provider:   {provider2}")
    log.info(f"    Model:      {model2 or '(not set)'}")
    log.info(f"    Base URL:   {display_url2}")
    log.info(f"    API Key:    {_mask(api_key2)}")

    # ── Slot 2b: VLM ─────────────────────────────────────────────────────
    vlm_provider = _e("PARSER_VLM_PROVIDER", "ollama")
    vlm_model = _e("PARSER_VLM_MODEL")
    vlm_base_url = s.parser_vlm_base_url
    vlm_api_key = s.parser_vlm_api_key or (s.gemini_api_key if vlm_provider == "gemini" else "")

    log.info("  [Slot 2b] Context VLM  (figure / table description)")
    log.info(f"    Provider:   {vlm_provider}")
    log.info(f"    Model:      {vlm_model or '(not set)'}")
    log.info(f"    Base URL:   {vlm_base_url or '(default)'}")
    log.info(f"    API Key:    {_mask(vlm_api_key)}")

    # ── Slot 3: Embeddings ────────────────────────────────────────────────
    emb_provider = _e("EMBEDDING_PROVIDER", "local")
    emb_model = _e("EMBEDDING_MODEL", _e("EMBED_LOCAL_MODEL", "BAAI/bge-m3"))
    emb_base_url = s.embedding_base_url
    emb_api_key = s.embedding_api_key

    log.info("  [Slot 3] Embeddings  (semantic search + cache + ingestion)")
    log.info(f"    Provider:   {emb_provider}")
    log.info(f"    Model:      {emb_model or '(not set)'}")
    log.info(f"    Base URL:   {emb_base_url or '(local)'}")
    log.info(f"    API Key:    {_mask(emb_api_key)}")

    log.info(_BORDER)
