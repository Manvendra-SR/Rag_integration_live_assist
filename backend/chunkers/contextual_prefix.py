"""Contextual enrichment via local LLM (Ollama).

For each structural chunk:
  1. Look up its parent section's full text (concat all blocks under the same
     section_path[:1] from the parsed doc).
  2. Ask the local LLM (default llama3.1) to write 2-3 sentences situating the
     chunk inside its parent section.
  3. Prepend the contextual prefix to the chunk text.

Local LLM via Ollama HTTP API. Parallelized with ThreadPoolExecutor — Ollama
serializes inference on a single GPU but HTTP overhead is reduced and we batch
the python-side post-processing.
"""
from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

# Load .env for cloud provider keys
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

log = logging.getLogger(__name__)

# Provider selection: "ollama" (default, local) | "groq" | "gemini"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()

# Local Ollama
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

# Cloud Groq (OpenAI-compatible)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_RPM = int(os.environ.get("GROQ_RPM", "28"))  # 28 < 30 free tier cap, with safety margin

MAX_PREFIX_TOKENS = 200  # raised from 120 → room for multi-topic coverage
PARALLELISM = int(os.environ.get("ENRICH_PARALLELISM", "6"))
HTTP_TIMEOUT = 120.0

PROMPT_TEMPLATE = """You are helping prepare chunks of a document for retrieval.

Here is the SECTION the chunk belongs to:
<section>
{section_text}
</section>

Here is the specific CHUNK we want to situate:
<chunk>
{chunk_text}
</chunk>

Write 2-4 sentences that situate this chunk for retrieval purposes.

CRITICAL: A chunk often contains MULTIPLE distinct topics, sub-topics, disclaimers, or procedural instructions. Describe ALL of them — not just the most prominent one. If the chunk has a main topic AND a disclaimer AND a chatbot instruction AND a side note, mention every one of them.

Cover specifically:
- every distinct topic, sub-topic, or Q&A in the chunk
- how it relates to its parent section
- any key terms, entities, plan names, numbers, or product names a search query might use
- any disclaimers, caveats, instructions, restrictions, or procedural rules
- any side notes or peripheral content that someone might search for

Output ONLY the 2-4 sentences of context. No preamble, no explanation, no quotes."""


# ---------------------------------------------------------------------------
# Parent-section assembly
# ---------------------------------------------------------------------------


def build_section_lookup(parsed: dict[str, Any]) -> dict[str, str]:
    """Map first-level section_path → concatenated text of all blocks in it.

    Falls back to whole-doc markdown if a chunk's section_path is empty.
    """
    sections: dict[str, list[str]] = {}
    for b in parsed.get("blocks", []):
        sp = b.get("section_path") or []
        key = sp[0] if sp else "__root__"
        text = (b.get("text") or "").strip()
        if not text:
            continue
        if b.get("type") == "section_heading":
            text = f"## {text}"
        elif b.get("type") == "list_item":
            text = f"- {text}"
        sections.setdefault(key, []).append(text)
    return {k: "\n\n".join(v) for k, v in sections.items()}


# ---------------------------------------------------------------------------
# LLM call (single)
# ---------------------------------------------------------------------------


def _build_prompt(chunk_text: str, section_text: str,
                  max_section_chars: int = 12000) -> str:
    """Render the prompt, truncating section if needed to keep things fast."""
    if len(section_text) > max_section_chars:
        # Keep beginning + end of section
        half = max_section_chars // 2
        section_text = (section_text[:half] + "\n\n[... section continues ...]\n\n"
                        + section_text[-half:])
    return PROMPT_TEMPLATE.format(section_text=section_text, chunk_text=chunk_text)


def generate_prefix_ollama(chunk_text: str, section_text: str,
                           model: str = OLLAMA_MODEL,
                           client: httpx.Client | None = None) -> str:
    """One Ollama call, returns the prefix string (no markdown wrappers)."""
    prompt = _build_prompt(chunk_text, section_text)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": MAX_PREFIX_TOKENS,
        },
    }
    own_client = False
    if client is None:
        client = httpx.Client(timeout=HTTP_TIMEOUT)
        own_client = True
    try:
        r = client.post(OLLAMA_URL, json=payload)
        r.raise_for_status()
        data = r.json()
        text = data.get("message", {}).get("content", "").strip()
        # Sanitize: drop accidental quotes/preamble
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()
        return text
    finally:
        if own_client:
            client.close()


# ---------------------------------------------------------------------------
# Groq (cloud) — OpenAI-compatible API
# ---------------------------------------------------------------------------

_GROQ_CLIENT = None


def _get_groq_client():
    global _GROQ_CLIENT
    if _GROQ_CLIENT is None:
        from openai import OpenAI
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set in env")
        _GROQ_CLIENT = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    return _GROQ_CLIENT


class _RPMLimiter:
    """Token-bucket-like RPM limiter. Thread-safe."""

    def __init__(self, rpm: int):
        self.rpm = max(1, rpm)
        self.interval = 60.0 / self.rpm
        self._lock = __import__("threading").Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.perf_counter()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.perf_counter()
            self._next_allowed = max(self._next_allowed, now) + self.interval


_GROQ_LIMITER: _RPMLimiter | None = None


def _get_groq_limiter() -> _RPMLimiter:
    global _GROQ_LIMITER
    if _GROQ_LIMITER is None:
        _GROQ_LIMITER = _RPMLimiter(GROQ_RPM)
    return _GROQ_LIMITER


def generate_prefix_groq(chunk_text: str, section_text: str,
                         model: str = GROQ_MODEL,
                         max_retries: int = 5) -> str:
    """Groq call with RPM throttling + 429 exponential backoff."""
    prompt = _build_prompt(chunk_text, section_text)
    client = _get_groq_client()
    limiter = _get_groq_limiter()
    last_err = None
    for attempt in range(max_retries):
        limiter.acquire()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=MAX_PREFIX_TOKENS,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1].strip()
            return text
        except Exception as exc:
            last_err = exc
            msg = str(exc).lower()
            # Retry on rate limit (429) or transient errors
            if "429" in msg or "rate" in msg or "timeout" in msg or "503" in msg:
                backoff = min(2 ** attempt, 30)
                log.warning(f"Groq transient error (attempt {attempt+1}/{max_retries}): {exc}; "
                            f"backing off {backoff}s")
                time.sleep(backoff)
                continue
            raise
    raise RuntimeError(f"Groq call failed after {max_retries} retries: {last_err}")


def generate_prefix(chunk_text: str, section_text: str,
                    provider: str | None = None, model: str | None = None) -> str:
    """Provider-agnostic dispatcher.

    Prefer the unified `clients.get_llm()` factory (reads GENERATION_PROVIDER /
    GENERATION_MODEL / *_API_KEY from .env with cloud→local fallback). When the
    factory is unavailable (or the caller pins a specific provider via the
    `provider` argument), fall back to the legacy per-provider paths.
    """
    if provider is None:
        try:
            from clients import get_llm
            llm = get_llm()
            prompt = _build_prompt(chunk_text, section_text)
            text = llm.complete(prompt, options={
                "temperature": 0.1,
                "max_tokens": MAX_PREFIX_TOKENS,
                "num_predict": MAX_PREFIX_TOKENS,
            }) or ""
            text = text.strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1].strip()
            return text
        except Exception as exc:
            log.warning(f"model_factory LLM unavailable ({exc}); "
                        f"falling back to legacy per-provider path")
    p = (provider or LLM_PROVIDER).lower()
    if p == "groq":
        return generate_prefix_groq(chunk_text, section_text,
                                    model=model or GROQ_MODEL)
    # default ollama
    with httpx.Client(timeout=HTTP_TIMEOUT) as cli:
        return generate_prefix_ollama(chunk_text, section_text,
                                      model=model or OLLAMA_MODEL, client=cli)


# ---------------------------------------------------------------------------
# Parallel orchestration
# ---------------------------------------------------------------------------


def enrich_chunks(chunks: list[dict[str, Any]],
                  section_lookup: dict[str, str],
                  doc_title: str = "",
                  model: str = OLLAMA_MODEL,
                  parallelism: int = PARALLELISM,
                  progress_every: int = 10) -> list[dict[str, Any]]:
    """Returns chunks with new fields contextual_prefix + text_with_context."""

    provider = (os.environ.get("LLM_PROVIDER", "ollama")).lower()

    def _one(idx_chunk: tuple[int, dict[str, Any]]) -> tuple[int, str, str]:
        idx, c = idx_chunk
        sp = c.get("section_path") or []
        section_key = sp[0] if sp else "__root__"
        section_text = section_lookup.get(section_key, c["text"])
        try:
            if provider == "groq":
                prefix = generate_prefix_groq(c["text"], section_text, model=model)
            else:
                with httpx.Client(timeout=HTTP_TIMEOUT) as cli:
                    prefix = generate_prefix_ollama(c["text"], section_text,
                                                    model=model, client=cli)
        except Exception as exc:
            log.warning(f"chunk {idx} prefix failed ({exc}); falling back to template")
            sp_str = " > ".join(c.get("section_path") or [])
            prefix = (f"This chunk is from {doc_title or 'the document'}'s "
                      f"section '{sp_str or 'main body'}'.")
        if not prefix:
            prefix = f"From section '{' > '.join(sp) if sp else 'main body'}'."
        text_with_context = f"{prefix}\n\n{c['text']}"
        return idx, prefix, text_with_context

    enriched = [dict(c) for c in chunks]
    t0 = time.perf_counter()
    completed = 0
    with ThreadPoolExecutor(max_workers=parallelism) as ex:
        futures = {ex.submit(_one, (i, c)): i for i, c in enumerate(enriched)}
        for fut in as_completed(futures):
            idx, prefix, text_with_context = fut.result()
            enriched[idx]["contextual_prefix"] = prefix
            enriched[idx]["text_with_context"] = text_with_context
            enriched[idx]["enrichment_model"] = model
            completed += 1
            if completed % progress_every == 0 or completed == len(enriched):
                elapsed = time.perf_counter() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(enriched) - completed) / rate if rate > 0 else 0
                log.info(f"  prefixed {completed}/{len(enriched)}  "
                         f"({rate:.2f}/s, eta {eta:.0f}s)")
    log.info(f"  enrichment done in {time.perf_counter()-t0:.1f}s")
    return enriched


def save_enriched(chunks: list[dict[str, Any]], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunk_count": len(chunks),
        "enrichment_model": chunks[0].get("enrichment_model", "?") if chunks else "?",
        "chunks": chunks,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return out_path
