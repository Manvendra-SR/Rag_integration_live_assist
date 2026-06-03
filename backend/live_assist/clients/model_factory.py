"""Unified cloud/local model factory (T-07).

One .env, three abstractions, automatic cloud→local fallback. Every model the
pipeline touches goes through this module:

    from live_assist.clients import get_vlm, get_llm, get_embedder

    vlm   = get_vlm()          # describe image / extract table
    llm   = get_llm()           # text generation / contextual prefixes
    embed = get_embedder()      # bge-m3 / OpenAI 3-large / etc.

Each factory reads `.env`:
    <NAMESPACE>_PROVIDER  (gemini | groq | openai | anthropic | cohere | ollama | local)
    <NAMESPACE>_MODEL     (model name)
    <PROVIDER>_API_KEY    (when cloud)

Cloud is tried first if the matching API key is non-empty. On failure (missing
SDK, blank/wrong key, auth 401/403, connection refused, timeout) the factory
**silently falls back to the local Ollama/SentenceTransformers entry** and
logs the swap so the user knows what happened. The chosen client is cached
per-namespace so subsequent calls don't re-decide.

Add a new cloud provider: add a `_make_<provider>_<kind>()` function that
returns a small object with the unified method (`.describe_image`,
`.complete`, `.embed`), then register it in the dispatch table at the bottom.
"""
from __future__ import annotations

import base64
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Best-effort .env load (no hard dep on python-dotenv).
def _load_dotenv() -> None:
    _backend_dir = Path(__file__).resolve().parent.parent.parent
    _root_dir = _backend_dir.parent
    
    try:
        from dotenv import load_dotenv
        load_dotenv(_root_dir / ".env")
        load_dotenv(_backend_dir / ".env", override=True)
    except Exception:
        pass


_load_dotenv()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cached clients (one per namespace)
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}


def reset_clients() -> None:
    """Drop all cached clients. Mostly for tests / when env is reloaded."""
    with _LOCK:
        _CACHE.clear()


# ---------------------------------------------------------------------------
# Unified interfaces
# ---------------------------------------------------------------------------
@dataclass
class VLMClient:
    name: str                # e.g. "gemini:gemini-2.0-flash" or "ollama:llama3.2-vision:11b"
    provider: str
    model: str
    is_local: bool
    _call: Any               # callable: (image_path:str, prompt:str, options:dict) -> str|None

    def describe_image(self, image_path: str, prompt: str,
                       options: dict | None = None) -> str | None:
        return self._call(image_path, prompt, options or {})


@dataclass
class LLMClient:
    name: str
    provider: str
    model: str
    is_local: bool
    _call: Any               # callable: (prompt:str, options:dict) -> str|None

    def complete(self, prompt: str, options: dict | None = None) -> str | None:
        return self._call(prompt, options or {})


@dataclass
class EmbedClient:
    name: str
    provider: str
    model: str
    is_local: bool
    dim: int
    _embed: Any              # callable: (texts:list[str]) -> list[list[float]]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# ===========================================================================
# Low-level builders — one per provider+kind. They raise on any failure;
# the dispatcher layer catches and falls back.
# ===========================================================================

# ---------- VLM ----------
def _make_ollama_vlm(model: str) -> VLMClient:
    import ollama
    host = os.environ.get("PARSER_OLLAMA_API_URL", "http://127.0.0.1:11434/api/chat")
    # ollama python client takes base URL, not /api/chat path
    base = host.split("/api/")[0]
    client = ollama.Client(host=base, timeout=None)

    def _call(image_path: str, prompt: str, options: dict) -> str | None:
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [image_path]}],
            options=options or {},
        )
        return ((resp.get("message") or {}).get("content") or "").strip() or None

    return VLMClient(f"ollama:{model}", "ollama", model, True, _call)


def _make_gemini_vlm(model: str, api_key: str) -> VLMClient:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(model)

    def _call(image_path: str, prompt: str, options: dict) -> str | None:
        from PIL import Image
        img = Image.open(image_path)
        cfg = {"temperature": float(options.get("temperature", 0.1))}
        if "num_predict" in options:
            cfg["max_output_tokens"] = int(options["num_predict"])
        resp = gm.generate_content([prompt, img], generation_config=cfg)
        return (resp.text or "").strip() or None

    return VLMClient(f"gemini:{model}", "gemini", model, False, _call)


def _make_anthropic_vlm(model: str, api_key: str) -> VLMClient:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    def _call(image_path: str, prompt: str, options: dict) -> str | None:
        with open(image_path, "rb") as fh:
            img_b64 = base64.b64encode(fh.read()).decode("ascii")
        media = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
        msg = client.messages.create(
            model=model,
            max_tokens=int(options.get("num_predict", 1024)),
            temperature=float(options.get("temperature", 0.1)),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": media, "data": img_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        if not msg.content:
            return None
        return (msg.content[0].text or "").strip() or None

    return VLMClient(f"anthropic:{model}", "anthropic", model, False, _call)


def _make_openai_vlm(model: str, api_key: str) -> VLMClient:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    def _call(image_path: str, prompt: str, options: dict) -> str | None:
        with open(image_path, "rb") as fh:
            img_b64 = base64.b64encode(fh.read()).decode("ascii")
        media = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
        resp = client.chat.completions.create(
            model=model,
            max_tokens=int(options.get("num_predict", 1024)),
            temperature=float(options.get("temperature", 0.1)),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{media};base64,{img_b64}"}},
                ],
            }],
        )
        return (resp.choices[0].message.content or "").strip() or None

    return VLMClient(f"openai:{model}", "openai", model, False, _call)


# ---------- LLM (text) ----------
def _make_ollama_llm(model: str) -> LLMClient:
    import ollama
    host = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat").split("/api/")[0]
    client = ollama.Client(host=host, timeout=None)

    def _call(prompt: str, options: dict) -> str | None:
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options=options or {},
        )
        return ((resp.get("message") or {}).get("content") or "").strip() or None

    return LLMClient(f"ollama:{model}", "ollama", model, True, _call)


def _make_groq_llm(model: str, api_key: str) -> LLMClient:
    from groq import Groq
    client = Groq(api_key=api_key)

    def _call(prompt: str, options: dict) -> str | None:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=int(options.get("max_tokens", options.get("num_predict", 1024))),
            temperature=float(options.get("temperature", 0.0)),
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip() or None

    return LLMClient(f"groq:{model}", "groq", model, False, _call)


def _make_gemini_llm(model: str, api_key: str) -> LLMClient:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(model)

    def _call(prompt: str, options: dict) -> str | None:
        cfg = {"temperature": float(options.get("temperature", 0.0))}
        if "max_tokens" in options or "num_predict" in options:
            cfg["max_output_tokens"] = int(options.get("max_tokens", options.get("num_predict", 1024)))
        resp = gm.generate_content(prompt, generation_config=cfg)
        return (resp.text or "").strip() or None

    return LLMClient(f"gemini:{model}", "gemini", model, False, _call)


def _make_openai_llm(model: str, api_key: str) -> LLMClient:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    def _call(prompt: str, options: dict) -> str | None:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=int(options.get("max_tokens", options.get("num_predict", 1024))),
            temperature=float(options.get("temperature", 0.0)),
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip() or None

    return LLMClient(f"openai:{model}", "openai", model, False, _call)


def _make_anthropic_llm(model: str, api_key: str) -> LLMClient:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    def _call(prompt: str, options: dict) -> str | None:
        msg = client.messages.create(
            model=model,
            max_tokens=int(options.get("max_tokens", options.get("num_predict", 1024))),
            temperature=float(options.get("temperature", 0.0)),
            messages=[{"role": "user", "content": prompt}],
        )
        return (msg.content[0].text or "").strip() or None

    return LLMClient(f"anthropic:{model}", "anthropic", model, False, _call)


# ---------- Embeddings ----------
def _make_local_embedder(model: str) -> EmbedClient:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model)
    dim = int(m.get_sentence_embedding_dimension() or 1024)

    def _embed(texts: list[str]) -> list[list[float]]:
        import numpy as np
        vecs = m.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs).tolist()

    return EmbedClient(f"local:{model}", "local", model, True, dim, _embed)


def _make_openai_embedder(model: str, api_key: str) -> EmbedClient:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    # Known dimensions for OpenAI embedding-3 family
    dim = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072,
           "text-embedding-ada-002": 1536}.get(model, 1536)

    def _embed(texts: list[str]) -> list[list[float]]:
        resp = client.embeddings.create(model=model, input=list(texts))
        return [d.embedding for d in resp.data]

    return EmbedClient(f"openai:{model}", "openai", model, False, dim, _embed)


def _make_cohere_embedder(model: str, api_key: str) -> EmbedClient:
    import cohere
    client = cohere.ClientV2(api_key=api_key)
    dim = {"embed-v4.0": 1536, "embed-english-v3.0": 1024}.get(model, 1024)

    def _embed(texts: list[str]) -> list[list[float]]:
        resp = client.embed(
            texts=list(texts),
            model=model,
            input_type="search_document",
            embedding_types=["float"],
        )
        return resp.embeddings.float_

    return EmbedClient(f"cohere:{model}", "cohere", model, False, dim, _embed)


def _make_gemini_embedder(model: str, api_key: str) -> EmbedClient:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    dim = 768  # default for embedding-001 / text-embedding-004

    def _embed(texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            r = genai.embed_content(model=model, content=t, task_type="retrieval_document")
            out.append(r["embedding"])
        return out

    return EmbedClient(f"gemini:{model}", "gemini", model, False, dim, _embed)


# ===========================================================================
# Provider dispatch tables
# ===========================================================================
_VLM_BUILDERS = {
    "gemini":    (_make_gemini_vlm,    "GEMINI_API_KEY"),
    "anthropic": (_make_anthropic_vlm, "ANTHROPIC_API_KEY"),
    "openai":    (_make_openai_vlm,    "OPENAI_API_KEY"),
}

_LLM_BUILDERS = {
    "groq":      (_make_groq_llm,      "GROQ_API_KEY"),
    "gemini":    (_make_gemini_llm,    "GEMINI_API_KEY"),
    "openai":    (_make_openai_llm,    "OPENAI_API_KEY"),
    "anthropic": (_make_anthropic_llm, "ANTHROPIC_API_KEY"),
}

_EMBED_BUILDERS = {
    "openai":  (_make_openai_embedder, "OPENAI_API_KEY"),
    "cohere":  (_make_cohere_embedder, "COHERE_API_KEY"),
    "gemini":  (_make_gemini_embedder, "GEMINI_API_KEY"),
}


# ===========================================================================
# Public factory functions
# ===========================================================================
def _env(key: str, default: str = "") -> str:
    return (os.environ.get(key, default) or "").strip()


def _try_cloud(kind: str, builder, model: str, key_env: str,
               provider: str) -> Any | None:
    """Try to build a cloud client. Return None (and log) on any failure."""
    key = _env(key_env)
    if not key:
        log.info(f"[model_factory] {kind} cloud {provider} skipped: "
                 f"{key_env} is empty → using local fallback")
        return None
    try:
        c = builder(model, key)
        log.info(f"[model_factory] {kind} → cloud {provider}:{model}")
        return c
    except ImportError as e:
        log.warning(f"[model_factory] {kind} cloud {provider} SDK missing "
                    f"({e}); falling back to local")
        return None
    except Exception as e:
        log.warning(f"[model_factory] {kind} cloud {provider}:{model} "
                    f"failed ({type(e).__name__}: {e}); falling back to local")
        return None


def get_vlm() -> VLMClient:
    """Return the configured VLM client. Cloud first if key valid, else local Ollama."""
    with _LOCK:
        if "vlm" in _CACHE:
            return _CACHE["vlm"]
        provider = (_env("PARSER_VLM_PROVIDER", "ollama") or "ollama").lower()
        cloud_model = _env("PARSER_VLM_MODEL") or {
            "gemini": "gemini-2.0-flash",
            "anthropic": "claude-haiku-4-5",
            "openai": "gpt-4o-mini",
        }.get(provider, "")
        local_model = _env("PARSER_OLLAMA_VISION_MODEL", "llama3.2-vision:11b")

        client: VLMClient | None = None
        if provider in _VLM_BUILDERS:
            builder, key_env = _VLM_BUILDERS[provider]
            client = _try_cloud("VLM", builder, cloud_model, key_env, provider)
        elif provider not in ("ollama", "local"):
            log.warning(f"[model_factory] unknown VLM provider '{provider}', "
                        f"using local Ollama")

        if client is None:
            try:
                client = _make_ollama_vlm(local_model)
                log.info(f"[model_factory] VLM → local ollama:{local_model}")
            except Exception as e:
                raise RuntimeError(
                    f"VLM unavailable — cloud disabled/failed and local Ollama "
                    f"{local_model!r} could not start: {e}"
                )
        _CACHE["vlm"] = client
        return client


def get_llm() -> LLMClient:
    """Return the configured text-generation LLM. Cloud first if key valid, else local Ollama."""
    with _LOCK:
        if "llm" in _CACHE:
            return _CACHE["llm"]
        provider = (_env("GENERATION_PROVIDER", "ollama") or "ollama").lower()
        cloud_model = _env("GENERATION_MODEL") or {
            "groq": "llama-3.3-70b-versatile",
            "gemini": "gemini-2.0-flash",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-haiku-4-5",
        }.get(provider, "")
        local_model = _env("LLM_LOCAL_MODEL", "llama3.1")

        client: LLMClient | None = None
        if provider in _LLM_BUILDERS:
            builder, key_env = _LLM_BUILDERS[provider]
            client = _try_cloud("LLM", builder, cloud_model, key_env, provider)
        elif provider not in ("ollama", "local"):
            log.warning(f"[model_factory] unknown LLM provider '{provider}', "
                        f"using local Ollama")

        if client is None:
            try:
                client = _make_ollama_llm(local_model)
                log.info(f"[model_factory] LLM → local ollama:{local_model}")
            except Exception as e:
                raise RuntimeError(
                    f"LLM unavailable — cloud disabled/failed and local Ollama "
                    f"{local_model!r} could not start: {e}"
                )
        _CACHE["llm"] = client
        return client


def get_embedder() -> EmbedClient:
    """Return the configured embedding client. Cloud first if key valid, else local Sentence-Transformers."""
    with _LOCK:
        if "embedder" in _CACHE:
            return _CACHE["embedder"]
        provider = (_env("EMBEDDING_PROVIDER", "local") or "local").lower()
        cloud_model = _env("EMBEDDING_MODEL") or {
            "openai": "text-embedding-3-large",
            "cohere": "embed-v4.0",
            "gemini": "text-embedding-004",
        }.get(provider, "")
        local_model = _env("EMBED_LOCAL_MODEL", "BAAI/bge-m3")

        client: EmbedClient | None = None
        if provider in _EMBED_BUILDERS:
            builder, key_env = _EMBED_BUILDERS[provider]
            client = _try_cloud("Embedder", builder, cloud_model, key_env, provider)
        elif provider not in ("local",):
            log.warning(f"[model_factory] unknown EMBEDDING provider '{provider}', "
                        f"using local sentence-transformers")

        if client is None:
            try:
                client = _make_local_embedder(local_model)
                log.info(f"[model_factory] Embedder → local {local_model}")
            except Exception as e:
                raise RuntimeError(
                    f"Embedder unavailable — cloud disabled/failed and local "
                    f"sentence-transformers {local_model!r} could not load: {e}"
                )
        _CACHE["embedder"] = client
        return client
