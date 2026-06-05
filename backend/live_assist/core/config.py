from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent


CONFIG_KEY_ALIASES = {
    "APP_ENV": "app_env",
    "INPUT_SOURCE": "input_source",
    "DESKTOP_AUDIO_CAPTURE_MODE": "desktop_audio_capture_mode",
    "LIVE_ASSIST_DIAGNOSTICS_ENABLED": "diagnostics_enabled",
    "HOST": "stream_host",
    "PYTHON_WS_PORT": "stream_port",
    "PYTHON_WS_SARVAM_API_KEY": "sarvam_api_key",
    "PYTHON_WS_SAMPLE_RATE": "sample_rate",
    "PYTHON_WS_CHUNK_SIZE": "chunk_size",
    "PYTHON_WS_BUFFER_CHUNKS": "buffer_chunks",
    "PYTHON_WS_ENERGY_THRESHOLD": "energy_threshold",
    "PYTHON_WS_STT_SEND_RMS_FLOOR": "stt_send_rms_floor",
    "PYTHON_WS_LOW_ENERGY_PACKET_TARGET": "low_energy_packet_target",
    "PYTHON_WS_MAX_UTTERANCE_SECONDS": "max_utterance_seconds",
    "PYTHON_WS_INTERRUPTION_PACKET_THRESHOLD": "interruption_packet_threshold",
    "PYTHON_WS_EXPERIMENTAL_ECHO_TOOLS": "experimental_echo_tools_enabled",
    "PYTHON_WS_ECHO_SUPPRESSION_ENABLED": "echo_suppression_enabled",
    "PYTHON_WS_ECHO_STRONG_TEXT_SIMILARITY_THRESHOLD": "echo_strong_text_similarity_threshold",
    "PYTHON_WS_ECHO_WEAK_TEXT_SIMILARITY_THRESHOLD": "echo_weak_text_similarity_threshold",
    "PYTHON_WS_ECHO_WINDOW_SECONDS": "echo_window_seconds",
    "PYTHON_WS_ECHO_MIN_TEXT_CHARS": "echo_min_text_chars",
    "PYTHON_WS_ECHO_SHORT_TEXT_MAX_CHARS": "echo_short_text_max_chars",
    "PYTHON_WS_ECHO_WORKER_DOMINANCE_RATIO": "echo_worker_dominance_ratio",
    "PYTHON_WS_MIC_BLEED_SUPPRESSION_ENABLED": "mic_bleed_suppression_enabled",
    "PYTHON_WS_MIC_BLEED_WORKER_DOMINANCE_RATIO": "mic_bleed_worker_dominance_ratio",
    "PYTHON_WS_MIC_BLEED_INTERJECTION_CONFIRM_PACKETS": "mic_bleed_interjection_confirm_packets",
    "PYTHON_WS_MIC_BLEED_CUSTOMER_LEAD_PACKETS": "mic_bleed_customer_lead_packets",
    "PYTHON_WS_MIC_BLEED_MIN_WORKER_RMS": "mic_bleed_min_worker_rms",
    "PYTHON_WS_SIMPLE_OVERLAP_SUPPRESSION_ENABLED": "simple_overlap_suppression_enabled",
    "PYTHON_WS_SIMPLE_OVERLAP_PACKET_TARGET": "simple_overlap_packet_target",
    "ASR_PROVIDER": "asr_provider",
    "TRANSLATION_PROVIDER": "translation_provider",
    "LLM_PROVIDER": "llm_provider",
    "RAG_PROVIDER": "rag_provider",
    "RAG_RETRIEVAL_MODE": "rag_retrieval_mode",
    "RAG_PIPELINE_NAME": "rag_pipeline_name",
    "RAG_TOP_K": "rag_top_k",
    "RAG_CANDIDATE_LIMIT": "rag_candidate_limit",
    "RAG_USER_SCOPE_ENABLED": "rag_user_scope_enabled",
    "RAG_RUNTIME_DIR": "rag_runtime_dir",
    "RAG_CHROMA_COLLECTION": "rag_chroma_collection",
    "TRANSCRIPT_STORAGE": "transcript_storage",
    "STATE_STORAGE": "state_storage",
    "LIVE_FEEDBACK_WEBHOOK_URL": "live_feedback_webhook_url",
    "LIVE_FEEDBACK_WEBHOOK_TIMEOUT": "live_feedback_webhook_timeout",
    "LIVE_FEEDBACK_BATCH_SIZE": "live_feedback_batch_size",
    "LIVE_FEEDBACK_SESSION_ID": "live_feedback_session_id",
    "LIVE_FEEDBACK_USER_ID": "live_feedback_user_id",
    "LIVE_FEEDBACK_MIN_CUSTOMER_WORKFLOW_CHARS": "live_feedback_min_customer_workflow_chars",
    "LIVE_FEEDBACK_RECENT_TURNS": "live_feedback_recent_turns",
    "GROQ_API_KEY": "groq_api_key",
    "OPENAI_API_KEY": "openai_api_key",
    "LLM_API_KEY": "llm_api_key",
    "LLM_BASE_URL": "llm_base_url",
    "LLM_STRUCTURED_OUTPUT_MODE": "llm_structured_output_mode",
    "HF_TOKEN": "hf_token",
    "LLM_MODEL": "llm_model",
    "TEMPERATURE": "temperature",
    "CHROMA_DB_COLLECTION_NAME": "chroma_db_collection_name",
    "EMBEDDING_MODEL": "embedding_model",
    "NUMBER_OF_CHUNKS_TO_RETRIVE": "number_of_chunks_to_retrieve",
    "NUMBER_OF_CHUNKS_TO_RETRIEVE": "number_of_chunks_to_retrieve",
    "WEIGHTAGE_OF_VECTOR_SIMILARITY": "weightage_of_vector_similarity",
    "CHROMA_DB_PERSISTENT_DIRECTORY": "chroma_db_persistent_directory",
    "LIVE_FEEDBACK_SQLITE_PATH": "live_feedback_sqlite_path",
    "MEMORY_SQLITE_PATH": "memory_sqlite_path",
    "ENABLE_MCP_SERVER_TOOLS": "enable_mcp_server_tools",
    "ENABLE_MCP_CLIENT_TOOLS": "enable_mcp_client_tools",
    "ENABLE_MEMORY": "enable_memory",
    "ENABLE_ENTITY_EXTRACTION": "enable_entity_extraction",
    "ENABLE_CRM_INTEGRATION": "enable_crm_integration",
    "ENABLE_CALENDAR_INTEGRATION": "enable_calendar_integration",
    "ENABLE_PRE_MEETING_AGENT": "enable_pre_meeting_agent",
    "ENABLE_POST_MEETING_AGENT": "enable_post_meeting_agent",
    "ENABLE_WINNING_PATTERN_CONTEXT": "enable_winning_pattern_context",
    "REWRITE_QUESTION_SYSTEM_PROMPT": "rewrite_question_system_prompt",
    "REWRITE_QUESTION_USER_PROMPT": "rewrite_question_user_prompt",
    "FINAL_RESPONSE_SYSTEM_PROMPT": "final_response_system_prompt",
    "FINAL_RESPONSE_USER_PROMPT": "final_response_user_prompt",
    "SUMMARIZATION_PROMPT": "summarization_prompt",
    "HISTORY_SUMMARIZATION_PROMPT": "summarization_prompt",
    "RECENT_N_MESSAGES_CONTEXT": "recent_n_messages_context",
}


def _coerce_config_key(key: str) -> str:
    return CONFIG_KEY_ALIASES.get(key, key.lower())


def _read_env_files() -> dict[str, str]:
    values: dict[str, str] = {}
    for env_file in (PROJECT_DIR / ".env", BACKEND_DIR / ".env", Path(".env")):
        if not env_file.exists():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    values.update(os.environ)
    return values


def _load_json_config(env_values: dict[str, str]) -> dict[str, Any]:
    configured_path = env_values.get("LIVE_ASSIST_CONFIG_FILE", "config/live_assist.local.json")
    config_path = Path(configured_path)
    if not config_path.is_absolute():
        config_path = PROJECT_DIR / config_path
    if not config_path.exists():
        return {}

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return {_coerce_config_key(key): value for key, value in raw.items()}


def _load_settings_values() -> dict[str, Any]:
    env_values = _read_env_files()
    values = _load_json_config(env_values)
    values.update(
        {
            _coerce_config_key(key): value
            for key, value in env_values.items()
            if key in CONFIG_KEY_ALIASES
        }
    )
    return values


class Settings(BaseSettings):
    """Environment-driven app settings.

    Keep secrets in environment variables or a local .env file that is never
    committed. Defaults are local-MVP friendly and can be replaced in Azure.
    """

    model_config = SettingsConfigDict(
        env_file=(str(PROJECT_DIR / ".env"), str(BACKEND_DIR / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["local", "dev", "prod"] = "local"
    input_source: Literal[
        "desktop_native",
        "desktop_native_stable",
        "desktop_native_diagnostic",
        "desktop_browser_fallback",
        "desktop",
        "extension",
    ] = "desktop_native_stable"
    desktop_audio_capture_mode: Literal[
        "desktop_native_stable",
        "desktop_native_diagnostic",
        "desktop_native",
        "desktop_browser_fallback",
        "native",
    ] = Field(default="desktop_native_stable", alias="DESKTOP_AUDIO_CAPTURE_MODE")
    diagnostics_enabled: bool = Field(default=False, alias="LIVE_ASSIST_DIAGNOSTICS_ENABLED")

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    stream_host: str = Field(default="0.0.0.0", alias="HOST")
    stream_port: int = Field(default=8089, alias="PYTHON_WS_PORT")

    asr_provider: Literal["sarvam"] = "sarvam"
    translation_provider: Literal["sarvam", "none"] = "sarvam"
    llm_provider: Literal["groq", "openai", "openai_compatible"] = "groq"
    # (removed duplicate rag_provider — see line 190 below)
    transcript_storage: Literal["sqlite"] = "sqlite"
    state_storage: Literal["sqlite", "memory"] = "sqlite"

    rag_provider: Literal["legacy_chroma", "advanced"] = "advanced"
    rag_retrieval_mode: str = "reranked"
    rag_pipeline_name: str = "anthropic"
    rag_top_k: int = 5
    rag_candidate_limit: int = 20
    rag_user_scope_enabled: bool = True
    rag_runtime_dir: str = "runtime"
    rag_chroma_collection: str = "LiveAssistAdvancedRAG"

    # Semantic query cache
    rag_cache_enabled: bool = True
    rag_cache_similarity_threshold: float = 0.97
    rag_cache_max_age_days: int = 7

    sarvam_api_key: str = Field(default="", alias="PYTHON_WS_SARVAM_API_KEY")
    sarvam_model: str = "saaras:v3"
    sarvam_mode: Literal["translate", "transcribe"] = "translate"
    sarvam_language_code: str = "en-IN"

    sample_rate: int = Field(default=16000, alias="PYTHON_WS_SAMPLE_RATE")
    chunk_size: int = Field(default=1024, alias="PYTHON_WS_CHUNK_SIZE")
    buffer_chunks: int = Field(default=6, alias="PYTHON_WS_BUFFER_CHUNKS")
    energy_threshold: int = Field(default=100, alias="PYTHON_WS_ENERGY_THRESHOLD")
    stt_send_rms_floor: float = Field(
        default=0.0,
        alias="PYTHON_WS_STT_SEND_RMS_FLOOR",
    )
    experimental_echo_tools_enabled: bool = Field(
        default=False,
        alias="PYTHON_WS_EXPERIMENTAL_ECHO_TOOLS",
    )
    low_energy_packet_target: int = Field(
        default=5,
        alias="PYTHON_WS_LOW_ENERGY_PACKET_TARGET",
    )
    max_utterance_seconds: float = Field(
        default=20.0,
        alias="PYTHON_WS_MAX_UTTERANCE_SECONDS",
    )
    interruption_packet_threshold: int = Field(
        default=3,
        alias="PYTHON_WS_INTERRUPTION_PACKET_THRESHOLD",
    )
    echo_suppression_enabled: bool = Field(
        default=False,
        alias="PYTHON_WS_ECHO_SUPPRESSION_ENABLED",
    )
    echo_strong_text_similarity_threshold: float = Field(
        default=0.78,
        alias="PYTHON_WS_ECHO_STRONG_TEXT_SIMILARITY_THRESHOLD",
    )
    echo_weak_text_similarity_threshold: float = Field(
        default=0.60,
        alias="PYTHON_WS_ECHO_WEAK_TEXT_SIMILARITY_THRESHOLD",
    )
    echo_window_seconds: float = Field(
        default=2.5,
        alias="PYTHON_WS_ECHO_WINDOW_SECONDS",
    )
    echo_min_text_chars: int = Field(
        default=4,
        alias="PYTHON_WS_ECHO_MIN_TEXT_CHARS",
    )
    echo_short_text_max_chars: int = Field(
        default=6,
        alias="PYTHON_WS_ECHO_SHORT_TEXT_MAX_CHARS",
    )
    echo_worker_dominance_ratio: float = Field(
        default=1.5,
        alias="PYTHON_WS_ECHO_WORKER_DOMINANCE_RATIO",
    )
    mic_bleed_suppression_enabled: bool = Field(
        default=False,
        alias="PYTHON_WS_MIC_BLEED_SUPPRESSION_ENABLED",
    )
    mic_bleed_worker_dominance_ratio: float = Field(
        default=1.4,
        alias="PYTHON_WS_MIC_BLEED_WORKER_DOMINANCE_RATIO",
    )
    mic_bleed_interjection_confirm_packets: int = Field(
        default=2,
        alias="PYTHON_WS_MIC_BLEED_INTERJECTION_CONFIRM_PACKETS",
    )
    mic_bleed_customer_lead_packets: int = Field(
        default=2,
        alias="PYTHON_WS_MIC_BLEED_CUSTOMER_LEAD_PACKETS",
    )
    mic_bleed_min_worker_rms: float = Field(
        default=100.0,
        alias="PYTHON_WS_MIC_BLEED_MIN_WORKER_RMS",
    )
    simple_overlap_suppression_enabled: bool = Field(
        default=False,
        alias="PYTHON_WS_SIMPLE_OVERLAP_SUPPRESSION_ENABLED",
    )
    simple_overlap_packet_target: int = Field(
        default=3,
        alias="PYTHON_WS_SIMPLE_OVERLAP_PACKET_TARGET",
    )

    customer_speaker_label: str = "Customer"
    worker_speaker_label: str = "Worker"

    live_feedback_webhook_url: str = (
        "http://127.0.0.1:8000/livefeedback/webhook"
    )
    live_feedback_webhook_timeout: float = 60.0
    live_feedback_batch_size: int = 1
    live_feedback_session_id: str = "global_session"
    live_feedback_user_id: str = "user_01"
    live_feedback_min_customer_workflow_chars: int = 15
    live_feedback_recent_turns: int = 5

    groq_api_key: str = ""
    openai_api_key: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_structured_output_mode: Literal["json_prompt", "native", "native_fallback"] = "json_prompt"
    hf_token: str = ""
    llm_model: str = "openai/gpt-oss-20b"
    temperature: float = 0

    chroma_db_collection_name: str = "finideas_collection"
    embedding_model: str = "all-MiniLM-L6-v2"
    number_of_chunks_to_retrieve: int = 3
    weightage_of_vector_similarity: float = 0.7
    chroma_db_persistent_directory: str = "./chroma_db"

    live_feedback_sqlite_path: str = "data/live_feedback.sqlite3"
    memory_sqlite_path: str = "data/finideas_memory.sqlite3"

    enable_mcp_server_tools: bool = False
    enable_mcp_client_tools: bool = False
    enable_memory: bool = False
    enable_entity_extraction: bool = False
    enable_crm_integration: bool = False
    enable_calendar_integration: bool = False
    enable_pre_meeting_agent: bool = False
    enable_post_meeting_agent: bool = False
    enable_winning_pattern_context: bool = False

    rewrite_question_system_prompt: str = (
        "You are a query rewriting assistant for a financial sales AI. "
        "Your job is to rewrite the user's question into a clear, standalone, "
        "retrieval-friendly query and identify the product if explicitly mentioned. "
        "Always return a rewritten question — never return an empty question. "
        "If the question is very short (e.g. a single word or greeting), expand it "
        "into a complete question using context clues. "
        "Return structured output only."
    )
    rewrite_question_user_prompt: str = (
        "Rewrite the question into a complete, self-contained retrieval query. "
        "If a financial product (ILTS, FGF, FinRakshak, BharatBond, SWP) is mentioned, "
        "include it in the rewritten question and set the product field. "
        "Always return a non-empty rewritten_question."
    )
    final_response_system_prompt: str = (
        "You are a strict, factual financial sales assistant. Generate responses "
        "only using retrieved context. If context does not answer the question, "
        "respond exactly with NO_MATCH. Keep the response concise and suitable "
        "for a salesperson to say to a customer."
    )
    final_response_user_prompt: str = (
        "Using only the context, generate a clear and concise answer to the "
        "customer query."
    )
    summarization_prompt: str = (
        "Create or update a crisp summary of the financial advisory conversation. "
        "Capture key customer questions, important answers, salesperson context, "
        "and products discussed. Return summary text only."
    )
    recent_n_messages_context: int = 5

    def workflow_config(self) -> dict:
        return {
            "LLM_PROVIDER": self.llm_provider,
            "GROQ_API_KEY": self.groq_api_key,
            "OPENAI_API_KEY": self.openai_api_key,
            "LLM_API_KEY": self.llm_api_key,
            "LLM_BASE_URL": self.llm_base_url,
            "LLM_STRUCTURED_OUTPUT_MODE": self.llm_structured_output_mode,
            "HF_TOKEN": self.hf_token,
            "CHROMA_DB_COLLECTION_NAME": self.chroma_db_collection_name,
            "EMBEDDING_MODEL": self.embedding_model,
            "NUMBER_OF_CHUNKS_TO_RETRIVE": self.number_of_chunks_to_retrieve,
            "WEIGHTAGE_OF_VECTOR_SIMILARITY": self.weightage_of_vector_similarity,
            "CHROMA_DB_PERSISTENT_DIRECTORY": self.chroma_db_persistent_directory,
            "LLM_MODEL": self.llm_model,
            "TEMPERATURE": self.temperature,
            "REWRITE_QUESTION_SYSTEM_PROMPT": self.rewrite_question_system_prompt,
            "REWRITE_QUESTION_USER_PROMPT": self.rewrite_question_user_prompt,
            "FINAL_RESPONSE_SYSTEM_PROMPT": self.final_response_system_prompt,
            "FINAL_RESPONSE_USER_PROMPT": self.final_response_user_prompt,
            "SUMMARIZATION_PROMPT": self.summarization_prompt,
            "RECENT_N_MESSAGES_CONTEXT": self.recent_n_messages_context,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings(**_load_settings_values())
