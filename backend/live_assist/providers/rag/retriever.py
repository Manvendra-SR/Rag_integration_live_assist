from __future__ import annotations

from typing import Any

from live_assist.core.config import get_settings

def build_rag_retriever(settings: Any = None) -> Any:
    if settings is None:
        settings = get_settings()

    provider = getattr(settings, "rag_provider", "legacy_chroma")

    if provider == "advanced":
        from live_assist.providers.rag.advanced import AdvancedRetriever
        return AdvancedRetriever(settings)
    else:
        # Fallback to legacy — build the uppercase-key dict it expects
        from live_assist.providers.rag.legacy_chroma import ChromaDBRetriever
        legacy_config = {
            "EMBEDDING_MODEL": settings.embedding_model,
            "CHROMA_DB_PERSISTENT_DIRECTORY": settings.chroma_db_persistent_directory,
            "CHROMA_DB_COLLECTION_NAME": settings.chroma_db_collection_name,
        }
        return ChromaDBRetriever(legacy_config)
