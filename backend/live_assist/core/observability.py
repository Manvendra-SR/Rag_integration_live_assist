"""
Langfuse Observability Helper Module

This module provides centralized functions for creating and managing Langfuse
callback handlers for observability in the live_assist RAG system.

Requirements: 1.4, 9.1, 9.2, 9.3, 9.4, 9.5
"""

from typing import Optional
import logging

from live_assist.core.config import get_settings


logger = logging.getLogger(__name__)


def create_langfuse_handler(
    session_id: str,
    user_id: str,
    trace_name: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[object]:
    """
    Create a LangfuseCallbackHandler with proper configuration.

    This function handles all error cases gracefully and returns None if
    Langfuse is disabled, credentials are missing, or initialization fails.
    The system will continue operating without tracing when None is returned.

    Args:
        session_id: Conversation identifier (used as trace identifier)
        user_id: User identifier (passed as metadata)
        trace_name: Optional name for the trace (e.g., "RAG Query: What is ILTS?")
        metadata: Optional additional metadata to attach to the trace

    Returns:
        LangfuseCallbackHandler instance if successful, None otherwise

    Requirements:
        - 1.4: Load credentials from environment variables
        - 9.1: Return None when credentials missing
        - 9.2: Return None when Langfuse Cloud unreachable
        - 9.3: Log warnings but never raise exceptions
        - 9.4: Continue operation without callbacks when initialization fails
        - 9.5: Never block user requests due to Langfuse errors
    """
    settings = get_settings()

    # Check if Langfuse is enabled
    if not settings.langfuse_enabled:
        logger.debug("Langfuse observability is disabled")
        return None

    # Check if credentials are configured
    if not settings.is_langfuse_configured():
        logger.warning(
            "Langfuse credentials not configured - tracing disabled. "
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in environment."
        )
        return None

    try:
        import os
        from langfuse.langchain import CallbackHandler

        # Langfuse 4.x relies on environment variables for auth
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
        os.environ["LANGFUSE_HOST"] = settings.langfuse_base_url

        # Create handler
        handler = CallbackHandler()
        
        # Try to set attributes directly for fallback compatibility
        try:
            handler.session_id = session_id
            handler.user_id = user_id
        except Exception:
            pass

        logger.debug(
            f"Created Langfuse handler for session={session_id}, user={user_id}"
        )
        return handler

    except ImportError as e:
        logger.warning(
            f"Langfuse package not installed - tracing disabled: {e}. "
            "Install with: pip install langfuse"
        )
        return None

    except Exception as e:
        logger.warning(
            f"Failed to initialize Langfuse handler - tracing disabled: {e}"
        )
        return None


def get_trace_url(trace_id: str) -> str:
    """
    Generate Langfuse dashboard URL for a specific trace.

    This helper function constructs the full URL to view a trace in the
    Langfuse dashboard, using the configured Langfuse host.

    Args:
        trace_id: The trace identifier (typically the session_id)

    Returns:
        Full URL to the trace in the Langfuse dashboard

    Example:
        >>> get_trace_url("session_123")
        "https://cloud.langfuse.com/trace/session_123"

    Requirements:
        - 11.1: Log trace URL after workflow invocation for verification
    """
    settings = get_settings()
    host = settings.langfuse_base_url.rstrip("/")
    return f"{host}/trace/{trace_id}"
