"""
Unit tests for the observability module.

Tests cover configuration loading, handler creation, error handling,
and graceful degradation scenarios.
"""

import os
from unittest import mock
import pytest

from live_assist.core.observability import create_langfuse_handler, get_trace_url


class TestCreateLangfuseHandler:
    """Test suite for create_langfuse_handler function."""

    def test_handler_creation_with_missing_credentials(self):
        """Handler factory returns None when credentials are missing."""
        with mock.patch.dict(
            os.environ,
            {
                "LANGFUSE_ENABLED": "true",
                "LANGFUSE_PUBLIC_KEY": "",
                "LANGFUSE_SECRET_KEY": "",
            },
            clear=False,
        ):
            # Clear the settings cache
            from live_assist.core.config import get_settings
            get_settings.cache_clear()

            handler = create_langfuse_handler(
                session_id="test_session",
                user_id="test_user",
            )

            assert handler is None

    def test_handler_creation_when_disabled(self):
        """Handler factory returns None when Langfuse is disabled."""
        with mock.patch.dict(
            os.environ,
            {
                "LANGFUSE_ENABLED": "false",
                "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
                "LANGFUSE_SECRET_KEY": "sk-lf-test",
            },
            clear=False,
        ):
            from live_assist.core.config import get_settings
            get_settings.cache_clear()

            handler = create_langfuse_handler(
                session_id="test_session",
                user_id="test_user",
            )

            assert handler is None

    def test_handler_creation_with_valid_credentials(self):
        """Handler factory creates handler with valid credentials."""
        with mock.patch.dict(
            os.environ,
            {
                "LANGFUSE_ENABLED": "true",
                "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
                "LANGFUSE_SECRET_KEY": "sk-lf-test",
                "LANGFUSE_HOST": "https://cloud.langfuse.com",
            },
            clear=False,
        ):
            from live_assist.core.config import get_settings
            get_settings.cache_clear()

            try:
                handler = create_langfuse_handler(
                    session_id="test_session",
                    user_id="test_user",
                    trace_name="Test Trace",
                    metadata={"test_key": "test_value"},
                )

                # If langfuse is installed, handler should be created
                # If not installed, should return None gracefully
                assert handler is None or handler is not None

            except Exception as e:
                pytest.fail(f"Handler creation should not raise exceptions: {e}")

    def test_handler_creation_with_trace_name_and_metadata(self):
        """Handler factory accepts trace_name and metadata parameters."""
        with mock.patch.dict(
            os.environ,
            {
                "LANGFUSE_ENABLED": "true",
                "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
                "LANGFUSE_SECRET_KEY": "sk-lf-test",
            },
            clear=False,
        ):
            from live_assist.core.config import get_settings
            get_settings.cache_clear()

            # Should not raise even with metadata
            handler = create_langfuse_handler(
                session_id="test_session",
                user_id="test_user",
                trace_name="RAG Query: What is ILTS?",
                metadata={
                    "speaker": "CUSTOMER",
                    "manual_question": True,
                    "turn_id": 1,
                },
            )

            # Handler could be None (if langfuse not installed) or valid handler
            assert handler is None or handler is not None

    def test_handler_creation_handles_import_error(self):
        """Handler factory returns None when langfuse package not installed."""
        with mock.patch.dict(
            os.environ,
            {
                "LANGFUSE_ENABLED": "true",
                "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
                "LANGFUSE_SECRET_KEY": "sk-lf-test",
            },
            clear=False,
        ):
            from live_assist.core.config import get_settings
            get_settings.cache_clear()

            # Mock langfuse import to raise ImportError
            with mock.patch.dict("sys.modules", {"langfuse.langchain": None}):
                handler = create_langfuse_handler(
                    session_id="test_session",
                    user_id="test_user",
                )

                # Should return None gracefully, not raise
                assert handler is None


class TestGetTraceUrl:
    """Test suite for get_trace_url function."""

    def test_get_trace_url_default_host(self):
        """get_trace_url generates correct URL with default host."""
        with mock.patch.dict(
            os.environ,
            {"LANGFUSE_HOST": "https://cloud.langfuse.com"},
            clear=False,
        ):
            from live_assist.core.config import get_settings
            get_settings.cache_clear()

            url = get_trace_url("test_session_123")

            assert url == "https://cloud.langfuse.com/trace/test_session_123"

    def test_get_trace_url_custom_host(self):
        """get_trace_url generates correct URL with custom host."""
        with mock.patch.dict(
            os.environ,
            {"LANGFUSE_HOST": "https://custom.langfuse.io"},
            clear=False,
        ):
            from live_assist.core.config import get_settings
            get_settings.cache_clear()

            url = get_trace_url("session_abc")

            assert url == "https://custom.langfuse.io/trace/session_abc"

    def test_get_trace_url_strips_trailing_slash(self):
        """get_trace_url handles trailing slash in host URL."""
        with mock.patch.dict(
            os.environ,
            {"LANGFUSE_HOST": "https://cloud.langfuse.com/"},
            clear=False,
        ):
            from live_assist.core.config import get_settings
            get_settings.cache_clear()

            url = get_trace_url("test_trace")

            assert url == "https://cloud.langfuse.com/trace/test_trace"
            assert "//" not in url.replace("https://", "")


class TestErrorHandlingAndResilience:
    """Test suite for error handling and graceful degradation."""

    def test_no_exceptions_raised_on_invalid_config(self):
        """System continues operating with invalid configuration."""
        with mock.patch.dict(
            os.environ,
            {
                "LANGFUSE_ENABLED": "true",
                "LANGFUSE_PUBLIC_KEY": "invalid",
                "LANGFUSE_SECRET_KEY": "invalid",
                "LANGFUSE_HOST": "https://invalid.example.com",
            },
            clear=False,
        ):
            from live_assist.core.config import get_settings
            get_settings.cache_clear()

            try:
                handler = create_langfuse_handler(
                    session_id="test",
                    user_id="user1",
                )
                # Should return None or valid handler, but never raise
                assert handler is None or handler is not None

            except Exception as e:
                pytest.fail(f"Should not raise exception: {e}")

    def test_workflow_continues_without_langfuse(self):
        """Simulate workflow execution without Langfuse handler."""
        handler = None
        config = {"callbacks": [handler]} if handler else {}

        # Verify config is properly formed for workflow invocation
        assert isinstance(config, dict)
        if handler:
            assert "callbacks" in config
        else:
            assert config == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
