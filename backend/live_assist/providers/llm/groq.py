"""Backward-compatibility shim.

All Live LLM logic has moved to llm_provider.py (LiveCycleLLM).
This module re-exports GroqLLM as an alias so any external code that imports
from this path continues to work without changes.
"""
from live_assist.providers.llm.llm_provider import LiveCycleLLM as GroqLLM  # noqa: F401

__all__ = ["GroqLLM"]
