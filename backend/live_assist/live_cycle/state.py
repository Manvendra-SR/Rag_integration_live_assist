from __future__ import annotations

from typing import Annotated

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class LiveAssistState(BaseModel):
    user_id: str = ""
    session_id: str = ""
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    speaker: str = ""
    trace_id: str = ""
    utterance_id: str = ""
    chunk_id: int = 0
    turn_id: int = 0
    turn_text: str = ""
    question: str = ""
    rewriten_question: str = ""
    product: str = ""
    product_context: str = ""
    running_summary: str = ""
    last_5_turns: list[dict[str, str]] = Field(default_factory=list)
    context: str = ""
    rag_top_chunks: list[str] = Field(default_factory=list)
    enrich_duration_ms: float = 0.0
    rag_retrieve_duration_ms: float = 0.0
    generation_duration_ms: float = 0.0
    answer: str = ""
    route: str = "context_only"
    manual_question: bool = False
    should_generate_answer: bool = False
    conversation_turn_count: int = 0
    summary_turn_count: int = 0
