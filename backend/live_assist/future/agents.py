from __future__ import annotations

from typing import Protocol

from live_assist.core.models import TranscriptTurn


class LiveAssistAgent(Protocol):
    """Contract for future specialist agents inside Live Assist."""

    name: str

    async def run(self, turn: TranscriptTurn, call_context: dict) -> dict:
        """Return a suggestion or state update for the current call turn."""


PLANNED_AGENT_SLOTS = [
    "product_answer_agent",
    "next_best_question_agent",
    "stage_guidance_agent",
    "objection_handling_agent",
    "relationship_building_agent",
    "compliance_risk_agent",
    "winning_pattern_agent",
]

