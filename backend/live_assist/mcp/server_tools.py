from __future__ import annotations

from typing import Any


def list_live_assist_tools() -> list[dict[str, Any]]:
    """Return future MCP tool descriptors.

    This is a lightweight placeholder so the backend has a clear place to expose
    Live Assist capabilities as MCP tools later without coupling MCP to the
    desktop or extension frontends.
    """

    return [
        {
            "name": "get_live_call_context",
            "status": "planned",
            "description": "Read current call summary, recent turns, products, and stage.",
        },
        {
            "name": "suggest_live_assist_action",
            "status": "planned",
            "description": "Return answer, next question, objection handling, or stage guidance.",
        },
    ]

