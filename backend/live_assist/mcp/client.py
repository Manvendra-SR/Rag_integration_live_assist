from __future__ import annotations

from typing import Any


class MCPClientBoundary:
    """Future boundary for calling external MCP servers.

    Examples later: CRM MCP, Calendar MCP, Email MCP, Slack/Teams MCP,
    long-term memory MCP, or web-search MCP.
    """

    async def call_tool(self, server_name: str, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("MCP client integration is intentionally deferred for MVP")

