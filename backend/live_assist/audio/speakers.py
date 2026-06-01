from __future__ import annotations

from live_assist.core.models import Speaker


def normalize_speaker(speaker: str) -> Speaker:
    value = (speaker or "").strip().lower()
    if value in {"customer", "client", "caller", "prospect"}:
        return Speaker.CUSTOMER
    if value in {"worker", "manager", "agent", "advisor", "rm", "relationship manager", "salesperson"}:
        return Speaker.WORKER
    if value == "assistant":
        return Speaker.ASSISTANT
    return Speaker.UNKNOWN


def workflow_target_for_speaker(speaker: Speaker) -> str:
    if speaker == Speaker.CUSTOMER:
        return "live_assist_customer_turn"
    if speaker == Speaker.WORKER:
        return "context_update_only"
    if speaker == Speaker.ASSISTANT:
        return "assistant_answer"
    return "generic_live_feedback"

