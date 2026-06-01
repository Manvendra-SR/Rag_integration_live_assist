from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UtteranceState:
    transcript_buffer: list[str] = field(default_factory=list)
    utterance_id: str | None = None
    chunk_id: int | None = None
    turn_id: int | None = None
    waiting_for_silence: bool = False
    consecutive_low_energy_packets: int = 0
    has_interruptions: bool = False
    possible_bleed: bool = False
    utterance_started_at: float | None = None
    utterance_started_perf: float | None = None
    last_transcript_at: float | None = None

    def reset(self) -> None:
        self.transcript_buffer.clear()
        self.utterance_id = None
        self.chunk_id = None
        self.turn_id = None
        self.waiting_for_silence = False
        self.consecutive_low_energy_packets = 0
        self.has_interruptions = False
        self.possible_bleed = False
        self.utterance_started_at = None
        self.utterance_started_perf = None
        self.last_transcript_at = None


def merge_transcript(buffer: list[str], new_text: str) -> bool:
    text = new_text.strip()
    if not text:
        return False

    if not buffer:
        buffer.append(text)
        return True

    last = buffer[-1]
    if text == last:
        return False
    if text.startswith(last):
        buffer[-1] = text
        return True
    if last.startswith(text):
        return False
    buffer.append(text)
    return True


def build_final_transcript(buffer: list[str]) -> str:
    return " ".join(part.strip() for part in buffer if part.strip()).strip()
