from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class SequencedUtterance:
    text: str
    speaker_name: str
    has_interruptions: bool
    reason: str
    utterance_id: str
    possible_bleed: bool
    trace_id: str
    chunk_id: int
    turn_id: int
    chunk_buffer_duration_ms: float
    customer_speech_started_perf: float | None
    audio_start_timestamp: float
    audio_start_chunk: int
    final_received_timestamp: float = field(default_factory=time.perf_counter)
    buffer_enter_time: float = field(default_factory=time.perf_counter)


@dataclass
class ActiveUtterance:
    utterance_id: str
    speaker_name: str
    audio_start_timestamp: float
    audio_start_chunk: int
    registered_at: float = field(default_factory=time.perf_counter)


class TranscriptSequencer:
    """
    Releases completed utterances in speech-start order while accounting for
    utterances that have started but have not finalized yet.

    A timeout-only buffer can reorder completed items, but it cannot know that an
    earlier customer utterance is still open. This sequencer receives active
    utterance registrations from the STT partial path and holds later completed
    utterances until earlier active utterances either complete or the call drains.
    """

    def __init__(
        self,
        buffer_ms: float,
        forward_fn: Callable[..., Awaitable[None]],
        csv_logger: Any,
        call_id: str,
        logging_enabled: bool = False,
    ):
        self.buffer_ms = buffer_ms
        self.forward_fn = forward_fn
        self.csv_logger = csv_logger
        self.call_id = call_id
        self.logging_enabled = logging_enabled

        self._queue: list[SequencedUtterance] = []
        self._active: dict[str, ActiveUtterance] = {}
        self._arrival_order: list[str] = []
        self._lock = asyncio.Lock()
        self._running = True
        self._forwarding_order_counter = 0
        self._forward_tasks: set[asyncio.Task] = set()

    async def register_active(
        self,
        *,
        utterance_id: str,
        speaker_name: str,
        audio_start_timestamp: float,
        audio_start_chunk: int,
    ) -> None:
        """Record an in-progress utterance as soon as its first STT partial arrives."""
        async with self._lock:
            if utterance_id in self._active:
                return
            self._active[utterance_id] = ActiveUtterance(
                utterance_id=utterance_id,
                speaker_name=speaker_name,
                audio_start_timestamp=audio_start_timestamp,
                audio_start_chunk=audio_start_chunk,
            )

        if self.logging_enabled:
            self.csv_logger.log_transcript_buffer(
                call_id=self.call_id,
                utterance_id=utterance_id,
                speaker=speaker_name,
                audio_start_timestamp=audio_start_timestamp,
                audio_start_chunk=audio_start_chunk,
                final_received_timestamp=0.0,
                buffer_enter_time=0.0,
                buffer_release_time=0.0,
                wait_duration_ms=0.0,
                forwarding_order=0,
                release_reason="active_registered",
                blocked_by="",
                active_earlier_count=0,
            )

    async def enqueue(self, utterance: SequencedUtterance) -> None:
        """Add a completed utterance to the sequencing buffer."""
        async with self._lock:
            self._active.pop(utterance.utterance_id, None)
            self._queue.append(utterance)
            self._arrival_order.append(utterance.utterance_id)
            self._queue.sort(key=lambda x: (x.audio_start_timestamp, x.final_received_timestamp))

        if self.logging_enabled:
            self.csv_logger.log_transcript_buffer(
                call_id=self.call_id,
                utterance_id=utterance.utterance_id,
                speaker=utterance.speaker_name,
                audio_start_timestamp=utterance.audio_start_timestamp,
                audio_start_chunk=utterance.audio_start_chunk,
                final_received_timestamp=utterance.final_received_timestamp,
                buffer_enter_time=utterance.buffer_enter_time,
                buffer_release_time=0.0,
                wait_duration_ms=0.0,
                forwarding_order=0,
                release_reason="entered_buffer",
                blocked_by="",
                active_earlier_count=0,
            )

        await self._check_and_release()

    async def discard_active(self, utterance_id: str) -> None:
        async with self._lock:
            self._active.pop(utterance_id, None)

    async def flush_loop(self) -> None:
        while self._running:
            await asyncio.sleep(0.05)
            await self._check_and_release()

    async def drain(self) -> None:
        self._running = False
        await self._check_and_release(force_all=True)
        if self._forward_tasks:
            await asyncio.gather(*self._forward_tasks, return_exceptions=True)

    async def _check_and_release(self, force_all: bool = False) -> None:
        to_forward: list[tuple[SequencedUtterance, str, str, int, bool]] = []
        blocked_logs: list[tuple[SequencedUtterance, str, int]] = []

        async with self._lock:
            now = time.perf_counter()

            while self._queue:
                candidate = self._queue[0]
                age_ms = (now - candidate.buffer_enter_time) * 1000
                blockers = self._earlier_active_blockers(candidate)

                if not force_all and blockers:
                    blocked_logs.append((candidate, ",".join(b.utterance_id for b in blockers), len(blockers)))
                    break

                if not force_all and age_ms < self.buffer_ms:
                    break

                self._queue.pop(0)
                reordered = False
                if self._arrival_order and self._arrival_order[0] != candidate.utterance_id:
                    reordered = True
                if candidate.utterance_id in self._arrival_order:
                    self._arrival_order.remove(candidate.utterance_id)

                release_reason = "drain" if force_all else (
                    "audio_start_ready" if not blockers else "forced_with_blocker"
                )
                blocked_by = ",".join(b.utterance_id for b in blockers)
                to_forward.append((candidate, release_reason, blocked_by, len(blockers), reordered))

        for u, release_reason, blocked_by, active_earlier_count, reordered in to_forward:
            self._release(u, release_reason, blocked_by, active_earlier_count, reordered)

        if self.logging_enabled:
            for u, blocked_by, active_earlier_count in blocked_logs:
                now = time.perf_counter()
                self.csv_logger.log_transcript_buffer(
                    call_id=self.call_id,
                    utterance_id=u.utterance_id,
                    speaker=u.speaker_name,
                    audio_start_timestamp=u.audio_start_timestamp,
                    audio_start_chunk=u.audio_start_chunk,
                    final_received_timestamp=u.final_received_timestamp,
                    buffer_enter_time=u.buffer_enter_time,
                    buffer_release_time=0.0,
                    wait_duration_ms=(now - u.buffer_enter_time) * 1000,
                    forwarding_order=0,
                    release_reason="blocked_by_active_earlier",
                    blocked_by=blocked_by,
                    active_earlier_count=active_earlier_count,
                )

    def _earlier_active_blockers(self, candidate: SequencedUtterance) -> list[ActiveUtterance]:
        return [
            active
            for active in self._active.values()
            if active.utterance_id != candidate.utterance_id
            and active.audio_start_timestamp < candidate.audio_start_timestamp
        ]

    def _release(
        self,
        u: SequencedUtterance,
        release_reason: str,
        blocked_by: str,
        active_earlier_count: int,
        reordered: bool,
    ) -> None:
        self._forwarding_order_counter += 1
        forwarding_order = self._forwarding_order_counter
        now = time.perf_counter()
        wait_ms = (now - u.buffer_enter_time) * 1000

        if self.logging_enabled:
            self.csv_logger.log_transcript_buffer(
                call_id=self.call_id,
                utterance_id=u.utterance_id,
                speaker=u.speaker_name,
                audio_start_timestamp=u.audio_start_timestamp,
                audio_start_chunk=u.audio_start_chunk,
                final_received_timestamp=u.final_received_timestamp,
                buffer_enter_time=u.buffer_enter_time,
                buffer_release_time=now,
                wait_duration_ms=wait_ms,
                forwarding_order=forwarding_order,
                release_reason=release_reason,
                blocked_by=blocked_by,
                active_earlier_count=active_earlier_count,
            )
            self.csv_logger.log_transcript_order(
                call_id=self.call_id,
                speaker=u.speaker_name,
                utterance_id=u.utterance_id,
                audio_start_timestamp=u.audio_start_timestamp,
                audio_start_chunk=u.audio_start_chunk,
                final_timestamp=time.time(),
                processing_order=forwarding_order,
                another_waiting=bool(self._queue),
                reordered=reordered,
                reason=release_reason,
                blocked_by=blocked_by,
                active_earlier_count=active_earlier_count,
            )

        task = asyncio.create_task(self._forward(u))
        self._forward_tasks.add(task)

        def _done(done_task: asyncio.Task) -> None:
            self._forward_tasks.discard(done_task)

        task.add_done_callback(_done)

    async def _forward(self, u: SequencedUtterance) -> None:
        try:
            await self.forward_fn(
                text=u.text,
                speaker_name=u.speaker_name,
                has_interruptions=u.has_interruptions,
                reason=u.reason,
                utterance_id=u.utterance_id,
                possible_bleed=u.possible_bleed,
                trace_id=u.trace_id,
                chunk_id=u.chunk_id,
                turn_id=u.turn_id,
                chunk_buffer_duration_ms=u.chunk_buffer_duration_ms,
                customer_speech_started_perf=u.customer_speech_started_perf,
            )
        except Exception as exc:
            print(
                f"[TranscriptSequencer] Error forwarding utterance {u.utterance_id}: {exc}",
                flush=True,
            )
