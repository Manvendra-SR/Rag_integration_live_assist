"""Transcript-level echo filter for Live Assist MVP.

Maintains a two-slot state (active customer utterance + recently flushed
customer utterance) and suppresses worker utterances that are acoustic
echoes of customer speech, based on token-overlap ratio and fuzzy
substring matching.
"""

from __future__ import annotations

import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz

from live_assist.core.diagnostics import log_event

__all__ = ["EchoFilter", "EchoFilterResult"]

# ---------------------------------------------------------------------------
# Filler words removed during normalisation
# ---------------------------------------------------------------------------
_FILLER_WORDS: frozenset[str] = frozenset(
    {"um", "uh", "er", "ah", "like", "okay", "ok"}
)


# ---------------------------------------------------------------------------
# Internal slot dataclass
# ---------------------------------------------------------------------------
@dataclass
class _Slot:
    """A single customer utterance slot."""

    text: str  # Normalised text (for flushed) or raw accumulated text (for active)
    timestamp: float  # Unix epoch float (seconds)


# ---------------------------------------------------------------------------
# Public result dataclass
# ---------------------------------------------------------------------------
@dataclass
class EchoFilterResult:
    """Result returned by :meth:`EchoFilter.evaluate_worker`."""

    suppress: bool
    reason: str  # "token_overlap" | "partial_ratio" | "both" |
    #              "no_match" | "no_slot" | "too_short"
    matched_slot_text: str  # empty string if no match
    token_overlap_ratio: float  # [0.0, 1.0]
    partial_ratio: int  # [0, 100]


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    """Return a normalised version of *text* suitable for echo comparison.

    Steps:
    1. Lowercase.
    2. Remove Unicode punctuation characters (``unicodedata.category`` starts
       with ``"P"``).
    3. Remove whole-word filler tokens.
    4. Collapse whitespace.
    """
    if not text:
        return ""

    # Step 1 — lowercase
    text = text.lower()

    # Step 2 — strip Unicode punctuation
    text = "".join(
        ch for ch in text if not unicodedata.category(ch).startswith("P")
    )

    # Step 3 — remove filler words (already lowercase)
    tokens = [w for w in text.split() if w not in _FILLER_WORDS]

    # Step 4 — join (implicitly collapses whitespace)
    return " ".join(tokens)


def _tokenize(text: str) -> list[str]:
    """Split normalised *text* into tokens using whitespace splitting."""
    return text.split()


# ---------------------------------------------------------------------------
# Token overlap (multiset Jaccard)
# ---------------------------------------------------------------------------
def _token_overlap(a_tokens: list[str], b_tokens: list[str]) -> float:
    """Compute multiset Jaccard token overlap ratio.

    Returns a float in [0.0, 1.0].  Returns 0.0 if either list is empty.
    """
    if not a_tokens or not b_tokens:
        return 0.0

    counter_a = Counter(a_tokens)
    counter_b = Counter(b_tokens)

    # Multiset intersection: min of counts for each token
    intersection = sum((counter_a & counter_b).values())
    # Multiset union: max of counts for each token
    union = sum((counter_a | counter_b).values())

    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Main EchoFilter class
# ---------------------------------------------------------------------------
class EchoFilter:
    """Transcript-level echo suppressor.

    Maintains two slots:

    * **Active slot** — the accumulated text of the current, ongoing customer
      utterance.  Updated on every customer ASR partial via
      :meth:`update_active_slot`.  Cleared when the customer flushes.

    * **Flushed slot** — the normalised text of the most recently completed
      customer utterance.  Populated by :meth:`on_customer_flush`.  Expires
      after *grace_period_seconds*.

    When a worker utterance flushes, :meth:`evaluate_worker` compares its
    normalised text against both non-expired slots using token-overlap ratio
    and ``rapidfuzz.fuzz.partial_ratio``.
    """

    def __init__(
        self,
        grace_period_seconds: float = 4.0,
        token_overlap_threshold: float = 0.80,
        partial_ratio_threshold: int = 85,
        min_chars: int = 3,
        call_id: str = "",
    ) -> None:
        self._grace_period = grace_period_seconds
        self._token_threshold = token_overlap_threshold
        self._partial_threshold = partial_ratio_threshold
        self._min_chars = min_chars
        self._call_id = call_id

        # Active slot: raw accumulated text (build_final_transcript result)
        self._active: Optional[_Slot] = None

        # Flushed slot: normalised text + flush timestamp
        self._flushed: Optional[_Slot] = None

    # ------------------------------------------------------------------
    # Slot update methods
    # ------------------------------------------------------------------

    def update_active_slot(self, accumulated_text: str) -> None:
        """Update the active slot with the current customer accumulated text.

        Called on every customer ASR partial *after* ``merge_transcript`` has
        been applied, so *accumulated_text* is the result of
        ``build_final_transcript(state.transcript_buffer)``.
        """
        now = time.time()
        self._active = _Slot(text=accumulated_text, timestamp=now)

        log_event(
            "echo_filter",
            call_id=self._call_id,
            event="transcript_echo_filter_state_updated",
            slot="active",
            text=accumulated_text,
            timestamp=int(now * 1000),
        )

    def on_customer_flush(self, final_text: str) -> None:
        """Handle a customer utterance flush.

        Moves the active slot into the flushed slot (normalised) and clears
        the active slot.  Any existing flushed slot is overwritten.
        """
        now = time.time()
        normalised = _normalize(final_text)

        # Overwrite flushed slot (requirement: at most one flushed slot at any time)
        self._flushed = _Slot(text=normalised, timestamp=now)

        # Clear active slot
        self._active = None

        log_event(
            "echo_filter",
            call_id=self._call_id,
            event="transcript_echo_filter_state_updated",
            slot="flushed",
            text=normalised,
            timestamp=int(now * 1000),
        )

    # ------------------------------------------------------------------
    # Worker evaluation
    # ------------------------------------------------------------------

    def evaluate_worker(self, worker_text: str) -> EchoFilterResult:
        """Evaluate a flushed worker utterance for echo suppression.

        Returns an :class:`EchoFilterResult` indicating whether the utterance
        should be suppressed and why.
        """
        worker_norm = _normalize(worker_text)

        # Guard: too short to compare meaningfully
        if len(worker_norm) < self._min_chars:
            result = EchoFilterResult(
                suppress=False,
                reason="too_short",
                matched_slot_text="",
                token_overlap_ratio=0.0,
                partial_ratio=0,
            )
            self._log_evaluated(worker_text, result)
            return result

        # Gather non-expired candidate slots
        now = time.time()
        candidates: list[tuple[str, str]] = []  # list of (label, normalised_text)

        if self._active is not None and self._active.text:
            # Active slot text is raw; normalise at comparison time
            candidates.append(("active", _normalize(self._active.text)))

        if self._flushed is not None:
            age = now - self._flushed.timestamp
            if age <= self._grace_period:
                candidates.append(("flushed", self._flushed.text))

        # Guard: no slot available
        if not candidates:
            result = EchoFilterResult(
                suppress=False,
                reason="no_slot",
                matched_slot_text="",
                token_overlap_ratio=0.0,
                partial_ratio=0,
            )
            self._log_evaluated(worker_text, result)
            return result

        # Compare worker against each candidate slot
        worker_tokens = _tokenize(worker_norm)
        best_overlap = 0.0
        best_partial = 0
        best_slot_text = ""
        suppress = False
        reason = "no_match"

        for _label, slot_norm in candidates:
            if not slot_norm:
                continue

            slot_tokens = _tokenize(slot_norm)

            # Token overlap ratio
            overlap = _token_overlap(worker_tokens, slot_tokens)

            # Partial ratio (rapidfuzz) — only invoke when tokens are non-empty
            if worker_tokens and slot_tokens:
                pr = fuzz.partial_ratio(worker_norm, slot_norm)
            else:
                pr = 0

            # Track best scores across all slots
            if overlap > best_overlap or (overlap == best_overlap and pr > best_partial):
                best_overlap = overlap
                best_partial = pr
                best_slot_text = slot_norm

            # Determine suppression
            token_hit = overlap >= self._token_threshold
            partial_hit = pr >= self._partial_threshold

            if token_hit and partial_hit:
                suppress = True
                reason = "both"
                best_overlap = overlap
                best_partial = pr
                best_slot_text = slot_norm
                break
            elif token_hit:
                suppress = True
                reason = "token_overlap"
                best_overlap = overlap
                best_partial = pr
                best_slot_text = slot_norm
                break
            elif partial_hit:
                suppress = True
                reason = "partial_ratio"
                best_overlap = overlap
                best_partial = pr
                best_slot_text = slot_norm
                break

        result = EchoFilterResult(
            suppress=suppress,
            reason=reason,
            matched_slot_text=best_slot_text,
            token_overlap_ratio=round(best_overlap, 6),
            partial_ratio=best_partial,
        )
        self._log_evaluated(worker_text, result)
        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_state_snapshot(self) -> dict:
        """Return a snapshot of the current slot state for debugging/testing."""
        now = time.time()

        flushed_text = self._flushed.text if self._flushed else None
        flushed_at = self._flushed.timestamp if self._flushed else None
        flushed_expires_at = (
            self._flushed.timestamp + self._grace_period if self._flushed else None
        )
        flushed_expired = (
            (now - self._flushed.timestamp) > self._grace_period
            if self._flushed is not None
            else False
        )

        return {
            "active_slot_text": self._active.text if self._active else None,
            "active_slot_last_updated": self._active.timestamp if self._active else None,
            "flushed_slot_text": flushed_text,
            "flushed_slot_flushed_at": flushed_at,
            "flushed_slot_expires_at": flushed_expires_at,
            "flushed_slot_expired": flushed_expired,
        }

    def _log_evaluated(self, worker_text: str, result: EchoFilterResult) -> None:
        """Emit a structured diagnostics log event for an evaluation."""
        log_event(
            "echo_filter",
            call_id=self._call_id,
            event="transcript_echo_filter_evaluated",
            worker_text=worker_text,
            matched_slot=result.matched_slot_text if result.matched_slot_text else None,
            token_overlap=result.token_overlap_ratio,
            partial_ratio=result.partial_ratio,
            suppressed=result.suppress,
            reason=result.reason,
        )
