from __future__ import annotations

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from dotenv import load_dotenv

_backend_dir = Path(__file__).resolve().parent.parent.parent
_root_dir = _backend_dir.parent
load_dotenv(_root_dir / ".env")
load_dotenv(_backend_dir / ".env", override=True)

import asyncio
import json
import math
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import count
from urllib.parse import parse_qs, urlparse

import httpx
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import websockets

from live_assist.audio.buffering import (
    UtteranceState,
    build_final_transcript,
    merge_transcript,
)
from live_assist.audio.pcm import (
    calculate_rms,
    pcm_to_wav_b64,
    split_stereo_to_mono,
    target_buffer_bytes,
)
from live_assist.core.config import get_settings
from live_assist.core.diagnostics import log_event

import csv
import threading
import queue
from datetime import datetime

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

class _AecCsvLogger:
    def __init__(self, log_dir="runtime/logs"):
        self.log_dir = log_dir
        self.q = queue.Queue(maxsize=10000)
        self.dropped_logs = 0
        self.thread = threading.Thread(target=self._writer_thread, daemon=True)
        self.thread.start()

    def _writer_thread(self):
        os.makedirs(self.log_dir, exist_ok=True)
        files = {}
        writers = {}
        
        def get_writer(filename, headers):
            if filename not in writers:
                filepath = os.path.join(self.log_dir, filename)
                file_exists = os.path.exists(filepath)
                f = open(filepath, "a", newline="", encoding="utf-8")
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(headers)
                files[filename] = f
                writers[filename] = writer
            return writers[filename], files[filename]

        last_flush = time.time()

        while True:
            try:
                item = self.q.get(timeout=1.0)
                if item is None: break
                filename, headers, row = item
                writer, _ = get_writer(filename, headers)
                writer.writerow(row)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[AEC Logger] Error writing to CSV: {e}", flush=True)
                
            now = time.time()
            if now - last_flush > 2.0:
                for f in files.values():
                    try: f.flush()
                    except Exception: pass
                last_flush = now

    def _put_safe(self, filename, headers, row):
        try:
            self.q.put_nowait((filename, headers, row))
        except queue.Full:
            self.dropped_logs += 1

    def log_nlms(self, call_id, chunk, delay, mu, ref_rms, mic_rms, clean_rms, error_rms, erle_db, alpha_est, w_peak, dt):
        echo_reduction_pct = 100.0 * (1.0 - clean_rms / mic_rms) if mic_rms > 0 else ""
        headers = ["timestamp","run_id","call_id","chunk","delay","mu","ref_rms","mic_rms","clean_rms","error_rms","erle_db","alpha_est","w_peak","dt","echo_reduction_pct"]
        row = [datetime.now().isoformat(), RUN_ID, call_id, chunk, delay, mu, ref_rms, mic_rms, clean_rms, error_rms, erle_db, alpha_est, w_peak, dt, echo_reduction_pct]
        self._put_safe("aec_metrics.csv", headers, row)

    def log_probe(self, call_id, window, lag_samples, lag_ms, correlation, ref_rms, mic_rms, stable_count):
        headers = ["timestamp","run_id","call_id","window","lag_samples","lag_ms","correlation","ref_rms","mic_rms","stable_count"]
        row = [datetime.now().isoformat(), RUN_ID, call_id, window, lag_samples, lag_ms, correlation, ref_rms, mic_rms, stable_count]
        self._put_safe("aec_probe.csv", headers, row)

    def log_stable_delay(self, call_id, current_delay, new_stable_delay, delta, threshold, action):
        headers = ["timestamp","run_id","call_id","current_delay","new_stable_delay","delta","threshold","action"]
        row = [datetime.now().isoformat(), RUN_ID, call_id, current_delay, new_stable_delay, delta, threshold, action]
        self._put_safe("aec_stable_delay.csv", headers, row)

_aec_csv_logger = _AecCsvLogger()
from live_assist.core.terminal_log import api_summary_timing, client_timing
from live_assist.providers.asr.sarvam import SarvamStreamingASR

settings = get_settings()

# ── Stage 0: AEC Delay Probe ─────────────────────────────────────────────────
# Activated by setting AEC_PROBE=1 in the environment (or .env file).
# When active, accumulates 2 seconds of PCM from both channels per session,
# computes a decimated normalised cross-correlation, and prints structured
# [AEC Probe] lines to stdout. Audio is NEVER modified.
#
# Cross-correlation convention:
#   C(lag) = sum_j ref[j] * mic[j + lag]
#   Positive lag  → mic LEADS ref  (echo arrives before SCKit captures it)
#   Negative lag  → mic LAGS  ref  (echo arrives after SCKit captures it)
# The expected result on macOS is positive lag (~40-120 ms) because
# ScreenCaptureKit introduces more latency than the acoustic echo path.

_AEC_PROBE_ENABLED: bool = os.environ.get("AEC_PROBE", "").strip() == "1"
_AEC_PROBE_SAMPLE_RATE: int = 16_000          # 16 kHz — matches the stereo PCM stream
_AEC_PROBE_WINDOW_SAMPLES: int = 32_000       # 2-second accumulation window
_AEC_PROBE_DECIMATION: int = 16               # 16:1 → ~1 kHz working rate
_AEC_PROBE_SEARCH_RANGE: int = 128            # ±128 decimated samples = ±128 ms
_AEC_PROBE_STABLE_THRESHOLD_SAMPLES: int = 64 # ±64 samples = ±4 ms for stability
_AEC_PROBE_STABLE_COUNT_REQUIRED: int = 3     # consecutive agreeing windows


@dataclass
class _AecProbe:
    """Per-session delay probe state. Thread-safe within a single async session."""

    call_id: str
    ref_buf: list = field(default_factory=list)   # float32 samples, left channel
    mic_buf: list = field(default_factory=list)   # float32 samples, right channel
    window_count: int = 0
    last_lag: int | None = None
    stable_count: int = 0
    stable_lag: int | None = None
    lag_candidates: deque = field(default_factory=lambda: deque(maxlen=5))

    def feed(self, left_pcm: bytes, right_pcm: bytes) -> None:
        """Decode PCM16 bytes, accumulate, compute cross-correlation every 2 seconds."""
        if left_pcm:
            self.ref_buf.extend(
                np.frombuffer(left_pcm, dtype=np.int16).astype(np.float32).tolist()
            )
        if right_pcm:
            self.mic_buf.extend(
                np.frombuffer(right_pcm, dtype=np.int16).astype(np.float32).tolist()
            )

        if (
            len(self.ref_buf) < _AEC_PROBE_WINDOW_SAMPLES
            or len(self.mic_buf) < _AEC_PROBE_WINDOW_SAMPLES
        ):
            return  # not enough data yet

        ref = np.array(self.ref_buf[: _AEC_PROBE_WINDOW_SAMPLES], dtype=np.float32)
        mic = np.array(self.mic_buf[: _AEC_PROBE_WINDOW_SAMPLES], dtype=np.float32)

        # Slide by 50 % so windows overlap and we see change over time
        slide = _AEC_PROBE_WINDOW_SAMPLES // 2
        self.ref_buf = self.ref_buf[slide:]
        self.mic_buf = self.mic_buf[slide:]

        lag_samples, correlation = self._estimate_delay(ref, mic)
        lag_ms = lag_samples / _AEC_PROBE_SAMPLE_RATE * 1000.0
        self.window_count += 1

        # Stability tracking
        if (
            self.last_lag is not None
            and abs(self.last_lag - lag_samples) <= _AEC_PROBE_STABLE_THRESHOLD_SAMPLES
        ):
            self.stable_count += 1
        else:
            self.stable_count = 0
            self.lag_candidates.clear()
            
        self.lag_candidates.append(lag_samples)
        self.last_lag = lag_samples

        if self.stable_count >= _AEC_PROBE_STABLE_COUNT_REQUIRED:
            new_stable_lag = int(np.median(self.lag_candidates))
            if _AEC_NLMS_ENABLED and self.stable_lag != new_stable_lag:
                print(
                    f"[AEC Probe] stable_lag_locked={new_stable_lag} candidates={list(self.lag_candidates)}",
                    flush=True,
                )
            self.stable_lag = new_stable_lag

        ref_rms = float(np.sqrt(np.mean(ref ** 2)))
        mic_rms = float(np.sqrt(np.mean(mic ** 2)))

        print(
            f"[AEC Probe] call={self.call_id} window={self.window_count}"
            f" lag_samples={lag_samples} lag_ms={lag_ms:.1f}"
            f" correlation={correlation:.4f}"
            f" ref_rms={ref_rms:.1f} mic_rms={mic_rms:.1f}"
            f" stable_count={self.stable_count}",
            flush=True,
        )
        _aec_csv_logger.log_probe(self.call_id, self.window_count, lag_samples, lag_ms, correlation, ref_rms, mic_rms, self.stable_count)

        if self.stable_count >= _AEC_PROBE_STABLE_COUNT_REQUIRED:
            print(
                f"[AEC Probe] call={self.call_id} status=STABLE"
                f" lag_samples={lag_samples} lag_ms={lag_ms:.1f}"
                f" correlation={correlation:.4f}"
                f" windows={self.window_count}",
                flush=True,
            )

    @staticmethod
    def _estimate_delay(ref: np.ndarray, mic: np.ndarray) -> tuple[int, float]:
        """Normalised cross-correlation on decimated signals. Returns (lag_samples, peak)."""
        step = _AEC_PROBE_DECIMATION
        ref_d = ref[::step]
        mic_d = mic[::step]
        n = min(len(ref_d), len(mic_d))
        if n < 8:
            return 0, 0.0

        search = _AEC_PROBE_SEARCH_RANGE
        best_lag = 0
        best_score = -math.inf

        for lag in range(-search, search + 1):
            j_start = max(0, -lag)
            j_end = min(n, n - lag)
            if j_end <= j_start:
                continue
            r = ref_d[j_start:j_end]
            m = mic_d[j_start + lag: j_end + lag]
            ss_r = float(np.dot(r, r))
            ss_m = float(np.dot(m, m))
            denom = math.sqrt(ss_r * ss_m)
            score = float(np.dot(r, m)) / denom if denom > 1e-9 else 0.0
            if score > best_score:
                best_score = score
                best_lag = lag

        return best_lag * step, best_score

# ── Stage 2: Block NLMS AEC ────────────────────────────────────────────────────────────
# Activated by AEC_NLMS=1.  Runs independently of AEC_PROBE.
# Replaces the right (mic) PCM channel with a cleaned signal before it
# reaches calculate_rms(), the STT gate, and the STT queue.
#
# Algorithm: block NLMS with fixed delay line
#   delay_samples  = 592  (Stage 0 dominant lag, ~37 ms @ 16 kHz)
#   filter_len     = 64   (4 ms @ 16 kHz, spans the 34–43 ms cluster)
#   One weight vector gradient step per chunk (O(N×L) numpy, no Python loop)
#   Double-talk detector freezes adaptation when mic >> ref
#
# alpha_est logged every chunk: the least-squares coupling coefficient that
# Stage 1 would have measured.  Useful for diagnosing convergence without
# having run Stage 1 separately.

_AEC_NLMS_ENABLED: bool      = os.environ.get("AEC_NLMS", "").strip() == "1"
_AEC_NLMS_DELAY_SAMPLES: int = 432    # Stage 0 dominant lag (~27 ms @ 16 kHz)
_AEC_NLMS_FILTER_LEN: int    = 64     # 4 ms @ 16 kHz
_AEC_NLMS_MU: float          = 0.15   # NLMS step size
_AEC_NLMS_EPS: float         = 1e-6   # power floor (prevent /0)
_AEC_NLMS_DT_RATIO: float    = 3.0    # mic/ref RMS ratio to freeze taps
_AEC_NLMS_DT_FLOOR: float    = 100.0  # minimum mic RMS (PCM16 scale) for DT


@dataclass
class _AecNlms:
    """
    Per-session block NLMS acoustic echo canceller (AEC_NLMS=1).

    For each chunk of N samples the update is fully vectorised:
      X  = sliding_window_view(aligned_ref_segment, filter_len)  # (N, L)
      y  = X @ w          # echo estimate
      e  = mic − y        # cleaned signal (sent to STT)
      w += mu/power * (X.T @ e) / N   [frozen during double-talk]

    delay_samples sets a fixed pre-delay on the reference before the
    adaptive taps.  The 64 taps then cover ±4 ms around that centre.
    """

    call_id: str
    delay_samples: int  = _AEC_NLMS_DELAY_SAMPLES
    filter_len: int     = _AEC_NLMS_FILTER_LEN
    mu: float           = _AEC_NLMS_MU
    eps: float          = _AEC_NLMS_EPS
    dt_ratio: float     = _AEC_NLMS_DT_RATIO
    dt_floor: float     = _AEC_NLMS_DT_FLOOR

    _ref_history: np.ndarray = field(init=False, repr=False)
    _w: np.ndarray           = field(init=False, repr=False)
    _max_history: int        = field(init=False, repr=False)
    _chunk_count: int        = field(default=0, init=False)
    _last_evaluated_delay: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._max_history = self.delay_samples + self.filter_len + 16_384
        self._ref_history = np.zeros(0, dtype=np.float64)
        self._w = np.zeros(self.filter_len, dtype=np.float64)

    def update_delay(self, new_stable_delay: int) -> bool:
        """
        Dynamically update the delay anchor if it shifts by a significant threshold.
        Returns True if a reset occurred, False otherwise.
        """
        if self._last_evaluated_delay == new_stable_delay:
            return False
            
        self._last_evaluated_delay = new_stable_delay
        
        threshold = 80
        delta = abs(self.delay_samples - new_stable_delay)
        
        if delta >= threshold:
            old_delay = self.delay_samples
            self.delay_samples = new_stable_delay
            
            # Reset weights: old alignment is invalid
            self._w = np.zeros(self.filter_len, dtype=np.float64)
            
            # Update max history bound
            new_max = self.delay_samples + self.filter_len + 16_384
            if new_max > self._max_history:
                self._max_history = new_max
                
            print(f"[AEC NLMS] current_delay={old_delay} new_stable_delay={new_stable_delay} "
                  f"delta={delta} threshold={threshold} action=update weights_reset=True", flush=True)
            _aec_csv_logger.log_stable_delay(self.call_id, old_delay, new_stable_delay, delta, threshold, "update")
            return True
        else:
            print(f"[AEC NLMS] current_delay={self.delay_samples} new_stable_delay={new_stable_delay} "
                  f"delta={delta} threshold={threshold} action=ignored", flush=True)
            _aec_csv_logger.log_stable_delay(self.call_id, self.delay_samples, new_stable_delay, delta, threshold, "ignored")
            return False

    def process(self, left_pcm: bytes, right_pcm: bytes) -> tuple[bytes, str]:
        """
        Apply block NLMS.  Returns (cleaned_right_pcm, log_line).
        Returns right_pcm unchanged if insufficient history has accumulated.
        """
        if not left_pcm or not right_pcm:
            return right_pcm, ""

        ref = np.frombuffer(left_pcm,  dtype=np.int16).astype(np.float64)
        mic = np.frombuffer(right_pcm, dtype=np.int16).astype(np.float64)
        n = min(len(ref), len(mic))
        ref, mic = ref[:n], mic[:n]

        # Append reference into running history; trim to cap memory
        self._ref_history = np.concatenate([self._ref_history, ref])
        if len(self._ref_history) > self._max_history:
            self._ref_history = self._ref_history[-self._max_history:]

        self._chunk_count += 1
        h = len(self._ref_history)
        needed = self.delay_samples + self.filter_len + n

        if h < needed:
            log = (
                f"[AEC NLMS] call={self.call_id} chunk={self._chunk_count}"
                f" status=warming_up history={h} needed={needed}"
            )
            return right_pcm, log

        # Build filter-input matrix X: shape (n, filter_len)
        # X[i, 0] = reference sample most recently aligned to mic[i]
        #         (i.e., delayed by delay_samples relative to the write head)
        # X[i, t] = t samples further in the past
        #
        # In _ref_history, the sample aligned to mic[i] is at index:
        #   h - n - delay_samples + i
        # The filter window for mic[i] spans from that index back by
        # (filter_len - 1) additional samples.
        center_end = h - self.delay_samples      # exclusive upper bound for mic[n-1]
        seg_start  = center_end - n - (self.filter_len - 1)
        seg_end    = center_end

        if seg_start < 0:
            log = (
                f"[AEC NLMS] call={self.call_id} chunk={self._chunk_count}"
                f" status=insufficient_history seg_start={seg_start}"
            )
            return right_pcm, log

        segment = self._ref_history[seg_start:seg_end]       # length: n + filter_len - 1
        X_raw   = sliding_window_view(segment, self.filter_len)  # shape (n, filter_len)
        # Reverse along the tap axis so w[0] corresponds to the aligned (most recent) sample
        X = X_raw[:, ::-1]                                   # shape (n, filter_len)

        # Echo estimate and cleaned residual
        y = X @ self._w     # (n,)  echo estimate
        e = mic - y         # (n,)  cleaned mic

        # Per-chunk RMS and alpha_est (Stage-1 equivalent: least-squares coupling)
        ref_rms = float(np.sqrt(np.mean(ref ** 2)))
        mic_rms = float(np.sqrt(np.mean(mic ** 2)))
        cln_rms = float(np.sqrt(np.mean(e   ** 2)))
        error_rms = cln_rms
        erle_db = (
            20.0 * math.log10(mic_rms / max(cln_rms, 1e-9))
            if mic_rms > 1e-9 else 0.0
        )
        aligned_col = X[:, 0]                                # most-recent aligned samples
        ss_ac  = float(np.dot(aligned_col, aligned_col))
        alpha_est = float(np.dot(mic, aligned_col)) / max(ss_ac, 1e-9)

        # Double-talk detection: freeze adaptation when mic >> ref
        dt = mic_rms > self.dt_ratio * ref_rms and mic_rms > self.dt_floor

        if not dt:
            grad    = X.T @ e                                # (filter_len,)
            x_power = float(np.sum(X ** 2)) / n + self.eps
            self._w += (self.mu / x_power) * grad / n

        w_peak = float(np.max(np.abs(self._w)))

        log = (
            f"[AEC NLMS] call={self.call_id} chunk={self._chunk_count}"
            f" delay={self.delay_samples} mu={self.mu:.3f}"
            f" ref_rms={ref_rms:.1f} mic_rms={mic_rms:.1f} clean_rms={cln_rms:.1f}"
            f" erle_db={erle_db:.2f} alpha_est={alpha_est:.4f}"
            f" w_peak={w_peak:.6f} dt={'frozen' if dt else 'adapting'}"
        )
        _aec_csv_logger.log_nlms(self.call_id, self._chunk_count, self.delay_samples, self.mu, ref_rms, mic_rms, cln_rms, error_rms, erle_db, alpha_est, w_peak, 'frozen' if dt else 'adapting')

        clean_i16 = np.clip(e, -32768, 32767).astype(np.int16)
        return clean_i16.tobytes(), log


@dataclass
class STTChunk:
    chunk_number: int
    pcm_bytes: bytes
    rms: float
    speech: bool
    queued_at: float
    native_seq: int | None = None
    trace_id: str = ""


def log_text(value: str, limit: int = 300) -> str:
    cleaned = " ".join((value or "").split())
    if len(cleaned) > limit:
        cleaned = f"{cleaned[:limit]}..."
    return json.dumps(cleaned, ensure_ascii=False)


def normalize_echo_text(value: str) -> str:
    text = (value or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = [
        word
        for word in text.split()
        if word not in {"um", "uh", "er", "ah", "like", "okay", "ok"}
    ]
    return " ".join(words)


def text_similarity(left: str, right: str) -> float:
    left_norm = normalize_echo_text(left)
    right_norm = normalize_echo_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm in right_norm or right_norm in left_norm:
        shorter = min(len(left_norm), len(right_norm))
        longer = max(len(left_norm), len(right_norm))
        return shorter / longer if longer else 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def classify_channel_activity(left_rms: float, right_rms: float, threshold: float) -> str:
    left_active = left_rms > threshold
    right_active = right_rms > threshold
    if left_active and right_active:
        return "both_active"
    if left_active:
        return "customer_only"
    if right_active:
        return "worker_only"
    return "silence"


def extract_call_id_from_path(path: str | None) -> str:
    if not path:
        return f"call-{uuid.uuid4().hex}"
    parsed = urlparse(path)
    query = parse_qs(parsed.query)
    for key in ("call_id", "session_id", "conversation_id"):
        values = query.get(key)
        if values and values[0]:
            return values[0]
    return f"call-{uuid.uuid4().hex}"


def extract_call_id_from_message(message: str) -> str | None:
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return None
    for key in ("call_id", "session_id", "conversation_id"):
        value = payload.get(key)
        if value:
            return str(value)
    return None


def parse_audio_chunk_meta(message: str) -> dict | None:
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return None
    if payload.get("type") != "AUDIO_CHUNK_META":
        return None
    return payload


async def handle_client(websocket) -> None:
    call_id = extract_call_id_from_path(getattr(websocket, "path", None))
    asr_provider = SarvamStreamingASR()
    diagnostic_mode = (
        settings.diagnostics_enabled
        or settings.desktop_audio_capture_mode == "desktop_native_diagnostic"
    )
    echo_tools_enabled = settings.experimental_echo_tools_enabled

    def diag_print(message: str) -> None:
        if diagnostic_mode:
            print(message, flush=True)

    async with (
        httpx.AsyncClient(timeout=settings.live_feedback_webhook_timeout) as feedback_client,
        asr_provider.connect() as ws_customer,
        asr_provider.connect() as ws_worker,
    ):
        diag_print(f"Two ASR sessions created for call_id={call_id}")

        audio_buffer = b""
        last_audio_recv_time = None
        latencies: list[float] = []
        customer_speech_packet_count = 0
        worker_speech_packet_count = 0
        customer_state = UtteranceState()
        worker_state = UtteranceState()
        utterance_lock = asyncio.Lock()
        buffer_target = target_buffer_bytes(settings.chunk_size, settings.buffer_chunks)
        received_audio_frames = 0
        processed_audio_chunks = 0
        utterance_counter = count(1)
        next_customer_chunk_id = 0
        last_left_rms = 0.0
        last_right_rms = 0.0
        recent_customer_text_events: list[dict[str, float | str]] = []
        recent_audio_activity: list[dict[str, float | int | str | bool]] = []
        consecutive_both_active_packets = 0
        customer_stt_queue: asyncio.Queue[STTChunk | None] = asyncio.Queue()
        worker_stt_queue: asyncio.Queue[STTChunk | None] = asyncio.Queue()
        background_tasks: set[asyncio.Task] = set()
        websocket_send_lock = asyncio.Lock()
        pending_audio_meta: deque[dict] = deque()
        expected_native_seq: int | None = None
        stt_backlog_warning_size = 10
        diag_print(
            f"[Buffer Config] call={call_id} target_bytes={buffer_target} "
            f"chunk_size={settings.chunk_size} buffer_chunks={settings.buffer_chunks} "
            f"threshold={settings.energy_threshold} "
            f"stt_send_floor={settings.stt_send_rms_floor} "
            f"silence_packets={settings.low_energy_packet_target} "
            f"max_utterance_seconds={settings.max_utterance_seconds} "
            f"diagnostics={'on' if diagnostic_mode else 'off'} "
            f"experimental_echo_tools={'on' if echo_tools_enabled else 'off'} "
            f"echo_suppression={'on' if echo_tools_enabled and settings.echo_suppression_enabled else 'off'} "
            f"mic_bleed_suppression={'on' if echo_tools_enabled and settings.mic_bleed_suppression_enabled else 'off'} "
            f"simple_overlap_suppression={'on' if echo_tools_enabled and settings.simple_overlap_suppression_enabled else 'off'} "
            f"simple_overlap_target={settings.simple_overlap_packet_target}"
        )

        def ensure_utterance_id(state: UtteranceState, speaker_name: str) -> str:
            if state.utterance_id is None:
                state.utterance_id = f"{call_id}:{speaker_name}:{next(utterance_counter)}"
            return state.utterance_id

        def timing_log(stage: str, **values) -> None:
            fields = " ".join(f"{key}={value}" for key, value in values.items())
            diag_print(f"[Timing] call={call_id} stage={stage} {fields}".rstrip())

        def make_trace_id(utterance_id: str) -> str:
            return f"{utterance_id}:{uuid.uuid4().hex[:8]}"

        def track_background_task(task: asyncio.Task) -> None:
            background_tasks.add(task)

            def _done(done_task: asyncio.Task) -> None:
                background_tasks.discard(done_task)
                try:
                    done_task.result()
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    diag_print(
                        f"[Background Task Error] call={call_id} "
                        f"type={type(exc).__name__} error={log_text(str(exc) or repr(exc))}"
                    )

            task.add_done_callback(_done)

        def remember_customer_text(text: str) -> None:
            now = time.time()
            recent_customer_text_events.append(
                {
                    "text": text,
                    "timestamp": now,
                    "left_rms": last_left_rms,
                    "right_rms": last_right_rms,
                }
            )
            cutoff = now - settings.echo_window_seconds
            recent_customer_text_events[:] = [
                event
                for event in recent_customer_text_events[-12:]
                if float(event.get("timestamp", 0.0)) >= cutoff
            ]

        def remember_audio_activity(
            chunk_number: int,
            left_rms: float,
            right_rms: float,
            left_has_speech: bool,
            right_has_speech: bool,
        ) -> None:
            now = time.time()
            activity = classify_channel_activity(left_rms, right_rms, settings.energy_threshold)
            recent_audio_activity.append(
                {
                    "timestamp": now,
                    "chunk": chunk_number,
                    "left_rms": left_rms,
                    "right_rms": right_rms,
                    "left_has_speech": left_has_speech,
                    "right_has_speech": right_has_speech,
                    "activity": activity,
                }
            )
            cutoff = now - settings.echo_window_seconds
            recent_audio_activity[:] = [
                event
                for event in recent_audio_activity[-80:]
                if float(event.get("timestamp", 0.0)) >= cutoff
            ]
            diag_print(
                f"[Channel Activity] call={call_id} chunk={chunk_number} "
                f"activity={activity} customer_rms={left_rms:.1f} worker_rms={right_rms:.1f}"
            )

        def recent_activity_summary() -> dict[str, bool | float | str]:
            cutoff = time.time() - settings.echo_window_seconds
            events = [
                event
                for event in recent_audio_activity
                if float(event.get("timestamp", 0.0)) >= cutoff
            ]
            customer_recent = any(str(event.get("activity")) in {"customer_only", "both_active"} for event in events)
            both_recent = any(str(event.get("activity")) == "both_active" for event in events)
            worker_recent = any(str(event.get("activity")) in {"worker_only", "both_active"} for event in events)
            max_left = max((float(event.get("left_rms", 0.0)) for event in events), default=0.0)
            max_right = max((float(event.get("right_rms", 0.0)) for event in events), default=0.0)
            latest_activity = str(events[-1].get("activity", "none")) if events else "none"
            worker_dominant = max_right >= max(
                settings.energy_threshold,
                max_left * settings.echo_worker_dominance_ratio,
            )
            return {
                "customer_recent": customer_recent,
                "both_recent": both_recent,
                "worker_recent": worker_recent,
                "max_left": max_left,
                "max_right": max_right,
                "latest_activity": latest_activity,
                "worker_dominant": worker_dominant,
            }

        def should_send_worker_stt(
            left_has_speech: bool,
            right_has_speech: bool,
            has_right_pcm: bool,
        ) -> tuple[bool, str, str]:
            if not has_right_pcm:
                return False, "customer_only", "no_worker_pcm"
            if (
                echo_tools_enabled
                and
                settings.simple_overlap_suppression_enabled
                and left_has_speech
                and right_has_speech
                and consecutive_both_active_packets >= settings.simple_overlap_packet_target
            ):
                return False, "simple_overlap", "simple_overlap_3_chunks"
            return True, "worker", "no_overlap_suppression"

        async def send_audio_to_stt(
            ws,
            speaker_name: str,
            item: STTChunk,
        ) -> None:
            queue_wait_ms = (time.perf_counter() - item.queued_at) * 1000
            diag_print(
                f"[STT Send Attempt] call={call_id} speaker={speaker_name} "
                f"chunk={item.chunk_number} native_seq={item.native_seq or 'none'} "
                f"rms={item.rms:.1f} speech={'yes' if item.speech else 'no'} "
                f"queue_wait_ms={queue_wait_ms:.1f}"
            )
            started_at = time.perf_counter()
            timing_log("stt_sent", speaker=speaker_name, chunk=item.chunk_number)
            log_event(
                "stt_sent",
                call_id=call_id,
                speaker=speaker_name,
                trace_id=item.trace_id,
                native_seq=item.native_seq,
                audio_chunk=item.chunk_number,
                rms=item.rms,
                speech=item.speech,
                queue_wait_ms=queue_wait_ms,
            )
            try:
                await ws.transcribe(
                    audio=pcm_to_wav_b64(item.pcm_bytes, settings.sample_rate),
                    encoding="audio/wav",
                    sample_rate=settings.sample_rate,
                )
                send_ms = (time.perf_counter() - started_at) * 1000
                diag_print(
                    f"[STT Send OK] call={call_id} speaker={speaker_name} "
                    f"chunk={item.chunk_number}"
                )
                diag_print(
                    f"[STT Timing] call={call_id} speaker={speaker_name} "
                    f"chunk={item.chunk_number} queue_wait_ms={queue_wait_ms:.1f} "
                    f"sarvam_send_ms={send_ms:.1f}"
                )
                log_event(
                    "stt_send_ok",
                    call_id=call_id,
                    speaker=speaker_name,
                    trace_id=item.trace_id,
                    native_seq=item.native_seq,
                    audio_chunk=item.chunk_number,
                    queue_wait_ms=queue_wait_ms,
                    sarvam_send_ms=send_ms,
                )
            except Exception as exc:
                diag_print(
                    f"[STT Send Error] call={call_id} speaker={speaker_name} "
                    f"chunk={item.chunk_number} type={type(exc).__name__} "
                    f"error={log_text(str(exc) or repr(exc))}"
                )

        async def stt_sender(
            ws,
            speaker_name: str,
            queue: asyncio.Queue[STTChunk | None],
        ) -> None:
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    return
                try:
                    diag_print(
                        f"[STT Queue] call={call_id} speaker={speaker_name} "
                        f"sent={item.chunk_number} native_seq={item.native_seq or 'none'} "
                        f"qsize={queue.qsize()}"
                    )
                    await send_audio_to_stt(ws, speaker_name, item)
                finally:
                    queue.task_done()

        def should_send_channel_to_stt(
            speaker_name: str,
            rms: float,
            has_pcm: bool,
            *,
            chunk_number: int,
            native_seq: int | None,
        ) -> bool:
            should_send = bool(
                has_pcm
                and (
                    settings.stt_send_rms_floor <= 0
                    or rms > settings.stt_send_rms_floor
                )
            )
            diag_print(
                f"[STT Send Gate] call={call_id} speaker={speaker_name} "
                f"decision={'send' if should_send else 'skip'} "
                f"reason={'above_floor' if should_send else 'padded_or_near_silence'} "
                f"rms={rms:.1f} floor={settings.stt_send_rms_floor}"
            )
            log_event(
                "stt_send_gate",
                call_id=call_id,
                speaker=speaker_name,
                decision="send" if should_send else "skip",
                reason="above_floor" if should_send else "padded_or_near_silence",
                native_seq=native_seq,
                audio_chunk=chunk_number,
                rms=rms,
                floor=settings.stt_send_rms_floor,
            )
            return should_send

        async def enqueue_stt_chunk(
            queue: asyncio.Queue[STTChunk | None],
            speaker_name: str,
            chunk_number: int,
            pcm_bytes: bytes,
            rms: float,
            speech: bool,
            native_seq: int | None,
            trace_id: str,
        ) -> None:
            item = STTChunk(
                chunk_number=chunk_number,
                pcm_bytes=pcm_bytes,
                rms=rms,
                speech=speech,
                queued_at=time.perf_counter(),
                native_seq=native_seq,
                trace_id=trace_id,
            )
            await queue.put(item)
            qsize = queue.qsize()
            diag_print(
                f"[STT Queue] call={call_id} speaker={speaker_name} "
                f"queued={chunk_number} native_seq={native_seq or 'none'} qsize={qsize}"
            )
            log_event(
                "stt_queued",
                call_id=call_id,
                speaker=speaker_name,
                trace_id=trace_id,
                native_seq=native_seq,
                audio_chunk=chunk_number,
                qsize=qsize,
                rms=rms,
                speech=speech,
            )
            if qsize >= stt_backlog_warning_size:
                diag_print(f"[STT Backlog Warning] call={call_id} speaker={speaker_name} qsize={qsize}")

        def classify_worker_echo(worker_text: str) -> dict[str, float | str | bool]:
            activity = recent_activity_summary()
            result: dict[str, float | str | bool] = {
                "suppress": False,
                "reason": "no_recent_customer_activity",
                "classification": "real_worker",
                "similarity": 0.0,
                "matched_text": "",
                "age_ms": 0.0,
                **activity,
            }
            if not echo_tools_enabled or not settings.echo_suppression_enabled:
                result["reason"] = "echo_suppression_disabled"
                return result

            normalized_worker = normalize_echo_text(worker_text)
            if not bool(activity["customer_recent"]):
                return result

            if bool(activity["worker_dominant"]):
                result["reason"] = "worker_dominant"
                return result

            now = time.time()
            best: dict[str, float | str] | None = None
            best_score = 0.0
            for event in recent_customer_text_events:
                age_seconds = now - float(event.get("timestamp", 0.0))
                if age_seconds < 0 or age_seconds > settings.echo_window_seconds:
                    continue
                customer_text = str(event.get("text", ""))
                if len(normalize_echo_text(customer_text)) < settings.echo_min_text_chars:
                    continue
                score = text_similarity(worker_text, customer_text)
                if score > best_score:
                    best_score = score
                    best = {
                        "text": customer_text,
                        "similarity": score,
                        "age_ms": age_seconds * 1000,
                        "left_rms": float(event.get("left_rms", 0.0)),
                        "right_rms": float(event.get("right_rms", 0.0)),
                    }

            if best:
                result.update(
                    {
                        "similarity": float(best["similarity"]),
                        "matched_text": str(best["text"]),
                        "age_ms": float(best["age_ms"]),
                    }
                )

            is_short_fragment = (
                0 < len(normalized_worker) <= settings.echo_short_text_max_chars
            )
            if (
                best_score >= settings.echo_strong_text_similarity_threshold
                and len(normalized_worker) >= settings.echo_min_text_chars
            ):
                result["suppress"] = True
                result["reason"] = "strong_text_match_with_recent_customer"
                result["classification"] = "probable_echo"
            elif (
                best_score >= settings.echo_weak_text_similarity_threshold
                and len(normalized_worker) >= settings.echo_min_text_chars
                and (bool(activity["both_recent"]) or bool(activity["customer_recent"]))
            ):
                result["suppress"] = True
                result["reason"] = "weak_text_match_with_customer_activity"
                result["classification"] = "probable_echo"
            elif is_short_fragment and (bool(activity["both_recent"]) or bool(activity["customer_recent"])):
                result["suppress"] = True
                result["reason"] = "short_worker_fragment_during_customer_audio"
                result["classification"] = "short_noise_echo"
            else:
                result["reason"] = (
                    "possible_overlap_text_different"
                    if bool(activity["both_recent"])
                    else "customer_recent_but_text_different"
                )
                result["classification"] = "possible_overlap" if bool(activity["both_recent"]) else "real_worker"

            diag_print(
                f"[Echo Candidate] call={call_id} classification={result['classification']} "
                f"worker_text={log_text(worker_text)} "
                f"matched_customer_text={log_text(str(result['matched_text']))} "
                f"similarity={float(result['similarity']):.2f} "
                f"age_ms={float(result['age_ms']):.0f} "
                f"recent_activity={result['latest_activity']} "
                f"customer_recent={'yes' if result['customer_recent'] else 'no'} "
                f"both_recent={'yes' if result['both_recent'] else 'no'} "
                f"worker_dominant={'yes' if result['worker_dominant'] else 'no'} "
                f"suppress={'yes' if result['suppress'] else 'no'} "
                f"reason={result['reason']}"
            )
            return result

        async def emit_transcript(
            text: str,
            channel_id: int,
            speaker_name: str,
            utterance_id: str,
            has_interruptions: bool = False,
            is_final: bool = False,
            metadata: dict | None = None,
        ) -> None:
            async with websocket_send_lock:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "TRANSCRIPT",
                            "call_id": call_id,
                            "utterance_id": utterance_id,
                            "speaker": speaker_name,
                            "is_final": is_final,
                            "channel": {"alternatives": [{"transcript": text}]},
                            "channel_index": [channel_id],
                            "channel_details": [{"hasInterruptions": has_interruptions}],
                            "metadata": metadata or {},
                        }
                    )
                )
            diag_print(
                f"[UI Transcript Emit] call={call_id} speaker={speaker_name} "
                f"final={'yes' if is_final else 'no'} utterance_id={utterance_id} "
                f"text={log_text(text, limit=120)}"
            )
            log_event(
                "ui_transcript_emit_final" if is_final else "ui_transcript_emit_partial",
                call_id=call_id,
                speaker=speaker_name,
                utterance_id=utterance_id,
                trace_id=(metadata or {}).get("trace_id", ""),
                text=text,
            )

        async def forward_utterance(
            text: str,
            speaker_name: str,
            has_interruptions: bool,
            reason: str,
            utterance_id: str,
            possible_bleed: bool,
            trace_id: str,
            chunk_id: int = 0,
            turn_id: int = 0,
            chunk_buffer_duration_ms: float = 0.0,
            customer_speech_started_perf: float | None = None,
        ) -> None:
            payload = {
                "body": {
                    "call_id": call_id,
                    "speaker": speaker_name,
                    "transcript": text,
                    "translated_text": text if settings.sarvam_mode == "translate" else None,
                    "timestamp": time.time(),
                    "hasInterruptions": has_interruptions,
                    "source": settings.input_source,
                    "metadata": {
                        "flush_reason": reason,
                        "utterance_id": utterance_id,
                        "trace_id": trace_id,
                        "chunk_id": chunk_id,
                        "turn_id": turn_id,
                        "possible_bleed": possible_bleed,
                    },
                }
            }

            try:
                started_at = time.perf_counter()
                if speaker_name == settings.customer_speaker_label:
                    client_timing(
                        call_id,
                        "webhook_request_started",
                        chunk_id=chunk_id,
                        turn_id=turn_id,
                    )
                diag_print(
                    f"[Live Assist Forward] call={call_id} speaker={speaker_name} "
                    f"utterance_id={utterance_id}"
                )
                trace_id = str(payload["body"]["metadata"].get("trace_id", ""))
                log_event(
                    "live_assist_background_started",
                    call_id=call_id,
                    speaker=speaker_name,
                    utterance_id=utterance_id,
                    trace_id=trace_id,
                    text=text,
                )
                timing_log("live_assist_start", speaker=speaker_name, utterance_id=utterance_id)
                response = await feedback_client.post(
                    settings.live_feedback_webhook_url,
                    json=payload,
                )
                webhook_duration_ms = (time.perf_counter() - started_at) * 1000
                if speaker_name == settings.customer_speaker_label:
                    client_timing(
                        call_id,
                        "webhook_response_received",
                        chunk_id=chunk_id,
                        turn_id=turn_id,
                        duration_ms=f"{webhook_duration_ms:.1f}",
                    )
                response.raise_for_status()
                result = response.json()
                action_result = result.get("action_result") if isinstance(result, dict) else {}
                answer = (action_result or {}).get("answer", "") if isinstance(action_result, dict) else ""
                status = str(result.get("status") if isinstance(result, dict) else "")
                action_status = str((action_result or {}).get("status", "") if isinstance(action_result, dict) else "")
                error = (
                    ((action_result or {}).get("metadata") or {}).get("error", "")
                    if isinstance(action_result, dict)
                    else ""
                )
                action_metadata = (
                    (action_result or {}).get("metadata") or {}
                    if isinstance(action_result, dict)
                    else {}
                )
                if speaker_name == settings.customer_speaker_label:
                    enriched_query = str(action_metadata.get("enriched_query") or "").strip()
                    api_summary_timing(
                        call_id,
                        chunk_id=chunk_id,
                        turn_id=turn_id,
                        workflow_ms=f"{float(action_metadata.get('workflow_duration_ms') or 0.0):.1f}",
                        query=log_text(enriched_query) if enriched_query else "NO MATCH",
                    )
                    top_chunks = action_metadata.get("rag_top_chunks") or []
                    api_summary_timing(
                        call_id,
                        chunk_id=chunk_id,
                        turn_id=turn_id,
                        event="retrieved_chunks",
                        chunk_1=log_text(str(top_chunks[0]), limit=180) if len(top_chunks) > 0 else None,
                        chunk_2=log_text(str(top_chunks[1]), limit=180) if len(top_chunks) > 1 else None,
                        chunk_3=log_text(str(top_chunks[2]), limit=180) if len(top_chunks) > 2 else None,
                    )
                    if customer_speech_started_perf is not None:
                        client_timing(
                            call_id,
                            "response_processed",
                            chunk_id=chunk_id,
                            turn_id=turn_id,
                            duration_ms=f"{(time.perf_counter() - customer_speech_started_perf) * 1000:.1f}",
                            chunk_buffer_ms=f"{chunk_buffer_duration_ms:.1f}",
                        )
                if speaker_name == settings.customer_speaker_label and answer:
                    async with websocket_send_lock:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "LIVE_ASSIST_RESULT",
                                    "call_id": call_id,
                                    "speaker": speaker_name,
                                    "utterance": text,
                                    "utterance_id": utterance_id,
                                    "result": result,
                                }
                            )
                        )
                elif (
                    speaker_name == settings.customer_speaker_label
                    and (status == "workflow_failed" or action_status == "workflow_failed" or error)
                ):
                    async with websocket_send_lock:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "LIVE_ASSIST_ERROR",
                                    "call_id": call_id,
                                    "speaker": speaker_name,
                                    "utterance": text,
                                    "utterance_id": utterance_id,
                                    "error": error or "Live Assist workflow failed.",
                                }
                            )
                        )
                duration_ms = (time.perf_counter() - started_at) * 1000
                diag_print(
                    f"[Live Assist Forward Done] call={call_id} speaker={speaker_name} "
                    f"utterance_id={utterance_id} status={result.get('status') if isinstance(result, dict) else 'unknown'} "
                    f"answer_len={len(answer)} duration_ms={duration_ms:.1f}"
                )
                log_event(
                    "live_assist_background_done",
                    call_id=call_id,
                    speaker=speaker_name,
                    utterance_id=utterance_id,
                    trace_id=trace_id,
                    duration_ms=duration_ms,
                    answer=answer,
                )
                timing_log("live_assist_done", speaker=speaker_name, utterance_id=utterance_id)
            except Exception as exc:
                error_detail = f"{type(exc).__name__}: {exc!r}"
                error_payload = {
                    "type": "LIVE_ASSIST_ERROR",
                    "call_id": call_id,
                    "speaker": speaker_name,
                    "utterance": text,
                    "utterance_id": utterance_id,
                    "error": error_detail,
                }
                diag_print(f"[Webhook Error] {error_payload}")
                async with websocket_send_lock:
                    await websocket.send(json.dumps(error_payload))

        async def flush_utterance(
            state: UtteranceState,
            speaker_name: str,
            channel_id: int,
            reason: str,
        ) -> None:
            nonlocal customer_speech_packet_count, worker_speech_packet_count

            async with utterance_lock:
                utterance_id = state.utterance_id or ensure_utterance_id(state, speaker_name)
                chunk_id = state.chunk_id or 0
                turn_id = state.turn_id or chunk_id
                final_text = build_final_transcript(state.transcript_buffer)
                has_interruptions = state.has_interruptions
                possible_bleed = state.possible_bleed
                customer_speech_started_perf = state.utterance_started_perf
                chunk_buffer_duration_ms = (
                    (time.perf_counter() - customer_speech_started_perf) * 1000
                    if customer_speech_started_perf is not None
                    else 0.0
                )
                state.reset()
                if speaker_name == settings.customer_speaker_label:
                    customer_speech_packet_count = 0
                else:
                    worker_speech_packet_count = 0

            if not final_text:
                return

            if speaker_name == settings.customer_speaker_label:
                client_timing(call_id, "asr_final_received", chunk_id=chunk_id, turn_id=turn_id)
                client_timing(
                    call_id,
                    "chunk_flushed",
                    chunk_id=chunk_id,
                    turn_id=turn_id,
                    duration_ms=f"{chunk_buffer_duration_ms:.1f}",
                    text=log_text(final_text),
                )
            else:
                diag_print(
                    f"[Text Chunk Final] call={call_id} speaker={speaker_name} "
                    f"utterance_id={utterance_id} reason={reason} "
                    f"possible_bleed={possible_bleed} text={log_text(final_text)}"
                )
            trace_id = make_trace_id(utterance_id)
            timing_log("final_flush", speaker=speaker_name, utterance_id=utterance_id, reason=reason)
            log_event(
                "text_chunk_final_flush",
                call_id=call_id,
                speaker=speaker_name,
                utterance_id=utterance_id,
                trace_id=trace_id,
                reason=reason,
                possible_bleed=possible_bleed,
                text=final_text,
            )

            await emit_transcript(
                final_text,
                channel_id=channel_id,
                speaker_name=speaker_name,
                utterance_id=utterance_id,
                has_interruptions=has_interruptions,
                is_final=True,
                metadata={
                    "flush_reason": reason,
                    "possible_bleed": possible_bleed,
                    "trace_id": trace_id,
                },
            )
            # Keep the UI transcript ahead of the heavier Live Assist/RAG path.
            track_background_task(
                asyncio.create_task(
                    forward_utterance(
                        final_text,
                        speaker_name=speaker_name,
                        has_interruptions=has_interruptions,
                        reason=reason,
                        utterance_id=utterance_id,
                        possible_bleed=possible_bleed,
                        trace_id=trace_id,
                        chunk_id=chunk_id,
                        turn_id=turn_id,
                        chunk_buffer_duration_ms=chunk_buffer_duration_ms,
                        customer_speech_started_perf=customer_speech_started_perf,
                    )
                )
            )
            if speaker_name == settings.customer_speaker_label:
                client_timing(
                    call_id,
                    "forward_utterance_task_created",
                    chunk_id=chunk_id,
                    turn_id=turn_id,
                )

        # Stage 0 probe instance (read-only, never modifies audio)
        _aec_probe: _AecProbe | None = _AecProbe(call_id=call_id) if _AEC_PROBE_ENABLED else None
        # Stage 2 NLMS instance (replaces right channel when AEC_NLMS=1)
        _aec_nlms: _AecNlms | None = _AecNlms(call_id=call_id) if _AEC_NLMS_ENABLED else None

        async def process_audio_chunk(
            chunk: bytes,
            native_seq: int | None = None,
            native_meta: dict | None = None,
        ) -> None:
            nonlocal customer_speech_packet_count, worker_speech_packet_count
            nonlocal processed_audio_chunks
            nonlocal last_left_rms, last_right_rms
            nonlocal consecutive_both_active_packets

            left, right = split_stereo_to_mono(chunk)

            # ── Stage 0: delay probe (logging only, audio unchanged) ────────
            if _aec_probe is not None:
                _aec_probe.feed(left, right)
                if _aec_probe.stable_lag is not None and _aec_nlms is not None:
                    _aec_nlms.update_delay(_aec_probe.stable_lag)
            # ── Stage 2: block NLMS (replaces right when AEC_NLMS=1) ───────
            if _aec_nlms is not None:
                right, _nlms_log = _aec_nlms.process(left, right)
                if _nlms_log:
                    print(_nlms_log, flush=True)
            # ─────────────────────────────────────────────────────────────

            left_rms = calculate_rms(left)
            right_rms = calculate_rms(right)
            last_left_rms = left_rms
            last_right_rms = right_rms
            processed_audio_chunks += 1
            timing_log(
                "chunk_processed",
                chunk=processed_audio_chunks,
                native_seq=native_seq or "none",
                bytes=len(chunk),
            )
            native_mic_rms = 0.0
            try:
                native_mic_rms = float((native_meta or {}).get("mic_rms", 0.0) or 0.0)
            except (TypeError, ValueError):
                native_mic_rms = 0.0
            left_has_speech = bool(left and left_rms > settings.energy_threshold)
            right_has_speech = bool(right and right_rms > settings.energy_threshold)
            if left_has_speech and right_has_speech:
                consecutive_both_active_packets += 1
            else:
                consecutive_both_active_packets = 0
            remember_audio_activity(
                processed_audio_chunks,
                left_rms,
                right_rms,
                left_has_speech,
                right_has_speech,
            )
            diag_print(
                f"[Audio Energy] call={call_id} chunk={processed_audio_chunks} "
                f"speaker={settings.customer_speaker_label} channel=left "
                f"rms={left_rms:.1f} threshold={settings.energy_threshold} "
                f"speech={'yes' if left_has_speech else 'no'} pcm_bytes={len(left)}"
            )
            log_event(
                "audio_chunk_processed",
                call_id=call_id,
                native_seq=native_seq,
                audio_chunk=processed_audio_chunks,
                bytes=len(chunk),
                customer_rms=left_rms,
                worker_rms=right_rms,
                native_mic_rms=native_mic_rms,
                customer_speech=left_has_speech,
                worker_speech=right_has_speech,
            )
            if native_mic_rms > 0.003 and right_rms <= settings.stt_send_rms_floor:
                diag_print(
                    f"[Worker Audio Warning] call={call_id} native_mic_seen=yes "
                    f"backend_right_rms={right_rms:.1f} native_mic_rms={native_mic_rms:.5f} "
                    f"action=check_mixer_or_pcm"
                )
                log_event(
                    "worker_audio_warning",
                    call_id=call_id,
                    native_seq=native_seq,
                    audio_chunk=processed_audio_chunks,
                    native_mic_seen=True,
                    backend_right_rms=right_rms,
                    native_mic_rms=native_mic_rms,
                    action="check_mixer_or_pcm",
                )
            diag_print(
                f"[Audio Energy] call={call_id} chunk={processed_audio_chunks} "
                f"speaker={settings.worker_speaker_label} channel=right "
                f"rms={right_rms:.1f} threshold={settings.energy_threshold} "
                f"speech={'yes' if right_has_speech else 'no'} pcm_bytes={len(right)}"
            )
            if echo_tools_enabled and settings.simple_overlap_suppression_enabled:
                diag_print(
                    f"[Simple Overlap] call={call_id} both_active_count={consecutive_both_active_packets} "
                    f"target={settings.simple_overlap_packet_target} "
                    f"left_rms={left_rms:.1f} right_rms={right_rms:.1f}"
                )

            if left:
                if left_has_speech:
                    customer_speech_packet_count += 1
                else:
                    customer_speech_packet_count = 0
                if should_send_channel_to_stt(
                    settings.customer_speaker_label,
                    left_rms,
                    bool(left),
                    chunk_number=processed_audio_chunks,
                    native_seq=native_seq,
                ):
                    await enqueue_stt_chunk(
                        customer_stt_queue,
                        settings.customer_speaker_label,
                        processed_audio_chunks,
                        left,
                        left_rms,
                        left_has_speech,
                        native_seq,
                        "",
                    )

            if right:
                if right_has_speech:
                    worker_speech_packet_count += 1
                else:
                    worker_speech_packet_count = 0
                send_worker = False
                if should_send_channel_to_stt(
                    settings.worker_speaker_label,
                    right_rms,
                    bool(right),
                    chunk_number=processed_audio_chunks,
                    native_seq=native_seq,
                ):
                    send_worker, gate_classification, gate_reason = should_send_worker_stt(
                        left_has_speech,
                        right_has_speech,
                        has_right_pcm=bool(right),
                    )
                    diag_print(
                        f"[Worker STT Gate] call={call_id} "
                        f"decision={'send' if send_worker else 'suppress'} "
                        f"classification={gate_classification} reason={gate_reason} "
                        f"left_rms={left_rms:.1f} right_rms={right_rms:.1f} "
                        f"both_packets={consecutive_both_active_packets}"
                    )
                if send_worker:
                    await enqueue_stt_chunk(
                        worker_stt_queue,
                        settings.worker_speaker_label,
                        processed_audio_chunks,
                        right,
                        right_rms,
                        right_has_speech,
                        native_seq,
                        "",
                    )

            flush_customer = False
            flush_worker = False
            customer_flush_reason = "customer_silence"
            worker_flush_reason = "worker_silence"
            async with utterance_lock:
                if customer_state.waiting_for_silence:
                    if left_rms < settings.energy_threshold:
                        customer_state.consecutive_low_energy_packets += 1
                    else:
                        customer_state.consecutive_low_energy_packets = 0
                    diag_print(
                        f"[Silence Count] call={call_id} speaker={settings.customer_speaker_label} "
                        f"rms={left_rms:.1f} active={'yes' if left_has_speech else 'no'} "
                        f"low_packets={customer_state.consecutive_low_energy_packets} "
                        f"target={settings.low_energy_packet_target}"
                    )

                    if worker_speech_packet_count >= settings.interruption_packet_threshold:
                        customer_state.has_interruptions = True

                    if customer_state.consecutive_low_energy_packets >= settings.low_energy_packet_target:
                        flush_customer = True
                    elif (
                        customer_state.utterance_started_at
                        and time.time() - customer_state.utterance_started_at >= settings.max_utterance_seconds
                    ):
                        flush_customer = True
                        customer_flush_reason = "customer_max_duration"

                if worker_state.waiting_for_silence:
                    if right_rms < settings.energy_threshold:
                        worker_state.consecutive_low_energy_packets += 1
                    else:
                        worker_state.consecutive_low_energy_packets = 0
                    diag_print(
                        f"[Silence Count] call={call_id} speaker={settings.worker_speaker_label} "
                        f"rms={right_rms:.1f} active={'yes' if right_has_speech else 'no'} "
                        f"low_packets={worker_state.consecutive_low_energy_packets} "
                        f"target={settings.low_energy_packet_target}"
                    )

                    if worker_state.consecutive_low_energy_packets >= settings.low_energy_packet_target:
                        flush_worker = True
                    elif (
                        worker_state.utterance_started_at
                        and time.time() - worker_state.utterance_started_at >= settings.max_utterance_seconds
                    ):
                        flush_worker = True
                        worker_flush_reason = "worker_max_duration"

            if flush_customer:
                await flush_utterance(
                    customer_state,
                    speaker_name=settings.customer_speaker_label,
                    channel_id=0,
                    reason=customer_flush_reason,
                )
            if flush_worker:
                await flush_utterance(
                    worker_state,
                    speaker_name=settings.worker_speaker_label,
                    channel_id=1,
                    reason=worker_flush_reason,
                )

        async def receive_audio() -> None:
            nonlocal audio_buffer, last_audio_recv_time, call_id, received_audio_frames
            nonlocal expected_native_seq

            async for message in websocket:
                if isinstance(message, str):
                    audio_meta = parse_audio_chunk_meta(message)
                    if audio_meta:
                        pending_audio_meta.append(audio_meta)
                        diag_print(
                            f"[Audio Meta] call={call_id} native_seq={audio_meta.get('chunk_index')} "
                            f"bytes={audio_meta.get('bytes')} sent_at_ms={audio_meta.get('sent_at_ms')}"
                        )
                        log_event(
                            "native_audio_meta_received",
                            call_id=call_id,
                            native_seq=audio_meta.get("chunk_index"),
                            bytes=audio_meta.get("bytes"),
                            sent_at_ms=audio_meta.get("sent_at_ms"),
                            mic_rms=audio_meta.get("mic_rms"),
                            system_rms=audio_meta.get("system_rms"),
                        )
                        continue
                    incoming_call_id = extract_call_id_from_message(message)
                    if incoming_call_id and incoming_call_id != call_id:
                        call_id = incoming_call_id
                    continue

                if not isinstance(message, (bytes, bytearray)):
                    continue

                last_audio_recv_time = time.perf_counter()
                audio_buffer += message
                received_audio_frames += 1
                audio_meta = pending_audio_meta.popleft() if pending_audio_meta else {}
                native_seq_value = audio_meta.get("chunk_index")
                try:
                    native_seq = int(native_seq_value) if native_seq_value is not None else None
                except (TypeError, ValueError):
                    native_seq = None
                if native_seq is not None:
                    if expected_native_seq is None:
                        expected_native_seq = native_seq
                    if native_seq != expected_native_seq:
                        missing = native_seq - expected_native_seq
                        diag_print(
                            f"[Audio Frame Gap] call={call_id} expected={expected_native_seq} "
                            f"got={native_seq} missing={missing if missing > 0 else 'out_of_order'}"
                        )
                        log_event(
                            "audio_frame_gap",
                            call_id=call_id,
                            native_seq=native_seq,
                            backend_frame=received_audio_frames,
                            expected_native_seq=expected_native_seq,
                            missing=missing if missing > 0 else "out_of_order",
                        )
                    diag_print(
                        f"[Audio Frame] call={call_id} native_seq={native_seq} "
                        f"backend_frame={received_audio_frames} "
                        f"gap={'no' if native_seq == expected_native_seq else 'yes'} "
                        f"bytes={len(message)}"
                    )
                    log_event(
                        "audio_frame_received",
                        call_id=call_id,
                        native_seq=native_seq,
                        backend_frame=received_audio_frames,
                        gap=native_seq != expected_native_seq,
                        bytes=len(message),
                        mic_rms=audio_meta.get("mic_rms"),
                        system_rms=audio_meta.get("system_rms"),
                    )
                    expected_native_seq = native_seq + 1
                else:
                    diag_print(
                        f"[Audio Frame] call={call_id} native_seq=none "
                        f"backend_frame={received_audio_frames} bytes={len(message)}"
                    )
                    log_event(
                        "audio_frame_received",
                        call_id=call_id,
                        backend_frame=received_audio_frames,
                        bytes=len(message),
                    )
                timing_log(
                    "audio_received",
                    frame=received_audio_frames,
                    native_seq=native_seq or "none",
                    bytes=len(message),
                    buffer_bytes=len(audio_buffer),
                )
                will_process = len(audio_buffer) >= buffer_target
                diag_print(
                    f"[Audio Receive] call={call_id} frame={received_audio_frames} "
                    f"message_bytes={len(message)} buffer_bytes={len(audio_buffer)} "
                    f"target_bytes={buffer_target} process={'yes' if will_process else 'no'}"
                )
                while len(audio_buffer) >= buffer_target:
                    chunk = audio_buffer[:buffer_target]
                    audio_buffer = audio_buffer[buffer_target:]
                    await process_audio_chunk(chunk, native_seq=native_seq, native_meta=audio_meta)

        async def send_text(
            ws,
            state: UtteranceState,
            channel_id: int,
            speaker_name: str,
        ) -> None:
            nonlocal next_customer_chunk_id
            async for message in ws:
                recv_time = time.perf_counter()
                event_type = getattr(message, "type", None)
                text = getattr(getattr(message, "data", None), "transcript", "") or ""
                diag_print(
                    f"[STT Event] call={call_id} speaker={speaker_name} "
                    f"type={event_type} transcript_len={len(text.strip())}"
                )
                if event_type != "data":
                    continue

                text = text.strip()
                if not text:
                    continue

                diag_print(
                    f"[STT Text] call={call_id} speaker={speaker_name} "
                    f"final=no text={log_text(text)}"
                )
                timing_log("stt_text_received", speaker=speaker_name, channel=channel_id)

                if echo_tools_enabled and speaker_name == settings.customer_speaker_label:
                    remember_customer_text(text)

                possible_bleed = (
                    echo_tools_enabled
                    and settings.echo_suppression_enabled
                    and speaker_name == settings.worker_speaker_label
                    and last_left_rms >= max(settings.energy_threshold * 2, last_right_rms * 2.5)
                )
                echo_decision = (
                    classify_worker_echo(text)
                    if echo_tools_enabled
                    and settings.echo_suppression_enabled
                    and speaker_name == settings.worker_speaker_label
                    else None
                )
                if echo_decision and bool(echo_decision.get("suppress")):
                    diag_print(
                        f"[Echo Suppressed] call={call_id} speaker={speaker_name} "
                        f"reason={echo_decision['reason']} "
                        f"classification={echo_decision['classification']} "
                        f"similarity={float(echo_decision['similarity']):.2f} "
                        f"age_ms={float(echo_decision['age_ms']):.0f} "
                        f"text={log_text(text)}"
                    )
                    continue
                if echo_decision:
                    diag_print(
                        f"[Echo Kept] call={call_id} speaker={speaker_name} "
                        f"reason={echo_decision['reason']} "
                        f"classification={echo_decision['classification']} "
                        f"similarity={float(echo_decision['similarity']):.2f} "
                        f"text={log_text(text)}"
                    )

                if possible_bleed:
                    diag_print(
                        f"[Bleed Suspected] call={call_id} speaker={speaker_name} "
                        f"left_rms={last_left_rms:.1f} right_rms={last_right_rms:.1f} "
                        f"text={log_text(text)} action=keep_no_text_match"
                    )

                if last_audio_recv_time:
                    latency = recv_time - last_audio_recv_time
                    latencies.append(latency)

                async with utterance_lock:
                    utterance_id = ensure_utterance_id(state, speaker_name)
                    state.possible_bleed = state.possible_bleed or possible_bleed
                    transcript_changed = merge_transcript(state.transcript_buffer, text)
                    state.waiting_for_silence = True
                    if transcript_changed:
                        state.consecutive_low_energy_packets = 0
                    state.last_transcript_at = time.time()
                    if state.utterance_started_at is None:
                        state.utterance_started_at = state.last_transcript_at
                        state.utterance_started_perf = recv_time
                        state.has_interruptions = False
                        if speaker_name == settings.customer_speaker_label:
                            next_customer_chunk_id += 1
                            state.chunk_id = next_customer_chunk_id
                            state.turn_id = next_customer_chunk_id
                            client_timing(
                                call_id,
                                "chunk_buffer_started",
                                chunk_id=state.chunk_id,
                                turn_id=state.turn_id,
                            )
                log_event(
                    "stt_text_partial_received",
                    call_id=call_id,
                    speaker=speaker_name,
                    utterance_id=utterance_id,
                    text=text,
                )

                if diagnostic_mode:
                    try:
                        await emit_transcript(
                            text,
                            channel_id=channel_id,
                            speaker_name=speaker_name,
                            utterance_id=utterance_id,
                            is_final=False,
                            metadata={"possible_bleed": possible_bleed},
                        )
                    except Exception:
                        pass

        receive_task = asyncio.create_task(receive_audio())
        customer_text_task = asyncio.create_task(
            send_text(ws_customer, customer_state, 0, settings.customer_speaker_label)
        )
        worker_text_task = asyncio.create_task(
            send_text(ws_worker, worker_state, 1, settings.worker_speaker_label)
        )
        customer_stt_task = asyncio.create_task(
            stt_sender(ws_customer, settings.customer_speaker_label, customer_stt_queue)
        )
        worker_stt_task = asyncio.create_task(
            stt_sender(ws_worker, settings.worker_speaker_label, worker_stt_queue)
        )
        connection_tasks = {
            receive_task,
            customer_text_task,
            worker_text_task,
            customer_stt_task,
            worker_stt_task,
        }

        try:
            done, _ = await asyncio.wait(connection_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            for task in (receive_task, customer_text_task, worker_text_task):
                if not task.done():
                    task.cancel()
            if audio_buffer:
                await process_audio_chunk(audio_buffer)
            await customer_stt_queue.put(None)
            await worker_stt_queue.put(None)
            await asyncio.gather(customer_stt_task, worker_stt_task, return_exceptions=True)
            await flush_utterance(
                customer_state,
                speaker_name=settings.customer_speaker_label,
                channel_id=0,
                reason="connection_closed",
            )
            await flush_utterance(
                worker_state,
                speaker_name=settings.worker_speaker_label,
                channel_id=1,
                reason="connection_closed",
            )
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)


async def main_async() -> None:
    async with websockets.serve(handle_client, settings.stream_host, settings.stream_port):
        debug_message = f"Stream server running at ws://{settings.stream_host}:{settings.stream_port}"
        print(debug_message, flush=True)
        await asyncio.Future()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
