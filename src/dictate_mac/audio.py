"""Audio capture and silence trimming for dictate-mac.

* Capture microphone audio at 16 kHz mono float32 via PortAudio
  (``sounddevice.InputStream``).
* Buffer samples in a thread-safe ``numpy.ndarray`` while recording.
* On stop, apply ``silero-vad`` to drop leading/trailing silence and to
  shorten internal pauses longer than 600 ms, so the recognizer sees only
  speech. If the recording contains no speech, the trimmed result is
  empty — callers MUST check before transcribing.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger("dictate_mac.audio")

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
BLOCK_SIZE = 4000  # ~250 ms at 16 kHz


class Recorder:
    """Captures audio from the default input device into a single buffer.

    Usage::

        rec = Recorder()
        rec.start()
        ...
        rec.stop() -> np.ndarray  # may be empty if start() was never called
    """

    def __init__(self) -> None:
        self._stream: Optional[sd.InputStream] = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,  # noqa: ARG001 — PortAudio passes CData
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.debug("input stream status: %s", status)
        with self._lock:
            if self._recording:
                self._chunks.append(indata.copy().reshape(-1))

    def _open_stream(self) -> sd.InputStream:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE,
            callback=self._callback,
        )
        stream.start()
        return stream

    def start(self) -> None:
        if self._recording:
            raise RuntimeError("Recorder already started")
        logger.info(
            "recording started (rate=%d ch=%d blocksize=%d)",
            SAMPLE_RATE,
            CHANNELS,
            BLOCK_SIZE,
        )
        with self._lock:
            self._chunks.clear()
        try:
            self._stream = self._open_stream()
        except sd.PortAudioError as exc:
            # PortAudio snapshots the device list at Pa_Initialize. If
            # the topology changed since (virtual devices appearing or
            # disappearing, coreaudiod restart, default input switch),
            # Pa_OpenStream on the stale default fails with
            # paInternalError (-9986) until the process restarts.
            # Re-initializing PortAudio refreshes the snapshot, so we
            # retry once instead of failing every recording.
            try:
                default_input = sd.query_devices(kind="input")
            except Exception:  # noqa: BLE001
                default_input = "<unavailable>"
            logger.warning(
                "input stream open failed (%s); default input now %r — "
                "re-initializing PortAudio and retrying once",
                exc,
                default_input,
            )
            sd._terminate()
            sd._initialize()
            self._stream = self._open_stream()
        self._recording = True

    def stop(self) -> np.ndarray:
        if not self._recording:
            raise RuntimeError("Recorder not started")
        assert self._stream is not None
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._recording = False
        with self._lock:
            if not self._chunks:
                audio = np.zeros(0, dtype=np.float32)
            else:
                audio = np.concatenate(self._chunks).astype(np.float32)
            self._chunks.clear()
        duration = audio.size / SAMPLE_RATE
        logger.info(
            "recording stopped — %d samples (%.2fs, peak=%.3f)",
            audio.size,
            duration,
            float(np.abs(audio).max()) if audio.size else 0.0,
        )
        return audio


# ---------------------------------------------------------------------------
# Silence trimming with silero-vad
# ---------------------------------------------------------------------------


_vad_model = None
_vad_lock = threading.Lock()


def _get_vad_model():
    """Lazy singleton — the silero-vad ONNX session is loaded once."""
    global _vad_model
    if _vad_model is None:
        with _vad_lock:
            if _vad_model is None:
                from silero_vad import load_silero_vad

                logger.info("loading silero-vad model …")
                _vad_model = load_silero_vad()
                logger.info("silero-vad ready")
    return _vad_model


def trim_silence(
    audio: np.ndarray,
    *,
    min_speech_ms: int = 300,
    min_silence_ms: int = 100,
    speech_pad_ms: int = 300,
    max_internal_pause_ms: int = 600,
) -> np.ndarray:
    """Return ``audio`` with leading/trailing silence removed.

    Internal pauses longer than ``max_internal_pause_ms`` are shortened
    to exactly ``max_internal_pause_ms`` (keeping the pause symmetrically
    around its midpoint); shorter pauses are left untouched.

    Returns an empty array if no speech is detected. ``audio`` must be
    float32 mono at 16 kHz.
    """
    if audio.size == 0:
        return audio
    from silero_vad import get_speech_timestamps

    model = _get_vad_model()
    timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=False,
    )
    if not timestamps:
        logger.info("vad: no speech detected in recording")
        return np.zeros(0, dtype=np.float32)

    # Timestamps already include speech_pad_ms on each side, so the raw
    # gap between two segments is the original pause minus the pads. Cap
    # the gap so the TOTAL internal pause never exceeds
    # max_internal_pause_ms.
    max_gap = max(
        0, (max_internal_pause_ms - 2 * speech_pad_ms) * SAMPLE_RATE // 1000
    )
    parts = []
    prev_end = None
    for ts in timestamps:
        if prev_end is not None:
            gap = audio[prev_end:ts["start"]]
            if gap.size > max_gap:
                half = max_gap // 2
                gap = np.concatenate([gap[:half], gap[gap.size - (max_gap - half):]])
            parts.append(gap)
        parts.append(audio[ts["start"]:ts["end"]])
        prev_end = ts["end"]
    trimmed = np.concatenate(parts)
    logger.info(
        "vad: trimmed %.2fs -> %.2fs (speech segments=%d)",
        audio.size / SAMPLE_RATE,
        trimmed.size / SAMPLE_RATE,
        len(timestamps),
    )
    return trimmed
