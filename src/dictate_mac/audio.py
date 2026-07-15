"""Audio capture and silence trimming for dictate-mac.

Phase 2 responsibilities:

* Capture microphone audio at 16 kHz mono float32 via PortAudio
  (``sounddevice.InputStream``).
* Buffer samples in a thread-safe ``numpy.ndarray`` while recording.
* On stop, apply ``silero-vad`` to drop leading/trailing silence so the
  recognizer sees only speech. If the recording contains no speech, the
  trimmed result is empty — callers MUST check before transcribing.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger("dictate_mac.audio")

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
BLOCK_SIZE = 4000  # ~250 ms at 16 kHz


@dataclass(frozen=True)
class AudioConfig:
    samplerate: int = SAMPLE_RATE
    channels: int = CHANNELS
    dtype: str = DTYPE
    blocksize: int = BLOCK_SIZE


class Recorder:
    """Captures audio from the default input device into a single buffer.

    Usage::

        rec = Recorder()
        rec.start()
        ...
        rec.stop() -> np.ndarray  # may be empty if start() was never called
    """

    def __init__(self, config: AudioConfig | None = None) -> None:
        self._config = config or AudioConfig()
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

    def start(self) -> None:
        if self._recording:
            raise RuntimeError("Recorder already started")
        cfg = self._config
        logger.info(
            "recording started (rate=%d ch=%d blocksize=%d)",
            cfg.samplerate,
            cfg.channels,
            cfg.blocksize,
        )
        with self._lock:
            self._chunks.clear()
        self._stream = sd.InputStream(
            samplerate=cfg.samplerate,
            channels=cfg.channels,
            dtype=cfg.dtype,
            blocksize=cfg.blocksize,
            callback=self._callback,
        )
        self._stream.start()
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
        duration = audio.size / self._config.samplerate
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
    speech_pad_ms: int = 100,
) -> np.ndarray:
    """Return ``audio`` with leading/trailing silence removed.

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

    start = timestamps[0]["start"]
    end = timestamps[-1]["end"]
    trimmed = audio[start:end].copy()
    logger.info(
        "vad: trimmed %.2fs -> %.2fs (speech segments=%d)",
        audio.size / SAMPLE_RATE,
        trimmed.size / SAMPLE_RATE,
        len(timestamps),
    )
    return trimmed


def has_speech(audio: np.ndarray) -> bool:
    """Cheap energy-based pre-check before invoking the VAD model."""
    if audio.size == 0:
        return False
    rms = float(np.sqrt(np.mean(audio**2)))
    return rms >= 1e-3
