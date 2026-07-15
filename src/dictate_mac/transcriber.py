"""mlx-whisper lazy singleton for dictate-mac.

Phase 3 responsibilities:

* Load ``mlx-community/whisper-large-v3-turbo`` exactly once and keep it
  resident in RAM for the lifetime of the daemon process (the user
  explicitly requested "loaded on first recognition, stays in RAM until
  reboot").
* Provide a thin ``transcribe(audio, language)`` wrapper that:
  - skips work on empty input (returns ``""``),
  - calls mlx-whisper with the requested language (or auto-detect when
    ``language == "auto"``), ``task="transcribe"``, ``fp16=True``,
  - strips leading/trailing whitespace before returning.
* ``ensure_warm_async()`` kicks off a background thread that downloads
  the model if needed and loads it into RAM — the daemon calls this at
  startup so the first user-driven recognition doesn't pay a 30-60 s
  download penalty.

Phase 9: progress callbacks for the menu bar UI.

* ``ensure_warm_async(on_phase=on_phase)`` accepts a callback invoked
  with one of the phase strings ``"downloading"``, ``"loading"``,
  ``"ready"``, or ``"error"`` so the menu bar can render
  ``Status: Downloading…`` / ``Status: Loading…`` / ``Status: Ready``.
* ``warm_vad()`` pre-loads the silero-vad ONNX session so the first
  recording doesn't pay the ~0.5 s load.

Phase 15: language is now a per-call parameter, no longer a module-level
constant. The caller (typically :class:`dictate_mac.state.DictationMachine`)
passes the language resolved from persisted settings (menubar entry
point) or from the ``--language`` CLI flag (daemon entry point).
``"auto"`` is mapped to ``language=None`` at the mlx-whisper boundary,
which triggers the encoder's first-30-second language detection path.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger("dictate_mac.transcriber")

MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
TASK = "transcribe"

WarmPhase = str  # "downloading" | "loading" | "ready" | "error"
WarmCallback = Callable[[WarmPhase, str], None]


_model = None
_model_lock = threading.Lock()
_first_call_done = False
_local_path_cache: Optional[str] = None

_warmup_thread: Optional[threading.Thread] = None
_warmup_lock = threading.Lock()
_warmup_callback: Optional[WarmCallback] = None


def _emit(phase: WarmPhase, detail: str = "") -> None:
    cb = _warmup_callback
    if cb is None:
        return
    try:
        cb(phase, detail)
    except Exception:  # noqa: BLE001
        logger.exception("warmup callback raised")


def _do_warmup_blocking() -> None:
    """Run the actual download + load. Runs on the background thread.

    Failures are logged but never propagated — the daemon must keep
    running so a network blip at startup doesn't kill it. The next
    ``transcribe()`` call will retry via ``_load_model``.
    """
    try:
        if not is_model_cached():
            _emit("downloading", MODEL_REPO)
            logger.info("downloading %s …", MODEL_REPO)
            _local_model_path()  # falls through to the network path
        _emit("loading", "")
        _load_model()
        _emit("ready", "")
    except Exception as exc:  # noqa: BLE001
        logger.exception("background warmup failed; the next transcribe() will retry")
        _emit("error", str(exc))


def ensure_warm_async(
    on_phase: Optional[WarmCallback] = None,
) -> threading.Thread:
    """Start a background warmup if one isn't already running.

    Idempotent: repeated calls while a warmup thread is alive just
    return the same thread. Used at daemon startup so that the first
    user-driven recording doesn't pay the model-load + (if uncached)
    download cost.

    ``on_phase`` (Phase 9) is invoked with the current lifecycle phase
    string. The callback is stored globally and consumed by the
    background thread; the caller does not have to keep a reference
    alive for the duration of the warmup.
    """
    global _warmup_thread, _warmup_callback
    with _warmup_lock:
        if on_phase is not None:
            _warmup_callback = on_phase
        if _warmup_thread is not None and _warmup_thread.is_alive():
            return _warmup_thread
        _warmup_thread = threading.Thread(
            target=_do_warmup_blocking,
            name="asr-warmup",
            daemon=True,
        )
        _warmup_thread.start()
    return _warmup_thread


def _local_model_path() -> str:
    """Return the local snapshot path for ``MODEL_REPO``.

    Resolution is fully offline after the first call: we ask
    ``snapshot_download`` with ``local_files_only=True``, which never
    hits the network. The result is memoized so we don't repeat even
    the disk-side cache lookup. The non-cached path is only used as a
    fallback on the very first run, before any warmup / model load.
    """
    global _local_path_cache
    if _local_path_cache is not None:
        return _local_path_cache

    from huggingface_hub import snapshot_download

    try:
        _local_path_cache = snapshot_download(
            repo_id=MODEL_REPO, local_files_only=True
        )
    except Exception:  # noqa: BLE001 — first-time use, model not yet cached
        _local_path_cache = snapshot_download(repo_id=MODEL_REPO)
    return _local_path_cache


def is_model_cached() -> bool:
    """True if the model is already in the HF cache (no network needed).

    Use this at daemon startup to warn the user that their first
    recognition will block on a ~1.5 GB download if they skipped
    ``warmup``.
    """
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo_id=MODEL_REPO, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def _load_model():
    """Load mlx-whisper model exactly once."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        t0 = time.perf_counter()
        from mlx_whisper.load_models import load_model

        local_path = _local_model_path()
        logger.info("loading mlx-whisper model %s …", MODEL_REPO)
        _model = load_model(local_path)
        dt = time.perf_counter() - t0
        logger.info("model loaded in %.1fs (will stay in RAM)", dt)
        return _model


def warm() -> None:
    """Force-load the model (used by ``dictate-mac warmup``)."""
    _load_model()


def transcribe(audio: np.ndarray, language: str = "auto") -> str:
    """Run ASR on a mono 16 kHz float32 buffer; return plain text.

    ``language`` may be either an ISO-639-1 code (e.g. ``"ru"``) or
    :data:`dictate_mac.config.AUTO` (the string ``"auto"``). When it is
    ``"auto"`` this function maps the value to ``None`` before calling
    ``mlx_whisper.transcribe``, which triggers Whisper's built-in
    first-30-second language detection on every recording
    (~0.3-0.8 s overhead per call).

    The model is loaded on first invocation and stays in RAM for the
    lifetime of the process — subsequent calls skip the load.
    """
    from dictate_mac.config import AUTO as CONFIG_AUTO

    global _first_call_done
    if audio is None or audio.size == 0:
        return ""
    _load_model()

    import mlx_whisper

    local_path = _local_model_path()

    whisper_lang: Optional[str] = None if language == CONFIG_AUTO else language

    t0 = time.perf_counter()
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=local_path,
        language=whisper_lang,
        task=TASK,
        fp16=True,
        verbose=False,
    )
    text = (result.get("text") or "").strip()
    dt = time.perf_counter() - t0
    if not _first_call_done:
        _first_call_done = True
        logger.info(
            "first recognition done in %.2fs — model warm in RAM "
            "(language=%s)",
            dt,
            language,
        )
    else:
        logger.info(
            "recognition done in %.2fs (%d chars, language=%s)",
            dt,
            len(text),
            language,
        )
    return text


def model_loaded() -> bool:
    return _model is not None
