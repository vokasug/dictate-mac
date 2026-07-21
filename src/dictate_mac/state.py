"""Asyncio state machine that drives the dictation loop.

Phase 4. Phase 9 extends the lifecycle with pre-ready phases.

Lifecycle:

    STARTING  ──► DOWNLOADING_MODEL (first run only)
              ──► LOADING_MODEL       (cache or download finished)
              ──► READY               (hotkey armed)
                  │
                  ├── Right Option press  ──►  RECORDING
                  ├── Right Option press  ──►  TRANSCRIBING
                  ├── audio ready         ──►  TYPING
                  └── done                ──►  READY

    ERROR     ── warmup failure is RETRYABLE (hotkey stays armed, a
                 Right Option press re-runs the warmup); hotkey-permission
                 failure is terminal until process exit.

The state machine is the only thing allowed to call ``Recorder.start`` /
``stop`` and to invoke ``transcriber.transcribe`` + ``typer.type_text``.
It owns a queue bridged to the hotkey tap thread.

The ``State`` enum is the user-facing status string — the menu bar
shows ``"Status: <state.value>"`` for the current enum value. Keep the
strings short and human-readable.

Phase 15 extends ``Settings`` with a ``language`` field (an ISO-639-1
code or the sentinel ``"auto"``). The hot-applied switch works without
a model reload because ``mlx_whisper.transcribe`` reads ``language``
per-call from the encoder — only the menubar persists this value to
``~/.config/dictate-mac/config.json``; the CLI daemon subcommand
takes its language from ``--language`` and ignores the config file.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from dictate_mac.audio import Recorder, trim_silence
from dictate_mac.config import (
    MODEL_KIND_API,
    MODEL_KIND_LOCAL,
)
from dictate_mac.hotkey import (
    K_VK_ESCAPE,
    HotkeyEdge,
    HotkeyEvent,
    HotkeyWatcher,
)
from dictate_mac.transcriber import (
    DEFAULT_API_TIMEOUT,
    ensure_warm_async,
    is_model_cached,
    transcribe as asr_transcribe,
)
from dictate_mac.typer import type_text as emit_text

logger = logging.getLogger("dictate_mac.state")

SOUND_START = "/System/Library/Sounds/Ping.aiff"
SOUND_END = "/System/Library/Sounds/Pop.aiff"


class State(str, enum.Enum):
    STARTING = "starting"
    DOWNLOADING_MODEL = "downloading"
    LOADING_MODEL = "loading"
    READY = "ready"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    TYPING = "typing"
    ERROR = "error"


@dataclass
class Settings:
    output_backend: str = "quartz"
    per_char_delay_ms: int = 8
    language: str = "auto"
    model_kind: str = MODEL_KIND_LOCAL
    api_endpoint: str = ""
    api_key: str = ""
    api_model_id: str = ""
    api_timeout: float = DEFAULT_API_TIMEOUT


def _play(sound_path: str) -> None:
    """Best-effort system sound playback via afplay."""
    try:
        subprocess.Popen(
            ["afplay", sound_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        logger.debug("afplay not available; skipping sound %s", sound_path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("afplay %s failed: %s", sound_path, exc)


class DictationMachine:
    """Top-level orchestrator. Owns the recorder, hotkey watcher, and state."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        # Internal state used by the asyncio pump. Mutated only from the
        # asyncio loop / pump methods.
        self._state = State.STARTING
        # Mirror published to other threads (menu bar timer) under a lock.
        self._state_lock = threading.Lock()
        self._state_value: State = State.STARTING

        self._recorder = Recorder()
        self._hotkey_queue: queue.Queue[HotkeyEvent] = queue.Queue(maxsize=32)
        self._watcher = HotkeyWatcher(self._hotkey_queue)

        # ``threading.Event`` rather than ``asyncio.Event`` so that
        # ``stop()`` can be called from any thread (main / menu / signal
        # handler) without coordinating with the worker's event loop.
        # The pump polls the flag every 10 ms.
        self._stopping = threading.Event()
        self._warmup_done = threading.Event()
        self._warmup_error: Optional[str] = None
        # True after a failed warmup attempt. The ERROR state is then
        # retryable: the hotkey watcher stays armed and a Right Option
        # press re-runs the warmup (first launch without network,
        # transient HF outage, …).
        self._warmup_failed = False
        # Set once the hotkey watcher thread has been started. Drives
        # the dead-tap detection in _pump_once (works in every state,
        # including the retryable ERROR state).
        self._watcher_started = False

    # -- public API -------------------------------------------------------

    @property
    def state(self) -> State:
        """Thread-safe snapshot of the current state.

        The menu bar timer polls this from the main thread every 0.5 s
        while the state machine mutates it from the asyncio loop.
        """
        with self._state_lock:
            return self._state_value

    @property
    def warmup_failed(self) -> bool:
        """True when the last warmup attempt failed and the machine is
        sitting in the retryable ERROR state (Right Option retries)."""
        return self._warmup_failed

    async def run(self) -> None:
        """Run the warmup → arm → pump loop until ``stop()`` is called."""
        await self._publish_state(State.STARTING, "[boot] starting")

        await self._warmup()
        # After warmup:
        #   * success → state is LOADING_MODEL / DOWNLOADING_MODEL
        #   * failure → state is ERROR, but RETRYABLE: the watcher is
        #     armed anyway and a Right Option press re-runs the warmup.

        try:
            self._watcher.start()
        except Exception as exc:  # noqa: BLE001
            logger.error("[hotkey] failed to start: %s", exc)
            await self._publish_state(
                State.ERROR,
                f"[hotkey] {exc} (grant Accessibility/Input Monitoring)",
            )
        else:
            self._watcher_started = True
            if not self._warmup_failed:
                await self._publish_state(State.READY, "[hotkey] ready")
            logger.info(
                "[hint] if Right Option presses do nothing, grant both "
                "Accessibility AND Input Monitoring (macOS 14+) to "
                "com.local.dictate-mac in System Settings → "
                "Privacy & Security"
            )

        try:
            while not self._stopping.is_set():
                await self._pump_once()
        finally:
            self._watcher.stop()
            if self.state == State.RECORDING:
                try:
                    self._recorder.stop()
                except Exception:  # noqa: BLE001
                    pass

    def stop(self) -> None:
        """Thread-safe stop request. Safe to call from any thread.

        Sets a ``threading.Event`` that the pump polls every 10 ms.
        """
        self._stopping.set()

    # -- warmup -----------------------------------------------------------

    async def _warmup(self) -> None:
        """Run ``ensure_warm_async`` and wait for it to finish.

        Bridges the cross-thread warmup callback into the asyncio loop
        via ``loop.call_soon_threadsafe`` so we can publish state
        transitions on the pump thread.

        In API mode there is no local model to load — the warmup
        thread is skipped entirely (no cache check, no download,
        no in-process import). The state machine still arms the
        hotkey and becomes ``READY`` so the menu bar can update.
        """
        loop = asyncio.get_running_loop()

        if self._settings.model_kind == MODEL_KIND_API:
            logger.info("[warmup] api mode — skipping local model load")
            await self._publish_state(
                State.READY,
                "[warmup] api mode, local model not loaded",
            )
            self._warmup_done.set()
            return

        if is_model_cached():
            await self._publish_state(
                State.LOADING_MODEL, "[warmup] loading model from cache"
            )
        else:
            await self._publish_state(
                State.DOWNLOADING_MODEL,
                "[warmup] downloading mlx-community/whisper-large-v3-turbo "
                "(first run; ~1.5 GB)",
            )

        def _on_phase(phase: str, detail: str) -> None:
            loop.call_soon_threadsafe(self._handle_warmup_phase, phase, detail)

        # Idempotent — a second call while a warmup is in flight is a
        # no-op and returns the existing thread.
        ensure_warm_async(on_phase=_on_phase)

        # Poll the cross-thread completion flag instead of awaiting an
        # asyncio.Event set from the other thread — the latter requires
        # call_soon_threadsafe as well, and the loop latency (1 ms) is
        # the same as a 50 ms poll.
        while not self._warmup_done.is_set() and not self._stopping.is_set():
            await asyncio.sleep(0.05)

        if self._warmup_error is not None:
            self._warmup_failed = True
            await self._publish_state(
                State.ERROR,
                f"[warmup] failed: {self._warmup_error} — press Right "
                "Option to retry (see logs)",
            )

    def _handle_warmup_phase(self, phase: str, detail: str) -> None:
        """Invoked on the asyncio loop for every warmup phase transition.

        Phase strings (from ``transcriber``):

        * ``"downloading"`` — first run, HF download in progress.
        * ``"loading"`` — mlx-whisper weights being read into RAM.
        * ``"ready"`` — both models resident in RAM.
        * ``"error"`` — the warmup thread swallowed an exception.
        """
        if phase == "downloading":
            asyncio.create_task(
                self._publish_state(
                    State.DOWNLOADING_MODEL,
                    f"[warmup] downloading {detail or 'model'} …",
                )
            )
        elif phase == "loading":
            asyncio.create_task(
                self._publish_state(
                    State.LOADING_MODEL,
                    "[warmup] loading model into RAM …",
                )
            )
        elif phase == "ready":
            self._warmup_done.set()
        elif phase == "error":
            self._warmup_error = detail or "unknown error"
            self._warmup_done.set()
        else:  # pragma: no cover — defensive
            logger.warning("unknown warmup phase: %r", phase)

    async def _retry_warmup(self) -> None:
        """Re-run the warmup after a previous failure.

        Triggered by a Right Option press while the machine sits in the
        retryable ERROR state (warmup failed, watcher still armed).
        ``ensure_warm_async`` starts a fresh background thread because
        the previous one exited with the failure.
        """
        assert self._state == State.ERROR and self._warmup_failed
        logger.info("[warmup] retry requested via Right Option")
        self._warmup_failed = False
        self._warmup_error = None
        self._warmup_done.clear()
        await self._warmup()
        if not self._warmup_failed and not self._stopping.is_set():
            await self._publish_state(State.READY, "[hotkey] ready")

    # -- state machine ----------------------------------------------------

    async def _pump_once(self) -> None:
        # Drain hotkey queue (non-blocking) into a local buffer.
        events: list[HotkeyEvent] = []
        while True:
            try:
                ev = self._hotkey_queue.get_nowait()
            except queue.Empty:
                break
            events.append(ev)

        # Sentinel events from the watcher when permission failed.
        # Checked BEFORE the dead-tap detection below so the user gets
        # the actionable "grant Accessibility / Input Monitoring"
        # message instead of a generic "tap thread is gone".
        if any(ev.flags == -1 for ev in events):
            logger.error(
                "[hotkey] permission denied — grant Accessibility / "
                "Input Monitoring to com.local.dictate-mac in System Settings"
            )
            await self._publish_state(
                State.ERROR,
                "[hotkey] permission denied (see System Settings → "
                "Privacy & Security → Accessibility / Input Monitoring)",
            )
            self.stop()
            return

        # If the tap thread died (callback crash, CFRunLoop exited
        # unexpectedly) after being started, surface the failure and
        # exit instead of silently waiting for events that will never
        # arrive.
        if self._watcher_started and not self._watcher.is_alive():
            logger.error(
                "[hotkey] tap thread is gone — daemon cannot receive "
                "Right Option presses anymore, exiting"
            )
            self.stop()
            return

        # We react only to key-down events. Esc is a dedicated cancel
        # key, meaningful only while RECORDING.
        presses = [ev for ev in events if ev.edge == HotkeyEdge.PRESS]
        option_presses = [ev for ev in presses if ev.keycode != K_VK_ESCAPE]
        escape_presses = [ev for ev in presses if ev.keycode == K_VK_ESCAPE]

        if self._state == State.READY and option_presses:
            await self._start_recording()
        elif (
            self._state == State.ERROR
            and self._warmup_failed
            and option_presses
        ):
            await self._retry_warmup()
        elif self._state == State.RECORDING and escape_presses:
            await self._cancel_recording()
        elif self._state == State.RECORDING and option_presses:
            await self._stop_and_process()

        # Light sleep so we don't spin when no events arrive.
        await asyncio.sleep(0.01)

    async def _start_recording(self) -> None:
        await self._publish_state(State.RECORDING, "[rec] recording started")
        threading.Thread(target=_play, args=(SOUND_START,), daemon=True).start()
        try:
            self._recorder.start()
        except Exception as exc:  # noqa: BLE001
            logger.error("[rec] start failed: %s", exc)
            await self._publish_state(State.READY, "[idle] ready")

    async def _cancel_recording(self) -> None:
        assert self._state == State.RECORDING
        try:
            self._recorder.stop()
        except Exception as exc:  # noqa: BLE001
            logger.error("[rec] stop failed during cancel: %s", exc)
        threading.Thread(target=_play, args=(SOUND_END,), daemon=True).start()
        logger.info("[rec] cancelled via Esc")
        await self._publish_state(State.READY, "[idle] ready")

    async def _stop_and_process(self) -> None:
        assert self._state == State.RECORDING
        try:
            audio = self._recorder.stop()
        except Exception as exc:  # noqa: BLE001
            logger.error("[rec] stop failed: %s", exc)
            await self._publish_state(State.READY, "[idle] ready")
            return

        # Play the completion sound IMMEDIATELY so the user gets
        # instant feedback that the second Option press was registered,
        # before the 1-3 s VAD + ASR + typing pipeline runs.
        threading.Thread(target=_play, args=(SOUND_END,), daemon=True).start()

        await self._publish_state(State.TRANSCRIBING, "[vad] trimming silence …")
        trimmed = trim_silence(audio)
        if trimmed.size == 0:
            logger.info("[vad] no speech — typing nothing")
            await self._publish_state(State.READY, "[idle] ready")
            return

        await self._publish_state(State.TRANSCRIBING, "[asr] transcribing …")
        t0 = time.perf_counter()
        try:
            if self._settings.model_kind == MODEL_KIND_API:
                text = await asyncio.to_thread(
                    asr_transcribe,
                    trimmed,
                    self._settings.language,
                    model_kind=MODEL_KIND_API,
                    api_endpoint=self._settings.api_endpoint,
                    api_key=self._settings.api_key,
                    api_model_id=self._settings.api_model_id,
                    api_timeout=self._settings.api_timeout,
                )
            else:
                text = await asyncio.to_thread(
                    asr_transcribe, trimmed, self._settings.language
                )
        except RuntimeError as exc:
            dt = time.perf_counter() - t0
            logger.warning(
                "[asr] api back-end failed after %.2fs: %s — typing nothing",
                dt,
                exc,
            )
            await self._publish_state(State.READY, "[idle] ready")
            return
        dt = time.perf_counter() - t0
        if not text:
            logger.info("[asr] empty result — typing nothing (took %.2fs)", dt)
            await self._publish_state(State.READY, "[idle] ready")
            return

        # Append a trailing space so the next dictation doesn't get
        # glued to this one. Whisper emits punctuation but no trailing
        # whitespace; editors don't auto-insert a separator either.
        text_with_sep = text + " "

        await self._publish_state(State.TYPING, f"[type] {len(text)} chars + sep …")
        await asyncio.to_thread(
            emit_text,
            text_with_sep,
            self._settings.output_backend,
            self._settings.per_char_delay_ms,
        )
        await self._publish_state(State.READY, "[idle] ready")

    async def _publish_state(self, new_state: State, message: str) -> None:
        """Set internal + published state, log, and signal watchers."""
        self._state = new_state
        with self._state_lock:
            self._state_value = new_state
        logger.info(message)
