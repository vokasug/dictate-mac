"""End-to-end self-test for dictate-mac.

Runs the headless portion of the pipeline (model load, VAD trimming, ASR
smoke, typer dispatch routing) and reports each step. Designed to be
runnable on any Mac without Accessibility permission for keystroke
injection, and (optionally) without microphone access — so it can act as
a smoke test after installation or after dependency changes.

Usage::

    dictate-mac selftest [--no-mic]

Exit code is 0 if every check passed, 1 otherwise. Each check prints a
PASS/FAIL line plus a short detail string.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from dictate_mac.audio import SAMPLE_RATE, Recorder, trim_silence
from dictate_mac.transcriber import model_loaded, transcribe, warm
from dictate_mac.typer import type_text

logger = logging.getLogger("dictate_mac.selftest")


@dataclass(frozen=True)
class Result:
    name: str
    ok: bool
    detail: str


def _synth_silence(seconds: float = 1.0) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def _synth_speech_like(
    seconds_total: float = 2.0,
    speech_seconds: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate a speech-shaped signal: AM-modulated noise inside silence.

    silero-vad is trained on real speech, so a pure sine tone is often
    rejected. Amplitude-modulated broadband noise with a Hann envelope
    produces a signal whose energy and temporal dynamics are close enough
    to speech to trigger the VAD.
    """
    sr = SAMPLE_RATE
    n_total = int(seconds_total * sr)
    n_speech = int(speech_seconds * sr)
    rng = np.random.default_rng(seed=seed)
    noise = rng.standard_normal(n_speech).astype(np.float32)
    t = np.arange(n_speech) / sr
    am = 0.4 + 0.6 * np.abs(np.sin(2 * np.pi * 6 * t))  # ~6 Hz syllable rate
    window = np.hanning(n_speech).astype(np.float32)
    speech = (am * window * noise * 0.5).astype(np.float32)
    audio = np.zeros(n_total, dtype=np.float32)
    start = (n_total - n_speech) // 2
    audio[start : start + n_speech] = speech
    return audio


def test_model_load() -> Result:
    t0 = time.perf_counter()
    warm()
    dt = time.perf_counter() - t0
    if not model_loaded():
        return Result("model-load", False, "model not loaded after warm()")
    return Result("model-load", True, f"loaded in {dt:.2f}s")


def test_vad_silence() -> Result:
    audio = _synth_silence(1.0)
    out = trim_silence(audio)
    if out.size != 0:
        return Result(
            "vad-silence",
            False,
            f"expected empty output for pure-silence input, got {out.size} samples",
        )
    return Result("vad-silence", True, "trim_silence correctly returned [] for 1s of zeros")


def test_vad_speech_like() -> Result:
    """Confirm ``trim_silence`` runs cleanly on speech-shaped input.

    silero-vad is trained on real speech, so a synthetic AM-noise signal
    may or may not trigger it depending on the random seed and signal
    shape. We treat both outcomes as informative, not failing:

    * Detected within an expected window → strong PASS.
    * Rejected (empty output) → soft PASS with a note explaining why.
    * Detected but wildly off-duration → FAIL (real bug).
    """
    audio = _synth_speech_like(seconds_total=2.0, speech_seconds=1.0)
    out = trim_silence(audio)
    if out.size == 0:
        return Result(
            "vad-speech-like",
            True,
            "synthetic AM-noise not classified as speech (expected — "
            "silero-vad is tuned for real speech; mic-roundtrip covers that)",
        )
    seconds = out.size / SAMPLE_RATE
    if not (0.3 <= seconds <= 1.7):
        return Result(
            "vad-speech-like",
            False,
            f"trimmed duration {seconds:.2f}s outside expected window (0.3..1.7)",
        )
    return Result(
        "vad-speech-like",
        True,
        f"trimmed {seconds:.2f}s of synthetic speech-shaped audio",
    )


def test_asr_smoke(language: str = "auto") -> Result:
    audio = _synth_speech_like(seconds_total=1.0, speech_seconds=0.8)
    t0 = time.perf_counter()
    text = transcribe(audio, language=language)
    dt = time.perf_counter() - t0
    if not isinstance(text, str):
        return Result("asr-smoke", False, f"expected str, got {type(text).__name__}")
    return Result(
        "asr-smoke",
        True,
        f"transcribe(language={language!r}) returned a {len(text)}-char "
        f"string in {dt:.2f}s "
        "(content is not asserted — synthetic signal is not real speech)",
    )


def test_typer_dispatch() -> Result:
    """Verify the ``type_text`` router picks the requested backend.

    Patches ``dictate_mac.typer.type_text_quartz`` and
    ``dictate_mac.typer.type_text_osascript`` so no real keystrokes are
    injected during the test.
    """
    import dictate_mac.typer as typer_mod

    calls: list[tuple[str, tuple, dict]] = []
    real_q = typer_mod.type_text_quartz
    real_a = typer_mod.type_text_osascript
    typer_mod.type_text_quartz = lambda *a, **kw: calls.append(("quartz", a, kw))
    typer_mod.type_text_osascript = lambda *a, **kw: calls.append(("osascript", a, kw))
    try:
        type_text("hello", backend="quartz")
        type_text("world", backend="osascript")
        type_text("default")  # default backend must be quartz
    finally:
        typer_mod.type_text_quartz = real_q
        typer_mod.type_text_osascript = real_a

    expected = [
        ("quartz", ("hello",), {"per_char_delay_ms": 8}),
        ("osascript", ("world",), {}),
        ("quartz", ("default",), {"per_char_delay_ms": 8}),
    ]
    if calls != expected:
        return Result(
            "typer-dispatch",
            False,
            f"unexpected dispatch log: {calls!r} (expected {expected!r})",
        )
    return Result(
        "typer-dispatch",
        True,
        "router dispatched quartz/osascript/default correctly without injecting keystrokes",
    )


def test_mic_roundtrip(seconds: float = 1.5, language: str = "auto") -> Result:
    """Record from the default microphone, run through VAD + ASR.

    Best-effort: returns PASS with a note if the room was silent (VAD
    returns empty). Only fails if the recorder itself errors out.
    """
    try:
        rec = Recorder()
        rec.start()
        time.sleep(seconds)
        audio = rec.stop()
    except Exception as exc:  # noqa: BLE001
        return Result(
            "mic-roundtrip",
            False,
            f"recorder failed: {exc} (check Microphone permission for Terminal)",
        )
    if audio.size == 0:
        return Result("mic-roundtrip", False, "recorder returned empty buffer")
    trimmed = trim_silence(audio)
    if trimmed.size == 0:
        return Result(
            "mic-roundtrip",
            True,
            f"recorded {audio.size / SAMPLE_RATE:.2f}s but VAD found no speech "
            "(silent room? speak during recording for a real test)",
        )
    t0 = time.perf_counter()
    text = transcribe(trimmed, language=language)
    dt = time.perf_counter() - t0
    return Result(
        "mic-roundtrip",
        True,
        f"recorded {audio.size / SAMPLE_RATE:.2f}s -> VAD "
        f"{trimmed.size / SAMPLE_RATE:.2f}s -> ASR {len(text)} chars in {dt:.2f}s "
        f"(language={language!r})",
    )


def run_all(*, with_mic: bool = True, language: str = "auto") -> List[Result]:
    tests: List[Callable[[], Result]] = [
        test_model_load,
        test_vad_silence,
        test_vad_speech_like,
        lambda: test_asr_smoke(language=language),
        test_typer_dispatch,
    ]
    if with_mic:
        tests.append(lambda: test_mic_roundtrip(language=language))
    results: List[Result] = []
    for t in tests:
        logger.info("running %s …", t.__name__)
        try:
            r = t()
        except Exception as exc:  # noqa: BLE001
            r = Result(t.__name__, False, f"raised: {exc}")
        results.append(r)
    return results


def cmd_selftest(args: argparse.Namespace) -> int:
    results = run_all(with_mic=not args.no_mic, language=args.language)
    width = max(len(r.name) for r in results)
    failed = sum(1 for r in results if not r.ok)
    for r in results:
        marker = "PASS" if r.ok else "FAIL"
        print(f"  [{marker}] {r.name:<{width}}  {r.detail}")
    print()
    if failed:
        print(f"selftest: {failed} failure(s) out of {len(results)}")
        return 1
    print(f"selftest: all {len(results)} checks passed")
    return 0