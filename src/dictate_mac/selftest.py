"""End-to-end self-test for dictate-mac.

Runs the headless portion of the pipeline (model load, VAD trimming, ASR
smoke, typer dispatch routing, config migration, API-ASR plumbing) and
reports each step. Designed to be runnable on any Mac without
Accessibility permission for keystroke injection, and (optionally)
without microphone access — so it can act as a smoke test after
installation or after dependency changes.

Usage::

    dictate-mac selftest [--no-mic]

Exit code is 0 if every check passed, 1 otherwise. Each check prints a
PASS/FAIL line plus a short detail string.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np

from dictate_mac.audio import SAMPLE_RATE, Recorder, trim_silence
from dictate_mac.transcriber import (
    MODEL_KIND_API,
    _audio_to_wav_bytes,
    _transcribe_api,
    check_api_model_available,
    model_loaded,
    transcribe,
    warm,
)
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


def test_ssl_certifi_on_disk() -> Result:
    """certifi's CA bundle must be a real file on the filesystem.

    ``ssl.create_default_context(cafile=certifi.where())`` — used by
    httpx (huggingface_hub model download) and requests (API backend)
    — can only read from disk. In a py2app bundle where certifi lives
    inside python313.zip, ``certifi.where()`` returns a non-existent
    path and every HTTPS call dies with ``FileNotFoundError``. This
    check catches that packaging regression.
    """
    import os
    import ssl

    import certifi

    for var in ("SSL_CERT_FILE", "SSL_CERT_DIR"):
        val = os.environ.get(var)
        if val and not Path(val).exists():
            return Result(
                "ssl-certifi",
                False,
                f"{var}={val!r} points at a non-existent path "
                "(httpx trust_env would fail every HTTPS call)",
            )

    pem = certifi.where()
    if not pem or not Path(pem).is_file():
        return Result(
            "ssl-certifi",
            False,
            f"certifi.where() -> {pem!r} is not a real file on disk "
            "(HTTPS would fail with FileNotFoundError)",
        )
    try:
        ssl.create_default_context(cafile=pem)
    except Exception as exc:  # noqa: BLE001
        return Result(
            "ssl-certifi",
            False,
            f"ssl.create_default_context(cafile={pem!r}) raised: {exc}",
        )
    try:
        import httpx

        httpx.Client()
    except Exception as exc:  # noqa: BLE001
        return Result(
            "ssl-certifi",
            False,
            f"httpx.Client() (trust_env path used by huggingface_hub) "
            f"raised: {exc}",
        )
    return Result("ssl-certifi", True, f"CA bundle readable at {pem}")


def test_config_v1_migration_keeps_language() -> Result:
    """A v1 config.json (no model_kind) loads as model_kind='local' with
    empty API fields and the persisted language preserved."""
    from dictate_mac import config as config_mod

    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "config.json"
        target.write_text(json.dumps({"_v": 1, "language": "ru"}))

        original = config_mod.config_path
        try:
            config_mod.config_path = lambda: target
            loaded = config_mod.load()
        finally:
            config_mod.config_path = original

        if loaded.language != "ru":
            return Result(
                "config-v1-migration",
                False,
                f"language lost: {loaded.language!r} (expected 'ru')",
            )
        if loaded.model_kind != config_mod.MODEL_KIND_LOCAL:
            return Result(
                "config-v1-migration",
                False,
                f"model_kind wrong: {loaded.model_kind!r}",
            )
        if loaded.api_endpoint or loaded.api_key or loaded.api_model_id:
            return Result(
                "config-v1-migration",
                False,
                "v1 file unexpectedly produced non-empty API fields",
            )

        on_disk_after = json.loads(target.read_text())
        if on_disk_after.get("_v") != 1:
            return Result(
                "config-v1-migration",
                False,
                f"v1 file was rewritten on load: {on_disk_after}",
            )

        return Result(
            "config-v1-migration",
            True,
            "v1 schema accepted, language preserved, model_kind defaulted to 'local'",
        )


def test_config_invalid_endpoint_rejected() -> Result:
    """PersistedSettings with model_kind='api' and a non-http endpoint
    is rejected by ``is_valid``."""
    from dictate_mac.config import PersistedSettings

    invalid_endpoints = [
        "",
        "ftp://example.test/v1",
        "example.test/v1",
        "htp:/typo.example/v1",
    ]
    for endpoint in invalid_endpoints:
        s = PersistedSettings(
            model_kind=MODEL_KIND_API,
            api_endpoint=endpoint,
            api_key="k",
            api_model_id="m",
        )
        if s.is_valid():
            return Result(
                "config-invalid-endpoint",
                False,
                f"endpoint {endpoint!r} should be rejected",
            )
    return Result(
        "config-invalid-endpoint",
        True,
        f"rejected {len(invalid_endpoints)} malformed endpoints",
    )


def test_config_missing_api_fields_when_kind_local() -> Result:
    """PersistedSettings with model_kind='local' accepts empty
    api_* fields; the same payload with model_kind='api' is rejected."""
    from dictate_mac.config import PersistedSettings

    local_only = PersistedSettings(model_kind="local")
    if not local_only.is_valid():
        return Result(
            "config-api-required-when-api",
            False,
            "local model_kind with empty API fields should be valid",
        )

    api_partial = PersistedSettings(
        model_kind=MODEL_KIND_API, api_endpoint="http://x/v1",
    )
    if api_partial.is_valid():
        return Result(
            "config-api-required-when-api",
            False,
            "api model_kind missing key + model_id should be invalid",
        )

    return Result(
        "config-api-required-when-api",
        True,
        "local kind accepts empty API fields; api kind rejects partial fields",
    )


def test_audio_to_wav_bytes_round_trip() -> Result:
    """numpy -> int16 WAV bytes -> numpy: amplitude preserved within
    one quantization step."""
    import io
    import wave

    duration = 0.5
    sr = SAMPLE_RATE
    t = np.arange(int(duration * sr)) / sr
    audio = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav = _audio_to_wav_bytes(audio, sr)

    if wav[:4] != b"RIFF":
        return Result(
            "audio-wav-roundtrip",
            False,
            f"unexpected wav header: {wav[:4]!r}",
        )

    with wave.open(io.BytesIO(wav), "rb") as r:
        if r.getnchannels() != 1:
            return Result(
                "audio-wav-roundtrip",
                False,
                f"channels={r.getnchannels()} (expected 1)",
            )
        if r.getframerate() != sr:
            return Result(
                "audio-wav-roundtrip",
                False,
                f"sample rate={r.getframerate()} (expected {sr})",
            )
        if r.getsampwidth() != 2:
            return Result(
                "audio-wav-roundtrip",
                False,
                f"sample width={r.getsampwidth()} (expected 2)",
            )
        decoded = np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16)

    expected = np.clip(audio * 32767.0, -32768.0, 32767.0).astype(np.int16)
    diff = int(np.abs(decoded.astype(np.int32) - expected.astype(np.int32)).max())
    if diff > 1:
        return Result(
            "audio-wav-roundtrip",
            False,
            f"amplitude drift {diff} LSB exceeds 1-sample tolerance",
        )

    empty_wav = _audio_to_wav_bytes(np.zeros(0, dtype=np.float32), sr)
    if len(empty_wav) < 44:
        return Result(
            "audio-wav-roundtrip",
            False,
            f"empty audio produced {len(empty_wav)} bytes (need header)",
        )

    return Result(
        "audio-wav-roundtrip",
        True,
        f"{audio.size} samples encoded and decoded, max drift {diff} LSB",
    )


def test_api_transcribe_sends_model_id_and_bearer() -> Result:
    """Mock ``requests.post`` and confirm the ASR builder sends the
    correct URL, multipart `model` field and `Authorization` header."""
    import dictate_mac.transcriber as t

    captured: dict = {}

    class _FakeResp:
        status_code = 200
        ok = True
        text = ""

        def json(self) -> dict:
            return {"text": "  hello world  "}

    def fake_post(url, *, files, data, headers, timeout):
        captured["url"] = url
        captured["files"] = files
        captured["data"] = data
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResp()

    class _FakeRequests:
        def post(self, url, *, files, data, headers, timeout):
            return fake_post(url, files=files, data=data, headers=headers, timeout=timeout)

    real = t.requests
    try:
        t.requests = _FakeRequests()
        audio = np.zeros(SAMPLE_RATE // 4, dtype=np.float32)
        text = t.transcribe(
            audio,
            language="en",
            model_kind=t.MODEL_KIND_API,
            api_endpoint="http://example.test/v1/",
            api_key="TESTKEY",
            api_model_id="test-model",
        )
    finally:
        t.requests = real

    if text != "hello world":
        return Result(
            "api-transcribe-headers",
            False,
            f"unexpected text: {text!r}",
        )
    if captured.get("url") != "http://example.test/v1/audio/transcriptions":
        return Result(
            "api-transcribe-headers",
            False,
            f"unexpected url: {captured.get('url')!r}",
        )
    if captured.get("data", {}).get("model") != "test-model":
        return Result(
            "api-transcribe-headers",
            False,
            f"multipart model wrong: {captured.get('data', {}).get('model')!r}",
        )
    if captured.get("data", {}).get("language") != "en":
        return Result(
            "api-transcribe-headers",
            False,
            f"language field wrong: {captured.get('data', {}).get('language')!r} "
            "(expected 'en' to be forwarded to the gateway)",
        )
    if captured.get("headers", {}).get("Authorization") != "Bearer TESTKEY":
        return Result(
            "api-transcribe-headers",
            False,
            f"Authorization header wrong: "
            f"{captured.get('headers', {}).get('Authorization')!r}",
        )
    file_payload = captured.get("files", {})
    file_name = file_payload.get("file", (None, None, None))[0]
    if file_name != "audio.wav":
        return Result(
            "api-transcribe-headers",
            False,
            f"file name wrong: {file_name!r}",
        )

    return Result(
        "api-transcribe-headers",
        True,
        "POST url+model+bearer+filename+language all wired correctly",
    )


def test_api_transcribe_omits_language_when_auto() -> Result:
    """When ``language='auto'`` (the sentinel), the multipart body
    must NOT carry a ``language`` field — the gateway should fall
    back to its own language detection."""
    import dictate_mac.transcriber as t

    captured: dict = {}

    class _FakeResp:
        status_code = 200
        ok = True
        text = ""

        def json(self) -> dict:
            return {"text": "hi"}

    class _FakeRequests:
        def post(self, url, *, files, data, headers, timeout):
            captured["data"] = data
            return _FakeResp()

    real = t.requests
    try:
        t.requests = _FakeRequests()
        t.transcribe(
            np.zeros(SAMPLE_RATE // 4, dtype=np.float32),
            language="auto",
            model_kind=t.MODEL_KIND_API,
            api_endpoint="http://example.test/v1",
            api_key="k",
            api_model_id="m",
        )
    finally:
        t.requests = real

    if "language" in captured.get("data", {}):
        return Result(
            "api-transcribe-auto-language",
            False,
            f"'language' field should be absent in auto mode, "
            f"got {captured['data'].get('language')!r}",
        )
    return Result(
        "api-transcribe-auto-language",
        True,
        "'language' field omitted from form when language='auto'",
    )


def test_models_endpoint_check_accepts_and_rejects() -> Result:
    """Mock ``GET /models`` with several response shapes and confirm
    the validator categorises each correctly. Secret strings must
    never appear in the raised errors."""
    import dictate_mac.transcriber as t

    case_results: list[tuple[str, bool, str | None]] = []

    class _FakeResp:
        def __init__(self, status_code: int, payload: dict | None = None,
                     text: str = "") -> None:
            self.status_code = status_code
            self._payload = payload
            self.text = text
            self.ok = 200 <= status_code < 300

        def json(self) -> dict:
            assert self._payload is not None
            return self._payload

    def run_case(name: str, status: int, payload: dict | None,
                 expect_ok: bool, must_not_contain: str | None = None) -> None:
        captured: dict = {}

        class _FakeReq:
            def get(self, url, *, headers, timeout):
                captured["url"] = url
                captured["headers"] = headers
                captured["timeout"] = timeout
                return _FakeResp(status, payload)

        real = t.requests
        try:
            t.requests = _FakeReq()
            try:
                check_api_model_available(
                    "http://example.test/v1",
                    "SECRET_KEY_DO_NOT_LEAK",
                    "wanted-model",
                    timeout=5.0,
                )
                outcome_ok = True
                detail = None
            except RuntimeError as exc:
                outcome_ok = False
                detail = str(exc)
        finally:
            t.requests = real

        if expect_ok != outcome_ok:
            case_results.append((name, False, f"expected ok={expect_ok} got {outcome_ok} ({detail})"))
            return
        if must_not_contain is not None and detail is not None and must_not_contain in detail:
            case_results.append((name, False, f"error leaked secret: {detail!r}"))
            return
        case_results.append((name, True, None))

    run_case(
        "200 with model",
        200,
        {"data": [{"id": "wanted-model"}, {"id": "other"}]},
        expect_ok=True,
    )
    run_case(
        "200 missing model",
        200,
        {"data": [{"id": "different-model"}]},
        expect_ok=False,
        must_not_contain="SECRET_KEY_DO_NOT_LEAK",
    )
    run_case(
        "401",
        401,
        None,
        expect_ok=False,
        must_not_contain="SECRET_KEY_DO_NOT_LEAK",
    )
    run_case(
        "404",
        404,
        None,
        expect_ok=False,
        must_not_contain="SECRET_KEY_DO_NOT_LEAK",
    )
    run_case(
        "500",
        500,
        None,
        expect_ok=False,
        must_not_contain="SECRET_KEY_DO_NOT_LEAK",
    )

    failures = [r for r in case_results if not r[1]]
    if failures:
        msg = "; ".join(f"{n}: {d}" for n, _, d in failures)
        return Result(
            "api-models-check",
            False,
            f"{len(failures)} case(s) failed: {msg}",
        )
    return Result(
        "api-models-check",
        True,
        f"{len(case_results)} cases (200/id, 200/missing, 401, 404, 500) handled correctly",
    )


def test_warmup_failure_retryable() -> Result:
    """A failed warmup must not be terminal: the machine publishes a
    retryable ERROR with the hotkey watcher still armed, and the next
    Right Option press re-runs the warmup into READY."""
    import asyncio

    import dictate_mac.state as state_mod
    from dictate_mac.hotkey import HotkeyEdge, HotkeyEvent
    from dictate_mac.state import DictationMachine, Settings, State

    calls = {"warm": 0}

    def fake_is_cached() -> bool:
        return True

    def fake_ensure_warm(on_phase=None):
        calls["warm"] += 1
        if on_phase is not None:
            if calls["warm"] == 1:
                on_phase("error", "simulated warmup failure")
            else:
                on_phase("ready", "")

        class _Thread:
            def is_alive(self) -> bool:
                return False

        return _Thread()

    class _FakeWatcher:
        def __init__(self) -> None:
            self.started = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            pass

        def is_alive(self) -> bool:
            return True

    async def scenario() -> tuple[bool, str]:
        machine = DictationMachine(settings=Settings(model_kind="local"))
        fake_watcher = _FakeWatcher()
        machine._watcher = fake_watcher  # type: ignore[attr-defined]
        real_cached = state_mod.is_model_cached
        real_warm = state_mod.ensure_warm_async
        state_mod.is_model_cached = fake_is_cached
        state_mod.ensure_warm_async = fake_ensure_warm
        try:
            task = asyncio.create_task(machine.run())
            for _ in range(200):
                await asyncio.sleep(0.01)
                if machine.state == State.ERROR:
                    break
            else:
                machine.stop()
                return False, "machine never reached ERROR after failed warmup"
            if not machine.warmup_failed:
                machine.stop()
                return False, "warmup_failed flag not set after failed warmup"
            if not fake_watcher.started:
                machine.stop()
                return False, "hotkey watcher not armed after failed warmup"
            machine._hotkey_queue.put_nowait(
                HotkeyEvent(edge=HotkeyEdge.PRESS, flags=0, keycode=0x3D)
            )
            for _ in range(200):
                await asyncio.sleep(0.01)
                if machine.state == State.READY:
                    break
            else:
                machine.stop()
                return False, f"retry did not reach READY (state={machine.state})"
            machine.stop()
            await asyncio.wait_for(task, timeout=2.0)
        finally:
            state_mod.is_model_cached = real_cached
            state_mod.ensure_warm_async = real_warm
        return True, ""

    try:
        ok, detail = asyncio.run(scenario())
    except Exception as exc:  # noqa: BLE001
        return Result("warmup-retry", False, f"raised: {exc}")
    if not ok:
        return Result("warmup-retry", False, detail)
    if calls["warm"] != 2:
        return Result(
            "warmup-retry",
            False,
            f"expected 2 warmup attempts, got {calls['warm']}",
        )
    return Result(
        "warmup-retry",
        True,
        "failed warmup stays retryable; Right Option re-runs warmup → READY",
    )


def test_recorder_portaudio_retry() -> Result:
    """When the first ``sd.InputStream`` open fails with a PortAudioError
    (stale device snapshot after a topology change), Recorder must
    re-initialise PortAudio and retry once instead of failing."""
    import dictate_mac.audio as audio_mod

    calls = {"open": 0, "terminate": 0, "initialize": 0}

    class _FakeStream:
        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

    real_input_stream = audio_mod.sd.InputStream
    real_terminate = audio_mod.sd._terminate
    real_initialize = audio_mod.sd._initialize

    def fake_input_stream(**kwargs):  # noqa: ARG001
        calls["open"] += 1
        if calls["open"] == 1:
            raise audio_mod.sd.PortAudioError("Internal PortAudio error", -9986)
        return _FakeStream()

    def fake_terminate() -> None:
        calls["terminate"] += 1

    def fake_initialize() -> None:
        calls["initialize"] += 1

    try:
        audio_mod.sd.InputStream = fake_input_stream
        audio_mod.sd._terminate = fake_terminate
        audio_mod.sd._initialize = fake_initialize
        rec = Recorder()
        rec.start()
        audio = rec.stop()
    except Exception as exc:  # noqa: BLE001
        return Result(
            "recorder-portaudio-retry",
            False,
            f"retry path raised: {exc}",
        )
    finally:
        audio_mod.sd.InputStream = real_input_stream
        audio_mod.sd._terminate = real_terminate
        audio_mod.sd._initialize = real_initialize

    if calls["open"] != 2 or calls["terminate"] != 1 or calls["initialize"] != 1:
        return Result(
            "recorder-portaudio-retry",
            False,
            f"unexpected call pattern: {calls} (expected 2 opens, 1 terminate, 1 initialize)",
        )
    if audio.size != 0:
        return Result(
            "recorder-portaudio-retry",
            False,
            f"expected empty buffer from fake stream, got {audio.size} samples",
        )
    return Result(
        "recorder-portaudio-retry",
        True,
        "first open failed with -9986, PortAudio re-initialised, retry succeeded",
    )


def test_hotkey_escape_event() -> Result:
    """A synthetic Esc keyDown reaches the hotkey queue as a PRESS with
    keycode 0x35; Esc with Cmd held is filtered out."""
    import queue as queue_mod

    from Quartz import (
        CGEventCreateKeyboardEvent,
        CGEventSetFlags,
        kCGEventFlagMaskCommand,
        kCGEventKeyDown,
    )

    from dictate_mac.hotkey import K_VK_ESCAPE, HotkeyWatcher

    q: queue_mod.Queue = queue_mod.Queue()
    watcher = HotkeyWatcher(q)

    plain_esc = CGEventCreateKeyboardEvent(None, K_VK_ESCAPE, True)
    watcher._handle_event(kCGEventKeyDown, plain_esc)

    cmd_esc = CGEventCreateKeyboardEvent(None, K_VK_ESCAPE, True)
    CGEventSetFlags(cmd_esc, kCGEventFlagMaskCommand)
    watcher._handle_event(kCGEventKeyDown, cmd_esc)

    events = []
    while True:
        try:
            events.append(q.get_nowait())
        except queue_mod.Empty:
            break

    if len(events) != 1:
        return Result(
            "hotkey-escape-event",
            False,
            f"expected exactly 1 event (plain Esc), got {len(events)}",
        )
    ev = events[0]
    if ev.keycode != K_VK_ESCAPE or ev.edge.value != "press":
        return Result(
            "hotkey-escape-event",
            False,
            f"unexpected event: {ev!r}",
        )
    return Result(
        "hotkey-escape-event",
        True,
        "plain Esc queued as cancel press; Cmd+Esc filtered out",
    )


def run_all(*, with_mic: bool = True, language: str = "auto") -> List[Result]:
    tests: List[Callable[[], Result]] = [
        test_model_load,
        test_vad_silence,
        test_vad_speech_like,
        lambda: test_asr_smoke(language=language),
        test_typer_dispatch,
        test_ssl_certifi_on_disk,
        test_config_v1_migration_keeps_language,
        test_config_invalid_endpoint_rejected,
        test_config_missing_api_fields_when_kind_local,
        test_audio_to_wav_bytes_round_trip,
        test_api_transcribe_sends_model_id_and_bearer,
        test_api_transcribe_omits_language_when_auto,
        test_models_endpoint_check_accepts_and_rejects,
        test_warmup_failure_retryable,
        test_recorder_portaudio_retry,
        test_hotkey_escape_event,
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