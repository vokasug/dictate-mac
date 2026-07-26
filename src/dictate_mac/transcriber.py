"""ASR backends for dictate-mac.

Three paths produce text from a mono 16 kHz float32 buffer:

* ``_transcribe_local`` runs ``mlx-community/whisper-large-v3-turbo``
  in-process via ``mlx_whisper.transcribe``. The model is loaded once
  through mlx-whisper's own ``ModelHolder`` cache — so the warmup and
  every later transcription share a single instance — and stays
  resident in RAM for the lifetime of the process. The MLX Metal
  buffer cache is returned to the OS (``mx.clear_cache``) after each
  transcription, so the footprint does not grow across dictations.
* ``_transcribe_gigaam`` runs the GigaAM multilingual CTC model
  (``gigaam-multilingual-mlx`` package, FP16 artifact) in-process.
  Same lifecycle as the whisper path: one instance pinned to the
  dedicated MLX thread, resident in RAM, ``mx.clear_cache`` after
  each transcription. The model is a fixed multilingual CTC — it
  has no language parameter, so ``language`` is ignored on this path.
* ``_transcribe_api`` POSTs a 16 kHz mono WAV to an
  OpenAI-compatible ``/v1/audio/transcriptions`` endpoint, passing the
  model id and bearer token the user configured in the menu bar.

The public :func:`transcribe` picks one of the three based on
``model_kind``; existing call sites that pass only ``(audio, language=)``
keep working unchanged because the API path parameters default to
disabled.

The :func:`check_api_model_available` helper does a ``GET
{endpoint}/models`` with the same bearer token and confirms the
configured model id appears in the response. The menu bar's API
settings dialog calls this on OK before persisting — a 401, a 404
endpoint, or a missing model id each surface as a categorised error
instead of being silently saved.

The API key is never logged. Error messages include only the
endpoint, the HTTP status, and (truncated) response body — not the
key, not the model id's full path on multi-segment identifiers.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np
import requests

from dictate_mac.audio import SAMPLE_RATE as _ASR_SAMPLE_RATE
from dictate_mac.config import (
    MODEL_KIND_API,
    MODEL_KIND_GIGAAM,
    MODEL_KIND_LOCAL,
    endpoint_scheme_ok,
    normalize_endpoint,
)

logger = logging.getLogger("dictate_mac.transcriber")


def _repair_ssl_cert_env() -> None:
    """Re-point broken ``SSL_CERT_FILE`` / ``SSL_CERT_DIR`` variables.

    py2app's ``__boot__.py`` exports both pointing at
    ``Resources/openssl.ca``, a directory the bundle strip removes.
    httpx honours ``SSL_CERT_FILE`` (``trust_env=True`` by default) and
    then raises ``FileNotFoundError`` on every HTTPS call — the Hugging
    Face model download included. When a variable points at a missing
    path, re-point it at certifi's ``cacert.pem`` (bundled on disk) or
    drop it so the default trust store is used. No-op outside the
    bundle where the variables are normally unset.
    """
    pem_env = os.environ.get("SSL_CERT_FILE")
    dir_env = os.environ.get("SSL_CERT_DIR")
    pem_ok = bool(pem_env) and os.path.isfile(pem_env)
    dir_ok = bool(dir_env) and os.path.isdir(dir_env)
    if pem_ok and (dir_ok or not dir_env):
        return
    fallback = ""
    try:
        import certifi

        candidate = certifi.where()
        if candidate and os.path.isfile(candidate):
            fallback = candidate
    except Exception:  # noqa: BLE001 — certifi missing: drop below
        pass
    if not pem_ok:
        if fallback:
            logger.info(
                "SSL_CERT_FILE pointed at a missing file — using %s", fallback
            )
            os.environ["SSL_CERT_FILE"] = fallback
        else:
            os.environ.pop("SSL_CERT_FILE", None)
    if not dir_ok:
        os.environ.pop("SSL_CERT_DIR", None)


_repair_ssl_cert_env()

MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
TASK = "transcribe"

# Per-language style prompts for the whisper path, keyed by ISO-639-1
# code (top 30 languages by number of speakers). Whisper is
# autoregressive and can fall into a "no-punctuation mode"
# (openai/whisper discussion #194); a short punctuated, capitalized
# sample sentence nudges the decoder into the with-punctuation mode
# and stabilizes capitalization (the model mimics the prompt style).
# Applied only when the user pinned one of these languages — a
# mismatched-language prompt hurts, and with language="auto" the
# spoken language is unknown at decode time.
WHISPER_INITIAL_PROMPTS = {
    "en": "Okay, let's begin. Is everything ready?",
    "zh": "好的，我们开始吧。都准备好了吗？",
    "hi": "अच्छा, शुरू करते हैं। सब तैयार है?",
    "es": "Bien, empecemos. ¿Está todo listo?",
    "ar": "حسنًا، لنبدأ. هل كل شيء جاهز؟",
    "fr": "Bon, commençons. Tout est prêt ?",
    "bn": "আচ্ছা, শুরু করি। সব কি প্রস্তুত?",
    "ru": "Хорошо, начинаем. Всё готово?",
    "pt": "Certo, vamos começar. Está tudo pronto?",
    "ur": "اچھا، شروع کرتے ہیں۔ کیا سب تیار ہے؟",
    "id": "Baik, kita mulai. Apakah semuanya sudah siap?",
    "de": "Gut, fangen wir an. Ist alles bereit?",
    "ja": "はい、始めましょう。準備はできていますか？",
    "mr": "ठीक आहे, सुरू करूया. सगळं तयार आहे का?",
    "te": "సరే, ప్రారంభిద్దాం. అంతా సిద్ధమైందా?",
    "tr": "Tamam, başlayalım. Her şey hazır mı?",
    "ta": "சரி, ஆரம்பிக்கலாம். எல்லாம் தயாரா?",
    "vi": "Được rồi, bắt đầu nào. Mọi thứ đã sẵn sàng chưa?",
    "ko": "좋아요, 시작하겠습니다. 준비됐나요?",
    "it": "Bene, iniziamo. È tutto pronto?",
    "th": "โอเค เรามาเริ่มกัน พร้อมหรือยัง?",
    "gu": "ઠીક છે, શરૂ કરીએ. બધું તૈયાર છે?",
    "fa": "خب، شروع کنیم. همه چیز آماده است؟",
    "pl": "Dobrze, zaczynamy. Wszystko gotowe?",
    "uk": "Гаразд, починаємо. Все готово?",
    "ms": "Baiklah, mari kita mulakan. Semuanya sudah bersedia?",
    "kn": "ಸರಿ, ಪ್ರಾರಂಭಿಸೋಣ. ಎಲ್ಲವೂ ಸಿದ್ಧವೇ?",
    "my": "ကောင်းပြီ၊ စလိုက်ကြရအောင်။ အကုန် အဆင်သင့် ဖြစ်ပြီလား?",
    "sw": "Vizuri, tuanze. Kila kitu tayari?",
    "pa": "ਠੀਕ ਹੈ, ਸ਼ੁਰੂ ਕਰੀਏ। ਸਭ ਕੁਝ ਤਿਆਰ ਹੈ?",
}

# GigaAM multilingual CTC (FP16 reference artifact). The repo id and
# pinned revision come from the package's own VARIANTS table so a
# package upgrade that bumps RELEASE_REVISION is picked up here.
GIGAAM_VARIANT = "fp16"
GIGAAM_REPO = "ai-babai/gigaam-multilingual-mlx"

# Long-buffer decoding window for the GigaAM path. The conformer is
# trained on ~20 s utterances; feeding a multi-minute buffer in one
# shot degrades recognition progressively with position (rotary
# attention outside its training length) and costs O(T^2) attention
# memory. The package's own CLI/server chunk at 20 s with 2 s overlap
# — mirror that.
GIGAAM_CHUNK_SECONDS = 20.0
GIGAAM_OVERLAP_SECONDS = 2.0

DEFAULT_API_TIMEOUT = 30.0
DEFAULT_CHECK_TIMEOUT = 5.0

WarmPhase = str  # "downloading" | "loading" | "ready" | "error"
WarmCallback = Callable[[WarmPhase, str], None]


_model = None
_model_lock = threading.Lock()
_first_call_done = False
_local_path_cache: Optional[str] = None

_gigaam_model = None
_gigaam_lock = threading.Lock()
_gigaam_first_call_done = False

_warmup_thread: Optional[threading.Thread] = None
_warmup_lock = threading.Lock()
_warmup_callback: Optional[WarmCallback] = None
_warmup_kind: str = MODEL_KIND_LOCAL

# MLX registers GPU stream handles per-thread: a model loaded on one
# thread dies on any other with "There is no Stream(gpu, N) in current
# thread". Callers reach us from arbitrary threads (the warmup thread,
# ``asyncio.to_thread`` pool workers, the CLI main thread), so the load
# and every transcription are pinned to one dedicated worker.
_mlx_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-asr")


def _run_on_mlx_thread(fn, *args, **kwargs):
    """Run ``fn`` on the dedicated MLX thread, wait, re-raise errors."""
    return _mlx_executor.submit(fn, *args, **kwargs).result()


def _emit(phase: WarmPhase, detail: str = "") -> None:
    cb = _warmup_callback
    if cb is None:
        return
    try:
        cb(phase, detail)
    except Exception:  # noqa: BLE001
        logger.exception("warmup callback raised")


def _do_warmup_blocking() -> None:
    """Run the actual download + load. Runs on the background thread."""
    kind = _warmup_kind
    try:
        if kind == MODEL_KIND_GIGAAM:
            if not _gigaam_model_cached():
                _emit("downloading", GIGAAM_REPO)
                logger.info("downloading %s (%s) …", GIGAAM_REPO, GIGAAM_VARIANT)
                _download_gigaam()
            _emit("loading", "")
            _run_on_mlx_thread(_load_gigaam)
        else:
            if not is_model_cached():
                _emit("downloading", MODEL_REPO)
                logger.info("downloading %s …", MODEL_REPO)
                _local_model_path()
            _emit("loading", "")
            _run_on_mlx_thread(_load_model)
        _emit("ready", "")
    except Exception as exc:  # noqa: BLE001
        logger.exception("background warmup failed; the next transcribe() will retry")
        _emit("error", str(exc))


def ensure_warm_async(
    on_phase: Optional[WarmCallback] = None,
    model_kind: str = MODEL_KIND_LOCAL,
) -> threading.Thread:
    """Start a background warmup if one isn't already running.

    Idempotent. The callback is invoked with lifecycle phase strings.
    Only used by the local paths (whisper / gigaam) — the API path
    has no model to load.
    """
    global _warmup_thread, _warmup_callback, _warmup_kind
    with _warmup_lock:
        if on_phase is not None:
            _warmup_callback = on_phase
        _warmup_kind = model_kind
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


def is_model_cached(model_kind: str = MODEL_KIND_LOCAL) -> bool:
    """True if the given local model is already in the HF cache."""
    if model_kind == MODEL_KIND_GIGAAM:
        return _gigaam_model_cached()
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo_id=MODEL_REPO, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def _gigaam_repo_revision() -> tuple[str, str]:
    """Repo id + pinned revision for the configured GigaAM variant.

    Read from the package's VARIANTS table so a package upgrade that
    moves RELEASE_REVISION forward is honoured here too.
    """
    from gigaam_multilingual_mlx.artifacts import VARIANTS

    meta = VARIANTS[GIGAAM_VARIANT]
    return str(meta["repo_id"]), str(meta["revision"])


def _gigaam_model_cached() -> bool:
    """True if the GigaAM artifact files are already in the HF cache."""
    from huggingface_hub import snapshot_download

    repo_id, revision = _gigaam_repo_revision()
    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=True,
            allow_patterns=["config.json", "manifest.json", "model.safetensors"],
        )
        return True
    except Exception:  # noqa: BLE001
        return False


def _download_gigaam() -> None:
    """Fetch the GigaAM artifact into the HF cache (resumable)."""
    from huggingface_hub import snapshot_download

    repo_id, revision = _gigaam_repo_revision()
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=["config.json", "manifest.json", "model.safetensors"],
    )


def _load_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        t0 = time.perf_counter()
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        local_path = _local_model_path()
        logger.info("loading mlx-whisper model %s …", MODEL_REPO)
        # Route the warmup through mlx-whisper's own ModelHolder so the
        # instance warmed here is the same object mlx_whisper.transcribe()
        # reuses later. A private load_model() copy would double the
        # weight footprint (~1.6 GB x 2).
        _model = ModelHolder.get_model(local_path, mx.float16)
        dt = time.perf_counter() - t0
        logger.info("model loaded in %.1fs (will stay in RAM)", dt)
        return _model


def _ensure_utf8_locale() -> None:
    """Re-point LC_CTYPE at a UTF-8 locale when the process runs ASCII.

    The py2app-launched bundle process can start with a POSIX/C locale,
    so ``Path.read_text()`` defaults to ASCII and the GigaAM artifact's
    ``config.json`` (Cyrillic vocabulary) fails to decode. ``open()``
    queries the locale on every call, so fixing it here takes effect
    immediately — no interpreter restart needed. No-op when the
    preferred encoding is already UTF-8 (venv, normal shells).
    """
    import locale

    def _norm(enc: str) -> str:
        return enc.lower().replace("-", "").replace("_", "")

    try:
        if _norm(locale.getencoding()) == "utf8":
            return
        for candidate in ("en_US.UTF-8", "C.UTF-8", "UTF-8"):
            try:
                locale.setlocale(locale.LC_CTYPE, candidate)
            except locale.Error:
                continue
            if _norm(locale.getencoding()) == "utf8":
                logger.info("LC_CTYPE was ASCII — switched to %s", candidate)
                return
    except Exception:  # noqa: BLE001 — best effort; load may still fail
        logger.debug("could not repair locale", exc_info=True)


def _load_gigaam():
    global _gigaam_model
    if _gigaam_model is not None:
        return _gigaam_model
    with _gigaam_lock:
        if _gigaam_model is not None:
            return _gigaam_model
        t0 = time.perf_counter()
        from gigaam_multilingual_mlx import load_model as _gigaam_load_model

        _ensure_utf8_locale()
        logger.info("loading gigaam-multilingual-mlx (%s) …", GIGAAM_VARIANT)
        _gigaam_model = _gigaam_load_model(variant=GIGAAM_VARIANT)
        # Materialize the weights eagerly, mirroring the whisper path:
        # load_weights() only memory-maps the safetensors, so without
        # this the ~1.2 GB page-in cost lands on the user's FIRST
        # dictation instead of on the warmup.
        import mlx.core as mx

        mx.eval(_gigaam_model.parameters())
        dt = time.perf_counter() - t0
        logger.info("gigaam model loaded in %.1fs (will stay in RAM)", dt)
        return _gigaam_model


def warm(model_kind: str = MODEL_KIND_LOCAL) -> None:
    """Force-load a local model (used by ``dictate-mac warmup``)."""
    if model_kind == MODEL_KIND_GIGAAM:
        _run_on_mlx_thread(_load_gigaam)
        return
    _run_on_mlx_thread(_load_model)


def model_loaded() -> bool:
    return _model is not None


def gigaam_loaded() -> bool:
    return _gigaam_model is not None


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode a mono float32 ``[-1, 1]`` buffer as 16-bit PCM WAV in memory."""
    if audio.size == 0:
        audio_int16 = np.zeros(0, dtype=np.int16)
    else:
        audio_int16 = np.clip(audio * 32767.0, -32768.0, 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def _http_error_detail(response: requests.Response) -> str:
    body = (response.text or "").strip()
    if len(body) > 200:
        body = body[:200] + "…"
    return f"HTTP {response.status_code} {body!r}"


def check_api_model_available(
    endpoint: str,
    api_key: str,
    model_id: str,
    *,
    timeout: float = DEFAULT_CHECK_TIMEOUT,
) -> None:
    """Validate the API endpoint, key and model id before persisting.

    Sends ``GET {endpoint}/models`` with an ``Authorization: Bearer``
    header and confirms the configured ``model_id`` appears in the
    returned list. Raises :class:`RuntimeError` with a category-specific
    message on any failure. The API key is never logged or included
    in error strings.
    """
    base = normalize_endpoint(endpoint)
    if not base:
        raise RuntimeError("Endpoint is empty")
    if not endpoint_scheme_ok(base):
        raise RuntimeError(
            f"Endpoint {base!r} must start with http:// or https://"
        )
    if not api_key:
        raise RuntimeError("API key is empty")
    if not model_id:
        raise RuntimeError("Model ID is empty")

    url = f"{base}/models"
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Could not reach {base}: request timed out after {timeout:.0f}s"
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not reach {base}: {exc.__class__.__name__}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Could not reach {base}: {exc.__class__.__name__}"
        ) from exc

    if response.status_code in (401, 403):
        raise RuntimeError(
            f"Authentication failed — check the API key (HTTP {response.status_code})"
        )
    if response.status_code == 404:
        raise RuntimeError(
            f"Models endpoint not found — confirm the URL ends with /v1 "
            f"(current: {base})"
        )
    if not response.ok:
        raise RuntimeError(
            f"Endpoint returned {_http_error_detail(response)}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Endpoint returned non-JSON body at {url}"
        ) from exc

    available = payload.get("data")
    if not isinstance(available, list):
        raise RuntimeError(
            f"Endpoint {url} returned JSON without a 'data' array — "
            "not an OpenAI-compatible models endpoint?"
        )

    for entry in available:
        if isinstance(entry, dict) and entry.get("id") == model_id:
            return
    raise RuntimeError(
        f"Model ID '{model_id}' not found at {base} (response listed "
        f"{len(available)} model(s))"
    )


def _transcribe_api(
    audio: np.ndarray,
    endpoint: str,
    api_key: str,
    model_id: str,
    *,
    language: str = "auto",
    timeout: float = DEFAULT_API_TIMEOUT,
) -> str:
    """POST the audio as 16 kHz mono WAV to ``{endpoint}/audio/transcriptions``.

    When ``language`` is set to a concrete ISO-639-1 code (``"ru"``, ``"en"``,
    …), it is forwarded to the gateway so the model skips its own
    language detection. With ``"auto"`` (or any other sentinel) the
    field is omitted and the gateway falls back to auto-detection —
    saving the ~0.3-0.8 s detection cost when the user has pinned a
    language.
    """
    if audio is None or audio.size == 0:
        return ""
    base = normalize_endpoint(endpoint)
    if not base:
        raise RuntimeError("Endpoint is empty")
    if not api_key or not model_id:
        raise RuntimeError("Missing API credentials for API-mode ASR")

    wav_bytes = _audio_to_wav_bytes(audio, _ASR_SAMPLE_RATE)
    url = f"{base}/audio/transcriptions"
    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    data = {"model": model_id, "response_format": "json"}
    if language and language != "auto":
        data["language"] = language

    t0 = time.perf_counter()
    try:
        response = requests.post(
            url,
            files=files,
            data=data,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"ASR request timed out after {timeout:.0f}s"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"ASR request failed: {exc.__class__.__name__}"
        ) from exc

    dt = time.perf_counter() - t0
    if not response.ok:
        logger.warning(
            "ASR API HTTP %d after %.2fs (model=%s, endpoint=%s)",
            response.status_code,
            dt,
            model_id,
            base,
        )
        raise RuntimeError(
            f"ASR endpoint returned {_http_error_detail(response)}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"ASR endpoint returned non-JSON body (HTTP 200)"
        ) from exc

    text = (payload.get("text") or "").strip()
    logger.info(
        "api recognition done in %.2fs (%d chars, model=%s, endpoint=%s)",
        dt,
        len(text),
        model_id,
        base,
    )
    return text


def _transcribe_local(audio: np.ndarray, language: str) -> str:
    """Run the in-process mlx-whisper model (pinned to the MLX thread)."""
    if audio is None or audio.size == 0:
        return ""
    return _run_on_mlx_thread(_transcribe_local_mlx, audio, language)


def _transcribe_local_mlx(audio: np.ndarray, language: str) -> str:
    from dictate_mac.config import AUTO as CONFIG_AUTO

    global _first_call_done
    _load_model()

    import mlx.core as mx
    import mlx_whisper

    local_path = _local_model_path()

    whisper_lang: Optional[str] = None if language == CONFIG_AUTO else language

    t0 = time.perf_counter()
    try:
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=local_path,
            language=whisper_lang,
            task=TASK,
            fp16=True,
            verbose=False,
            condition_on_previous_text=True,
            temperature=(0.0, 0.2, 0.4),
            initial_prompt=WHISPER_INITIAL_PROMPTS.get(language),
            # Sampled fallback temperatures (fired on compression-ratio
            # / logprob failures) take the best of 3 trajectories —
            # the mlx-whisper stand-in for beam search, which upstream
            # never implemented (raises NotImplementedError).
            best_of=3,
        )
    finally:
        # Return the Metal free-list cache to the OS. The decoder grows
        # its KV buffers by concatenation at every token step, so each
        # decode leaves hundreds of MB of unique-size buffers cached;
        # without this, the footprint climbs with every dictation.
        # Measured cost of re-allocating on the next run: ~0.01-0.05 s.
        cached = mx.get_cache_memory()
        if cached:
            mx.clear_cache()
            logger.debug("returned %.0f MB of MLX buffer cache to the OS", cached / 1e6)
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


def _transcribe_gigaam(audio: np.ndarray) -> str:
    """Run the in-process GigaAM CTC model (pinned to the MLX thread)."""
    if audio is None or audio.size == 0:
        return ""
    return _run_on_mlx_thread(_transcribe_gigaam_mlx, audio)


def _gigaam_transcribe_chunks(model, audio: np.ndarray) -> str:
    """Decode a long buffer in overlapping windows, stitching by midpoint.

    Mirrors the chunking strategy of the gigaam-multilingual-mlx
    package's own ``service.transcribe_file``: fixed 20 s windows with
    2 s overlap; each decoded word is kept only in the window where its
    temporal midpoint falls, so overlap regions contribute no
    duplicated fragments. Word timing comes from CTC token frames.
    """
    import mlx.core as mx

    sr = _ASR_SAMPLE_RATE
    size = round(GIGAAM_CHUNK_SECONDS * sr)
    overlap = round(GIGAAM_OVERLAP_SECONDS * sr)
    step = size - overlap
    words_out: list[str] = []
    n_chunks = 0

    start = 0
    while start < audio.size:
        end = min(start + size, audio.size)
        samples = audio[start:end]
        is_last = end == audio.size
        n_chunks += 1

        log_probs, lengths = model(mx.array(samples)[None, :], mx.array([len(samples)]))
        decoded = model.greedy_decode(log_probs, lengths)[0]
        # Free each window's Metal buffers before decoding the next —
        # the caller's finally-clear only runs once at the end.
        cached = mx.get_cache_memory()
        if cached:
            mx.clear_cache()

        enc_len = int(np.asarray(lengths)[0])
        if enc_len > 0:
            chunk_dur = len(samples) / sr
            shift = chunk_dur / enc_len  # seconds per encoded frame
            keep_lo = 0.0 if start == 0 else GIGAAM_OVERLAP_SECONDS / 2
            keep_hi = chunk_dur if is_last else chunk_dur - GIGAAM_OVERLAP_SECONDS / 2

            chars: list[str] = []
            frames: list[int] = []

            def flush() -> None:
                if chars and frames:
                    w_start = frames[0] * shift
                    w_end = (frames[-1] + 1) * shift
                    mid = (w_start + w_end) / 2
                    # The overlap is split at its midpoint: each word is
                    # kept only by the window whose core contains it.
                    # (The package's service.py drops the keep_lo guard
                    # for the final window, which duplicates overlap
                    # words there — we keep the guard in all windows.)
                    kept = (
                        keep_lo <= mid <= keep_hi
                        if is_last
                        else keep_lo <= mid < keep_hi
                    )
                    if kept:
                        words_out.append("".join(chars))
                chars.clear()
                frames.clear()

            for token_id, frame in zip(decoded["token_ids"], decoded["token_frames"]):
                if model.config.vocabulary[token_id] == " ":
                    flush()
                else:
                    chars.append(model.config.vocabulary[token_id])
                    frames.append(frame)
            flush()

        if is_last:
            break
        start += step

    logger.info("gigaam long-form: decoded %d chunks (%.1fs audio)", n_chunks, audio.size / sr)
    return " ".join(words_out).strip()


def _transcribe_gigaam_mlx(audio: np.ndarray) -> str:
    global _gigaam_first_call_done
    model = _load_gigaam()

    import mlx.core as mx

    t0 = time.perf_counter()
    try:
        if audio.size > round(GIGAAM_CHUNK_SECONDS * _ASR_SAMPLE_RATE):
            text = _gigaam_transcribe_chunks(model, audio)
        else:
            log_probs, lengths = model(mx.array(audio, dtype=mx.float32))
            decoded = model.greedy_decode(log_probs, lengths)
            text = str(decoded[0].get("text") or "").strip()
    finally:
        # Same Metal free-list discipline as the whisper path — CTC
        # has no autoregressive KV growth, but the encoder's unique-size
        # intermediates would otherwise sit in the cache.
        cached = mx.get_cache_memory()
        if cached:
            mx.clear_cache()
            logger.debug("returned %.0f MB of MLX buffer cache to the OS", cached / 1e6)
    dt = time.perf_counter() - t0
    if not _gigaam_first_call_done:
        _gigaam_first_call_done = True
        logger.info("first gigaam recognition done in %.2fs — model warm in RAM", dt)
    else:
        logger.info("gigaam recognition done in %.2fs (%d chars)", dt, len(text))
    return text


def transcribe(
    audio: np.ndarray,
    language: str = "auto",
    *,
    model_kind: str = MODEL_KIND_LOCAL,
    api_endpoint: str = "",
    api_key: str = "",
    api_model_id: str = "",
    api_timeout: float = DEFAULT_API_TIMEOUT,
) -> str:
    """Run ASR on a mono 16 kHz float32 buffer; return plain text.

    Dispatches to :func:`_transcribe_local`, :func:`_transcribe_gigaam`
    or :func:`_transcribe_api` based on ``model_kind``. Callers passing
    only ``(audio, language=)`` keep the historical behaviour. The
    GigaAM path is a fixed multilingual CTC and ignores ``language``.

    On any failure in the API path a :class:`RuntimeError` is raised
    with a categorised message. The local paths keep their legacy
    behaviour: errors during the warmup never propagate.
    """
    if model_kind == MODEL_KIND_API:
        return _transcribe_api(
            audio,
            api_endpoint,
            api_key,
            api_model_id,
            language=language,
            timeout=api_timeout,
        )
    if model_kind == MODEL_KIND_GIGAAM:
        return _transcribe_gigaam(audio)
    return _transcribe_local(audio, language)
