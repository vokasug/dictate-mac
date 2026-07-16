"""Persisted user settings.

Stores recognition language, ASR backend choice, and API-mode
credentials in an XDG-style JSON file at
``$XDG_CONFIG_HOME/dictate-mac/config.json`` (default
``~/.config/dictate-mac/config.json``). Atomic write via ``os.replace``,
mode ``0o600``. Created lazily on first save.

CLI subcommands (``daemon``, ``warmup``, ``selftest``) never read or write
this file — the only reader/writer is the menu bar entry point. CLI paths
take all of their settings from command-line flags directly.

Schema
------

v1 (legacy) — recognised for read but never written by the current code:

    ``{"_v": 1, "language": "<iso-639-1 or 'auto'>"}``

v2 (current) — produced on every save:

    ``{"_v": 2, "language": ..., "model_kind": "local|api",
       "api_endpoint": ..., "api_key": ..., "api_model_id": ...}``

v1 files load as ``model_kind="local"`` and empty API fields; the
persisted language is preserved untouched. Subsequent saves rewrite
the file as v2 — no manual migration needed.

Behaviour
---------

* ``detect_system_primary_language()`` queries the macOS
  ``Foundation.NSLocale.preferredLanguages()`` API (PyObjC) and
  returns the first entry that maps to a supported ISO-639-1 code,
  or ``None`` if the call fails or no entry is supported.
* ``resolve_initial_language()`` wraps the detector: a non-supported
  result or ``None`` maps to :data:`AUTO` (``"auto"``).
* ``load()`` reads the config file. If valid, returns the parsed
  settings. If the file is missing or corrupted, runs
  ``resolve_initial_language()``, writes the resolved value back as a
  fresh v2 file, and returns those settings.
* ``save()`` performs atomic replace (tmp file → ``os.replace``) with
  mode ``0o600``; safe to call from the menu bar's main thread.
* ``normalize_endpoint()`` strips whitespace and trailing ``/`` from
  a user-entered endpoint URL so storage and request building always
  operate on a canonical form. The result is what gets written to
  the config file when the modal saves.

The 100 supported languages follow ``mlx_whisper.tokenizer.LANGUAGES``,
matching what Whisper's ``transcribe`` function accepts as
``language=``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, FrozenSet, Optional

logger = logging.getLogger("dictate_mac.config")

SCHEMA_VERSION = 2
AUTO = "auto"
MODEL_KIND_LOCAL = "local"
MODEL_KIND_API = "api"
MODEL_KINDS = (MODEL_KIND_LOCAL, MODEL_KIND_API)

LANGUAGES: Dict[str, str] = {
    "en": "english", "zh": "chinese", "de": "german", "es": "spanish",
    "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese",
    "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan",
    "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian",
    "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
    "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay",
    "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian",
    "ta": "tamil", "no": "norwegian", "th": "thai", "ur": "urdu",
    "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin",
    "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak",
    "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali",
    "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada",
    "et": "estonian", "mk": "macedonian", "br": "breton", "eu": "basque",
    "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian",
    "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
    "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala",
    "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali",
    "af": "afrikaans", "oc": "occitan", "ka": "georgian", "be": "belarusian",
    "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic",
    "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese",
    "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk",
    "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar",
    "bo": "tibetan", "tl": "tagalog", "mg": "malagasy", "as": "assamese",
    "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa",
    "ba": "bashkir", "jw": "javanese", "su": "sundanese", "yue": "cantonese",
}

SUPPORTED_ISO_639_1: FrozenSet[str] = frozenset(LANGUAGES.keys())

LANGUAGE_NAMES: Dict[str, str] = {
    code: name.title() for code, name in LANGUAGES.items()
}


@dataclass(frozen=True)
class PersistedSettings:
    language: str = AUTO
    model_kind: str = MODEL_KIND_LOCAL
    api_endpoint: str = ""
    api_key: str = ""
    api_model_id: str = ""

    def is_valid(self) -> bool:
        if self.language != AUTO and self.language not in SUPPORTED_ISO_639_1:
            return False
        if self.model_kind not in MODEL_KINDS:
            return False
        if self.model_kind == MODEL_KIND_API:
            if not endpoint_scheme_ok(normalize_endpoint(self.api_endpoint)):
                return False
            if not self.api_key:
                return False
            if not self.api_model_id:
                return False
        return True


def normalize_endpoint(endpoint: str) -> str:
    """Return ``endpoint`` with surrounding whitespace and trailing ``/`` stripped.

    Returns an empty string for a falsy input. The result is the canonical
    form stored on disk and used as the base of every request URL the
    ASR builder, the modal validator, and the menu label compose.
    """
    if not endpoint:
        return ""
    return endpoint.strip().rstrip("/")


def endpoint_scheme_ok(endpoint: str) -> bool:
    """``True`` if ``endpoint`` is non-empty and starts with ``http://`` or ``https://``."""
    if not endpoint:
        return False
    return endpoint.startswith("http://") or endpoint.startswith("https://")


def config_path() -> Path:
    """Return the config file path, honoring ``$XDG_CONFIG_HOME``.

    Default is ``~/.config/dictate-mac/config.json``. Parent directory
    is created with mode ``0o700`` if missing.
    """
    base = os.environ.get("XDG_CONFIG_HOME")
    if not base:
        base = str(Path.home() / ".config")
    path = Path(base).expanduser() / "dictate-mac" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def _normalize(bcp47_or_locale: str) -> Optional[str]:
    """Map a macOS locale / BCP-47 code to an ISO-639-1 string.

    Strips the region subtag (``ru-RU`` → ``ru``), lowercases. Returns
    ``None`` if the input cannot yield a 2- or 3-letter code in
    ``SUPPORTED_ISO_639_1``.
    """
    if not bcp47_or_locale:
        return None
    code = bcp47_or_locale.split("-")[0].split("_")[0].strip().lower()
    if not code or len(code) > 3:
        return None
    if code not in SUPPORTED_ISO_639_1:
        return None
    return code


def detect_system_primary_language() -> Optional[str]:
    """Detect the macOS primary system language as an ISO-639-1 code.

    Uses one strategy: ``Foundation.NSLocale.preferredLanguages()``
    (PyObjC, canonical BCP-47, ~1 ms, in-process). Iterates the
    returned array in priority order and returns the first entry that
    normalizes to a supported code.

    Returns ``None`` if the call fails (Foundation unavailable) or no
    entry is supported. Never raises.
    """
    try:
        from Foundation import NSLocale  # type: ignore

        langs = NSLocale.preferredLanguages()
        for raw in langs:
            code = _normalize(str(raw))
            if code is not None:
                return code
    except Exception:
        logger.debug("NSLocale.preferredLanguages unavailable", exc_info=True)
    return None


def resolve_initial_language() -> str:
    """Compute the first-run language choice.

    Either an ISO-639-1 code in :data:`SUPPORTED_ISO_639_1` or
    :data:`AUTO` (``"auto"``).
    """
    code = detect_system_primary_language()
    return code if code in SUPPORTED_ISO_639_1 else AUTO


def _read(path: Path) -> Optional[PersistedSettings]:
    """Parse the config file.

    Returns the parsed :class:`PersistedSettings` on success. Returns
    ``None`` when:

    * the file is missing (treated as the legitimate first-run case),
    * the file is unreadable (OS-level error),
    * the file is malformed (JSON syntax / non-object / non-string
      language field / unsupported code / out-of-range model_kind).

    In every non-missing case :func:`_read` logs a warning so the user
    can find out from the log why defaults are being used. The caller
    (:func:`load`) is responsible for distinguishing the missing case
    from the corrupt-but-present case before any rewrite of the file.

    v1 files (no ``model_kind``) load as ``model_kind="local"`` with
    empty API fields — language is preserved untouched.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning(
            "config %s is invalid JSON (%s); leaving file intact for "
            "manual repair and falling back to defaults",
            path,
            exc,
        )
        return None
    except OSError as exc:
        logger.warning("config %s read failed: %s", path, exc)
        return None

    if not isinstance(data, dict):
        logger.warning(
            "config %s is not a JSON object; leaving file intact "
            "and falling back to defaults",
            path,
        )
        return None

    language = data.get("language")
    if not isinstance(language, str):
        logger.warning(
            "config %s has non-string language field; leaving file "
            "intact and falling back to defaults",
            path,
        )
        return None

    model_kind = data.get("model_kind", MODEL_KIND_LOCAL)
    if not isinstance(model_kind, str) or model_kind not in MODEL_KINDS:
        logger.warning(
            "config %s has unsupported model_kind %r; leaving file "
            "intact and falling back to defaults",
            path,
            model_kind,
        )
        return None

    raw_endpoint = data.get("api_endpoint", "")
    raw_key = data.get("api_key", "")
    raw_model_id = data.get("api_model_id", "")
    if (
        not isinstance(raw_endpoint, str)
        or not isinstance(raw_key, str)
        or not isinstance(raw_model_id, str)
    ):
        logger.warning(
            "config %s has non-string API field(s); leaving file "
            "intact and falling back to defaults",
            path,
        )
        return None

    settings = PersistedSettings(
        language=language,
        model_kind=model_kind,
        api_endpoint=raw_endpoint,
        api_key=raw_key,
        api_model_id=raw_model_id,
    )
    if not settings.is_valid():
        logger.warning(
            "config %s is structurally valid but fails business rules "
            "(model_kind=%r); leaving file intact and falling back to "
            "defaults",
            path,
            settings.model_kind,
        )
        return None
    return settings


def load() -> PersistedSettings:
    """Return the user's persisted settings, creating them if needed.

    Resolution algorithm:

    1. If the file exists and parses as valid
       → return its contents.
    2. If the file exists but is corrupt (any reason logged inside
       :func:`_read`)
       → return a fresh in-memory :class:`PersistedSettings`
       resolved via :func:`resolve_initial_language`. **The corrupt
       file on disk is left untouched** so the user can repair it by
       hand.
    3. If the file does not exist (true first run)
       → resolve the initial language, write a fresh v2 file to disk
       (atomic, ``0o600``), and return it.

    Never raises.
    """
    path = config_path()
    settings = _read(path)
    if settings is not None:
        return settings

    resolved = resolve_initial_language()
    fresh = PersistedSettings(language=resolved)

    if path.exists():
        return fresh

    try:
        save(fresh)
        logger.info("wrote initial config: language=%s", resolved)
    except OSError as exc:
        logger.warning("could not write initial config at %s: %s", path, exc)
    return fresh


def save(settings: PersistedSettings) -> None:
    """Atomically write ``settings`` to the config file.

    Canonicalises ``api_endpoint`` via :func:`normalize_endpoint`
    before writing so on-disk values never carry a stray trailing ``/``
    or surrounding whitespace. Writes to a sibling ``.tmp`` file
    first (``0o600``), then ``os.replace`` on the same filesystem.
    POSIX guarantees the rename is atomic; readers will see either
    the old or the new file, never a half-written one.

    The API key is never logged: only the endpoint, language and
    model kind reach the logger.
    """
    if not settings.is_valid():
        raise ValueError(
            f"invalid settings: language={settings.language!r}, "
            f"model_kind={settings.model_kind!r}"
        )

    canonical = replace(
        settings, api_endpoint=normalize_endpoint(settings.api_endpoint)
    )

    path = config_path()
    payload = {"_v": SCHEMA_VERSION, **asdict(canonical)}
    tmp = path.with_suffix(path.suffix + ".tmp")

    fd = os.open(
        str(tmp),
        os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
        0o600,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise

    os.replace(tmp, path)
    logger.debug(
        "config saved: language=%s, model_kind=%s, endpoint=%s -> %s",
        canonical.language,
        canonical.model_kind,
        canonical.api_endpoint or "(none)",
        path,
    )


def menu_items() -> list[tuple[str, str]]:
    """Return ``(code_or_AUTO, display_label)`` rows for the menubar submenu.

    Order: :data:`AUTO` first, then all 99 supported languages sorted by
    their English display label (``LANGUAGE_NAMES``).
    """
    rows: list[tuple[str, str]] = [(AUTO, "Auto-detect")]
    for code in sorted(SUPPORTED_ISO_639_1, key=lambda c: LANGUAGE_NAMES[c]):
        rows.append((code, LANGUAGE_NAMES[code]))
    return rows


def display_name(code_or_auto: str) -> str:
    """Human-readable label for a stored language code (or ``AUTO``)."""
    if code_or_auto == AUTO:
        return "Auto-detect"
    return LANGUAGE_NAMES.get(code_or_auto, code_or_auto)
