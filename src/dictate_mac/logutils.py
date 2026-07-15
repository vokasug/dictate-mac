"""Logging setup for dictate-mac — shared by CLI and menu bar.

Phase 10 — when the daemon is launched from a ``DictateMac.app``
bundle, there is no controlling terminal, so logs go to
``~/Library/Logs/dictate-mac/dictate-mac.log`` (truncated on every
start). In CLI mode (``dictate-mac daemon`` from a terminal), logs
stay on stderr.

We detect ``.app`` mode by looking for the canonical bundle
layout in ``sys.executable``::

    /.../DictateMac.app/Contents/MacOS/DictateMac
    /.../DictateMac.app/Contents/MacOS/python  (alt)

Anything containing ``.app/Contents/MacOS/`` is treated as a bundle.

The setup is a no-op if called twice — ``logging.basicConfig(force=True)``
is used to replace any earlier handlers (the ``fileConfig`` API doesn't
fit because we set it up from code, not from an INI file).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Final

LOG_FORMAT: Final = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"
DEFAULT_LOG_LEVEL: Final = "INFO"
LOG_DIR: Final[Path] = Path.home() / "Library" / "Logs" / "dictate-mac"
LOG_FILE: Final[Path] = LOG_DIR / "dictate-mac.log"

logger = logging.getLogger("dictate_mac.logutils")


def is_app_bundle() -> bool:
    """True when running inside a built ``DictateMac.app`` bundle.

    Detected by ``.app/Contents/MacOS/`` in ``sys.executable``. Returns
    False for ``uv``-launched Python, source-tree runs, and tests.
    """
    exe = sys.executable
    return ".app/Contents/MacOS/" in exe


def is_quiet() -> bool:
    """True if the ``--quiet`` flag was set on the command line."""
    return "--quiet" in sys.argv


def log_level_from_argv(fallback: str = DEFAULT_LOG_LEVEL) -> str:
    """Return the ``--log-level=DEBUG|INFO|...`` value (or fallback).

    Looks at ``sys.argv`` directly because this runs before argparse
    parses anything (we configure logging very early so even early
    import errors land in the right place).
    """
    valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    for arg in sys.argv[1:]:
        if arg.startswith("--log-level="):
            level = arg.split("=", 1)[1].upper()
            return level if level in valid else fallback
        if arg == "--log-level" or arg.startswith("--log-level"):
            # Form ``--log-level DEBUG`` is fine for argparse but we
            # don't have the next-token yet; let argparse handle it.
            continue
    return fallback


def configure_logging(level: str | None = None) -> None:
    """Install handlers according to the launch context.

    * ``.app`` bundle (Finder/Launchpad launch) → truncate + append to
      ``LOG_FILE``.
    * CLI (`daemon`, `warmup`, `selftest`, dev runs) → stderr.

    The handler is a rotating-free ``FileHandler`` with ``mode='w'`` —
    truncated on every start, as agreed in Phase 10.
    """
    if level is None:
        level = DEFAULT_LOG_LEVEL
    if is_quiet() and level == DEFAULT_LOG_LEVEL:
        level = "WARNING"

    numeric = getattr(logging, level, logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric)
    # Replace any handlers a previous configure_logging call left
    # behind (e.g. menubar.py importing this twice in tests).
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(LOG_FORMAT)

    if is_app_bundle():
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            handler: logging.Handler = logging.FileHandler(
                LOG_FILE, mode="w", encoding="utf-8"
            )
        except OSError as exc:
            # Permission issues, sandboxed launch, weird FS — fall back
            # to stderr so the user still sees *something*.
            handler = logging.StreamHandler(sys.stderr)
            logger.warning(
                "could not open log file %s (%s); falling back to stderr",
                LOG_FILE,
                exc,
            )
        else:
            logger.info(
                "logging to %s (truncate-on-start)", LOG_FILE
            )
    else:
        handler = logging.StreamHandler(sys.stderr)

    handler.setFormatter(formatter)
    root.addHandler(handler)


__all__ = [
    "LOG_FORMAT",
    "DEFAULT_LOG_LEVEL",
    "LOG_FILE",
    "is_app_bundle",
    "configure_logging",
    "log_level_from_argv",
    "is_quiet",
]
