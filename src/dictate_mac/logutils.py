"""Logging setup for dictate-mac — shared by CLI and menu bar.

When the daemon is launched from a ``DictateMac.app``
bundle, there is no controlling terminal, so logs go to
``~/Library/Logs/dictate-mac/dictate-mac.log`` (truncated on every
start). In CLI mode (``dictate-mac daemon`` from a terminal), logs
stay on stderr.

We detect ``.app`` mode by looking for the canonical bundle
layout in ``sys.executable``::

    /.../DictateMac.app/Contents/MacOS/DictateMac
    /.../DictateMac.app/Contents/MacOS/python  (alt)

Anything containing ``.app/Contents/MacOS/`` is treated as a bundle.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final

LOG_FORMAT: Final = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"
DEFAULT_LOG_LEVEL: Final = "INFO"
LOG_DIR: Final[Path] = Path.home() / "Library" / "Logs" / "dictate-mac"
LOG_FILE: Final[Path] = LOG_DIR / "dictate-mac.log"
# Every dictation overwrites this with the VAD-trimmed audio that was
# sent to the ASR backend (16 kHz mono 16-bit PCM). Useful for
# debugging bad recognition: replay exactly what the model heard.
LAST_RECORDING_WAV: Final[Path] = LOG_DIR / "last-recording.wav"

logger = logging.getLogger("dictate_mac.logutils")


def is_app_bundle() -> bool:
    """True when running inside a built ``DictateMac.app`` bundle.

    Detected by ``.app/Contents/MacOS/`` in ``sys.executable``. Returns
    False for ``uv``-launched Python, source-tree runs, and tests.
    """
    exe = sys.executable
    return ".app/Contents/MacOS/" in exe


def configure_logging(level: str | None = None, *, quiet: bool = False) -> None:
    """Install handlers according to the launch context.

    * ``.app`` bundle (Finder/Launchpad launch) → truncate + append to
      ``LOG_FILE``.
    * CLI (`daemon`, `warmup`, `selftest`, dev runs) → stderr.

    ``quiet`` maps the default level to WARNING (the ``--quiet``
    flag); an explicit ``--log-level`` always wins.
    """
    if level is None:
        level = DEFAULT_LOG_LEVEL
    if quiet and level == DEFAULT_LOG_LEVEL:
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
