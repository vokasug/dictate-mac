"""Command-line interface and entry point for dictate-mac.

Responsibilities:

* Parse global options: ``--quiet``, ``--log-level``, ``--output``.
* Default invocation (``dictate-mac`` with no subcommand) — launch
  the menu bar app (Phase 9). This is the recommended interface for
  everyday use.
* Subcommand ``daemon`` — start the asyncio state machine in plain
  CLI mode (no menu bar). Same code path as the menu bar app, minus
  rumps. Useful for SSH, tmux, CI.
* Subcommand ``warmup`` — download the model into the Hugging Face
  cache, then optionally run a microphone test, and exit.
* Subcommand ``selftest`` — headless smoke test covering model load,
  VAD trimming, ASR, and typer dispatch routing. Exits non-zero on
  any failure.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Sequence

from dictate_mac import __version__
from dictate_mac.config import (
    MODEL_KINDS,
    MODEL_KIND_API,
    MODEL_KIND_LOCAL,
    normalize_endpoint,
)
from dictate_mac.logutils import (
    DEFAULT_LOG_LEVEL,
    LOG_FORMAT,
    configure_logging,
    is_app_bundle,
    log_level_from_argv,
)

logger = logging.getLogger("dictate_mac")

logger = logging.getLogger("dictate_mac")


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO-level log messages.",
    )
    common.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Set the logging verbosity (default: %(default)s).",
    )
    common.add_argument(
        "--output",
        choices=("quartz", "osascript"),
        default="quartz",
        help=(
            "Keystroke injection backend. 'quartz' uses raw CGEvent "
            "(default, best for Citrix). 'osascript' uses System Events "
            "via AppleScript — fallback when Citrix drops characters."
        ),
    )
    common.add_argument(
        "--language",
        default="auto",
        metavar="CODE",
        help=(
            "Recognition language for CLI subcommands. Either an "
            "ISO-639-1 code understood by mlx-whisper (e.g. 'ru', 'en', "
            "'de') or the sentinel 'auto' to let Whisper detect the "
            "language from the first 30 seconds of audio. Default: "
            "'auto'. The CLI does not consult or write the persisted "
            "config file — use the menu bar entry point (default "
            "dictate-mac invocation) for that."
        ),
    )
    common.add_argument(
        "--model-kind",
        dest="model_kind",
        choices=MODEL_KINDS,
        default=MODEL_KIND_LOCAL,
        help=(
            "ASR backend. 'local' (default) runs mlx-whisper "
            "in-process. 'api' POSTs 16 kHz mono WAVs to a "
            "OpenAI-compatible endpoint; requires --api-endpoint, "
            "--api-key and --model-id. The CLI does NOT verify the "
            "endpoint at startup — failures surface in the log on "
            "the first recording, exactly like with the local "
            "backend."
        ),
    )
    common.add_argument(
        "--api-endpoint",
        dest="api_endpoint",
        default="",
        metavar="URL",
        help=(
            "OpenAI-compatible base URL for the API backend "
            "(e.g. '<your-endpoint>/v1'). Only meaningful with "
            "--model-kind=api; trailing '/' is stripped."
        ),
    )
    common.add_argument(
        "--api-key",
        dest="api_key",
        default="",
        metavar="KEY",
        help=(
            "Bearer token for the API backend. Only meaningful with "
            "--model-kind=api. NEVER logged."
        ),
    )
    common.add_argument(
        "--model-id",
        dest="model_id",
        default="",
        metavar="ID",
        help=(
            "Model id the gateway should use for transcription. "
            "Only meaningful with --model-kind=api; consult the "
            "gateway's documentation for supported values."
        ),
    )

    parser = argparse.ArgumentParser(
        prog="dictate-mac",
        description=(
            "Local Russian voice dictation for macOS. "
            "Right Option starts/stops recording; recognized text is typed "
            "into the focused window (including Citrix)."
        ),
        parents=[common],
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"dictate-mac {__version__}",
    )

    sub = parser.add_subparsers(dest="command", required=False)

    daemon_p = sub.add_parser(
        "daemon",
        help=(
            "Start the dictation daemon in CLI mode (no menu bar). "
            "Same code path as the menu bar app, without rumps. "
            "Useful for SSH sessions, tmux, and CI."
        ),
        parents=[common],
    )

    warmup = sub.add_parser(
        "warmup",
        help=(
            "Download the mlx-whisper model into the Hugging Face cache "
            "and (optionally) run a microphone sanity check. Exits 0 on "
            "success."
        ),
    )
    warmup.add_argument(
        "--skip-mic-test",
        action="store_true",
        help="Do not record from the microphone during warmup.",
    )
    warmup.add_argument(
        "--mic-test-seconds",
        type=float,
        default=2.0,
        help="Duration of the microphone sanity test (default: %(default)s).",
    )

    selftest = sub.add_parser(
        "selftest",
        help=(
            "Run headless checks: model load, VAD trimming, ASR smoke, "
            "typer dispatch routing. Optionally records from the "
            "microphone. Exits 0 if all checks pass, 1 otherwise."
        ),
    )
    selftest.add_argument(
        "--no-mic",
        action="store_true",
        help="Skip the microphone roundtrip check.",
    )
    selftest.add_argument(
        "--language",
        default="auto",
        metavar="CODE",
        help=(
            "Recognition language used for the ASR smoke and mic "
            "roundtrip checks. Same values as the global --language "
            "flag. Default: 'auto'."
        ),
    )

    return parser


def _configure_logging(level: str) -> None:
    """Set up logging handlers per Phase 10.

    In ``.app`` bundle mode logs go to
    ``~/Library/Logs/dictate-mac/dictate-mac.log`` (truncate-on-start);
    in CLI mode they go to stderr.

    Thin wrapper around :func:`dictate_mac.logutils.configure_logging`
    kept for backwards compatibility with callers that have already
    parsed ``--log-level`` from argparse.
    """
    configure_logging(level=level)


def _download_model() -> Path:
    """Trigger the Hugging Face download for mlx-community/whisper-large-v3-turbo.

    mlx-whisper itself downloads weights lazily; we use ``huggingface_hub``
    so the download progress is visible and resumable.
    """
    from huggingface_hub import snapshot_download

    repo_id = "mlx-community/whisper-large-v3-turbo"
    logger.info("downloading %s …", repo_id)
    local_dir = Path(snapshot_download(repo_id=repo_id))
    logger.info("model ready at %s", local_dir)
    return local_dir


def _mic_test(seconds: float) -> None:
    """Record a short clip and save it to /tmp for inspection."""
    import numpy as np
    import sounddevice as sd

    logger.info("recording %.1fs from the default microphone …", seconds)
    samplerate = 16000
    frames = int(seconds * samplerate)
    recording = sd.rec(frames, samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    peak = float(np.abs(recording).max())
    rms = float(np.sqrt(np.mean(recording**2)))
    logger.info(
        "mic test done — peak=%.3f rms=%.4f (samples=%d)", peak, rms, recording.size
    )
    if rms < 1e-4:
        logger.warning(
            "microphone looks silent (rms<1e-4). check input volume or "
            "microphone permissions for Terminal."
        )


def cmd_warmup(args: argparse.Namespace) -> int:
    _download_model()
    # Pre-load the model into RAM so the first real dictation is fast.
    from dictate_mac.transcriber import warm

    warm()
    if not args.skip_mic_test:
        try:
            _mic_test(args.mic_test_seconds)
        except Exception as exc:  # noqa: BLE001
            logger.error("microphone test failed: %s", exc)
            return 2
    logger.info("warmup complete.")
    return 0


def cmd_daemon(args: argparse.Namespace) -> int:
    """Start the dictation daemon in plain CLI mode (no menu bar).

    Wires up the asyncio state machine, the Quartz CGEventTap hotkey
    watcher, mlx-whisper and the Unicode typer. Runs until Ctrl-C.

    Per Phase 15: this entry point neither reads nor writes the
    persisted ``~/.config/dictate-mac/config.json`` file. All
    settings — language, ASR backend, API credentials — come from
    command-line flags.
    """
    import asyncio

    from dictate_mac.state import DictationMachine, Settings

    if args.model_kind == MODEL_KIND_API:
        endpoint = normalize_endpoint(args.api_endpoint)
        if not endpoint or not args.api_key or not args.model_id:
            print(
                "dictate-mac daemon: --model-kind=api requires "
                "--api-endpoint, --api-key and --model-id",
                file=sys.stderr,
            )
            return 2

    logger.info(
        "starting daemon — output backend=%s, language=%s, "
        "model_kind=%s, endpoint=%s, model_id=%s",
        args.output,
        args.language,
        args.model_kind,
        normalize_endpoint(args.api_endpoint) or "(local)",
        args.model_id or "(local)",
    )
    settings = Settings(
        output_backend=args.output,
        language=args.language,
        model_kind=args.model_kind,
        api_endpoint=normalize_endpoint(args.api_endpoint),
        api_key=args.api_key,
        api_model_id=args.model_id,
    )
    machine = DictationMachine(settings=settings)

    loop = asyncio.new_event_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, machine.stop)
        loop.add_signal_handler(signal.SIGTERM, machine.stop)
    except (NotImplementedError, RuntimeError):
        # SIGINT handlers are not supported in some environments — fall
        # back to KeyboardInterrupt handling in the loop.
        pass

    try:
        loop.run_until_complete(machine.run())
    except KeyboardInterrupt:
        logger.info("interrupted")
    finally:
        loop.close()
    return 0


def cmd_menubar(args: argparse.Namespace) -> int:
    """Launch the menu bar app.

    Sources all persisted settings (language, ASR backend,
    API credentials) from
    ``~/.config/dictate-mac/config.json`` — created on first run if
    needed via :func:`dictate_mac.config.load`. The ``--language``,
    ``--model-kind`` and ``--api-*`` flags from argparse are NOT
    applied here; CLI flags exist only for subcommands that do not
    persist state.
    """
    from dictate_mac.config import load as load_persisted
    from dictate_mac.menubar import run_menubar
    from dictate_mac.state import Settings

    persisted = load_persisted()
    settings = Settings(
        output_backend=args.output,
        language=persisted.language,
        model_kind=persisted.model_kind,
        api_endpoint=persisted.api_endpoint,
        api_key=persisted.api_key,
        api_model_id=persisted.api_model_id,
    )
    return run_menubar(settings)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    if is_app_bundle():
        logger.info("running from .app bundle: %s", sys.executable)
    logger.debug("argv=%r", argv if argv is not None else sys.argv[1:])
    logger.debug("parsed args=%r", args)

    if args.command == "warmup":
        return cmd_warmup(args)
    if args.command == "selftest":
        from dictate_mac.selftest import cmd_selftest

        return cmd_selftest(args)
    if args.command == "daemon":
        return cmd_daemon(args)
    # No subcommand → menu bar app (Phase 9 default). Trying to launch
    # rumps on a non-mac host raises; surface a clean error.
    if args.command is None:
        try:
            import rumps  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            print(
                f"dictate-mac: menu bar app is only supported on macOS ({exc}). "
                "Use `dictate-mac daemon` for a CLI-only mode.",
                file=sys.stderr,
            )
            return 1
        return cmd_menubar(args)
    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
