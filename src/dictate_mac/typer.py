"""Quartz Unicode keystroke injector for dictate-mac.

Phase 5: type text into the focused window via raw CGEvent.

Why raw CGEvent and not pynput:

* Direct call to ``CGEventKeyboardSetUnicodeString`` is the canonical
  macOS path for Unicode typing — it is the same code path the OS uses
  for dead-keys, IME input, etc.
* Predictable forwarding through the Citrix ICA channel when "Send
  Unicode keyboard input" is enabled.
* ``pynput.keyboard.Controller.type`` does roughly the same thing
  through Cocoa but adds an opaque layer that has been reported to drop
  Cyrillic characters in some configurations.
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Iterable

logger = logging.getLogger("dictate_mac.typer")

DEFAULT_PER_CHAR_DELAY_MS = 8
RETURN_KEYCODE = 36  # kVK_Return on macOS
UNICODE_CHUNK = 20  # characters per CGEventKeyboardSetUnicodeString call


def _post_unicode(chars: str) -> None:
    """Send a single character (or small chunk) as a Unicode keystroke."""
    from Quartz import (
        CGEventCreateKeyboardEvent,
        CGEventKeyboardSetUnicodeString,
        CGEventPost,
        kCGHIDEventTap,
        kCGEventFlagMaskCommand,
    )

    if not chars:
        return

    down = CGEventCreateKeyboardEvent(None, 0, True)
    up = CGEventCreateKeyboardEvent(None, 0, False)
    CGEventKeyboardSetUnicodeString(down, len(chars), chars)
    CGEventKeyboardSetUnicodeString(up, len(chars), chars)

    # Release the Command modifier so the keystroke isn't treated as a
    # shortcut. CGEventPost ignores extra flags but it doesn't hurt to
    # be explicit.
    CGEventPost(kCGHIDEventTap, down)
    CGEventPost(kCGHIDEventTap, up)


def _post_return() -> None:
    from Quartz import (
        CGEventCreateKeyboardEvent,
        CGEventPost,
        kCGHIDEventTap,
    )

    down = CGEventCreateKeyboardEvent(None, RETURN_KEYCODE, True)
    up = CGEventCreateKeyboardEvent(None, RETURN_KEYCODE, False)
    CGEventPost(kCGHIDEventTap, down)
    CGEventPost(kCGHIDEventTap, up)


def type_text_quartz(
    text: str,
    per_char_delay_ms: int = DEFAULT_PER_CHAR_DELAY_MS,
    newline_returns: bool = True,
) -> None:
    """Type ``text`` character-by-character using CGEvent Unicode events.

    Newlines are mapped to the Return key (KeyCode 36). Per-character
    delay is necessary because the Citrix ICA channel may coalesce
    closely-spaced events and drop characters.
    """
    if not text:
        return
    delay = per_char_delay_ms / 1000.0
    n = len(text)
    logger.info("typing %d chars (delay=%dms)", n, per_char_delay_ms)

    # Brief pause before injecting so any window-focus transition has
    # time to settle.
    time.sleep(0.15)

    for ch in text:
        if ch == "\n" and newline_returns:
            _post_return()
        else:
            _post_unicode(ch)
        time.sleep(delay)


def type_text_osascript(text: str) -> None:
    """Fallback typer using AppleScript / System Events.

    Used when the Quartz path loses characters inside Citrix Viewer.
    ``text`` is passed via stdin to ``osascript`` so we don't have to
    worry about shell escaping.
    """
    if not text:
        return
    script = (
        'tell application "System Events" to keystroke the_text\n'
        "on set_text(the_text)\n"
        '    tell application "System Events" to keystroke the_text\n'
        "end set_text\n"
        "set_text(" + _applescript_quote(text) + ")"
    )
    logger.info("typing %d chars via osascript", len(text))
    subprocess.run(["osascript", "-e", script], check=False)


def _applescript_quote(s: str) -> str:
    # AppleScript strings are wrapped in double quotes with backslash and
    # double-quote escaping.
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def type_text(
    text: str,
    backend: str = "quartz",
    per_char_delay_ms: int = DEFAULT_PER_CHAR_DELAY_MS,
) -> None:
    """Dispatch to the selected backend."""
    if backend == "osascript":
        type_text_osascript(text)
        return
    type_text_quartz(text, per_char_delay_ms=per_char_delay_ms)
