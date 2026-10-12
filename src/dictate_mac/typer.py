"""Keystroke / paste injectors for dictate-mac.

Backends:

* ``quartz`` (default) — types character-by-character via raw CGEvent
  Unicode events. The canonical path for Unicode typing (same code
  path the OS uses for dead-keys, IME input) with predictable
  forwarding through remote-desktop channels (Citrix ICA, RDP, VNC)
  where the clipboard is not shared.
* ``paste`` — instant insertion via the clipboard + a simulated Cmd+V.
  Fast for long texts in ordinary apps, at the cost of temporarily
  overwriting the clipboard.
* ``osascript`` — System Events fallback.

Why raw CGEvent and not pynput for the quartz backend:

* Direct call to ``CGEventKeyboardSetUnicodeString`` is the canonical
  macOS path for Unicode typing.
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
PASTE_KEYCODE = 9  # kVK_ANSI_V on macOS
UNICODE_CHUNK = 20  # characters per CGEventKeyboardSetUnicodeString call
PASTE_SETTLE_S = 0.15  # pause between setting the clipboard and Cmd+V
PASTE_RESTORE_DELAY_S = 0.25  # pause before restoring the old clipboard


def _post_unicode(chars: str) -> None:
    """Send a single character (or small chunk) as a Unicode keystroke."""
    from Quartz import (
        CGEventCreateKeyboardEvent,
        CGEventKeyboardSetUnicodeString,
        CGEventPost,
        kCGHIDEventTap,
    )

    if not chars:
        return

    down = CGEventCreateKeyboardEvent(None, 0, True)
    up = CGEventCreateKeyboardEvent(None, 0, False)
    CGEventKeyboardSetUnicodeString(down, len(chars), chars)
    CGEventKeyboardSetUnicodeString(up, len(chars), chars)

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


def _post_cmd_v() -> None:
    from Quartz import (
        CGEventCreateKeyboardEvent,
        CGEventPost,
        CGEventSetFlags,
        kCGEventFlagMaskCommand,
        kCGHIDEventTap,
    )

    down = CGEventCreateKeyboardEvent(None, PASTE_KEYCODE, True)
    up = CGEventCreateKeyboardEvent(None, PASTE_KEYCODE, False)
    CGEventSetFlags(down, kCGEventFlagMaskCommand)
    CGEventSetFlags(up, kCGEventFlagMaskCommand)
    CGEventPost(kCGHIDEventTap, down)
    CGEventPost(kCGHIDEventTap, up)


def paste_text(
    text: str,
    restore_delay_s: float = PASTE_RESTORE_DELAY_S,
    restore_clipboard: bool = True,
) -> None:
    """Instant insertion: clipboard + simulated Cmd+V.

    Puts ``text`` on the general pasteboard, posts Cmd+V into the
    focused window, then — when ``restore_clipboard`` is true — restores
    the previous clipboard contents after a short delay (the receiving
    app reads the pasteboard asynchronously, so restoring immediately
    would race it). With ``restore_clipboard=False`` the pasted text
    stays on the clipboard, so the user can paste it again by hand.

    Known limitations: the clipboard is temporarily overwritten, and
    only *string* clipboard content is preserved across the paste —
    images/files previously on the pasteboard are lost. Remote-desktop
    sessions (Citrix, RDP, VNC) that don't share the macOS clipboard
    won't receive anything; use the quartz backend there.
    """
    if not text:
        return
    logger.info(
        "pasting %d chars via clipboard + Cmd+V (restore=%s)",
        len(text),
        restore_clipboard,
    )

    try:
        from AppKit import NSPasteboard, NSPasteboardTypeString

        pb = NSPasteboard.generalPasteboard()
        old = None
        if restore_clipboard:
            saved = pb.stringForType_(NSPasteboardTypeString)
            old = str(saved) if saved is not None else None

        pb.clearContents()
        pb.setString_forType_(text, NSPasteboardTypeString)

        time.sleep(PASTE_SETTLE_S)
        _post_cmd_v()

        if restore_clipboard:
            # Give the receiving app time to read the pasteboard before
            # we overwrite it with the restored content.
            time.sleep(restore_delay_s)
            pb.clearContents()
            if old is not None:
                pb.setString_forType_(old, NSPasteboardTypeString)
    except Exception:  # noqa: BLE001
        logger.exception("paste_text failed")


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
    paste_restore_clipboard: bool = True,
) -> None:
    """Dispatch to the selected backend."""
    if backend == "paste":
        paste_text(text, restore_clipboard=paste_restore_clipboard)
        return
    if backend == "osascript":
        type_text_osascript(text)
        return
    type_text_quartz(text, per_char_delay_ms=per_char_delay_ms)
