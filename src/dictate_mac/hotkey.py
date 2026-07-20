"""Hotkey watcher — Quartz CGEvent tap on Right Option.

Phase 6.

The tap runs in its own CFRunLoop on a dedicated background thread.
Events are converted to ``HotkeyEvent`` records and pushed into a
``queue.Queue`` consumed by the asyncio state machine. The bridge is
thread-safe and bounded.

The callback is a plain Python function; PyObjC bridges it to a C
function pointer automatically. We must NOT wrap it in
``ctypes.CFUNCTYPE(c_void_p, ...)`` because the actual callback receives
Objective-C objects (``CGEventTapProxy``, ``CGEventRef``) that cannot be
marshalled into raw void pointers.

The callback signature is **(proxy, type, event, userInfo)** — the 4th
argument is the ``userInfo`` pointer passed to ``CGEventTapCreate`` (we
pass ``None``). Omitting it triggers ``TypeError`` on every key event,
which silently breaks the tap while the CFRunLoop keeps spinning.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger("dictate_mac.hotkey")

# macOS virtual key codes.
#   0x3A = kVK_Option  (Left Option)
#   0x3D = kVK_RightOption (Right Option) — the spec-mandated trigger.
K_VK_RIGHT_OPTION = 0x3D
K_VK_LEFT_OPTION = 0x3A
K_VK_ESCAPE = 0x35
K_VK_OPTION_KEYS = (K_VK_RIGHT_OPTION,)


def _build_mask_blockers() -> int:
    """Build the modifier-mask from Quartz at import time.

    Falls back to known constants if the Quartz import fails (e.g.
    during unit tests on a non-mac host). The literal values below are
    documented by Apple in <CoreGraphics/CGEventTypes.h>:

        kCGEventFlagMaskShift       = 0x00020000
        kCGEventFlagMaskControl     = 0x00040000
        kCGEventFlagMaskCommand     = 0x00100000

    Older revisions of this code used the byte-aligned values
    ``0x000002 / 0x000004 / 0x000008`` which never matched any real
    CGEvent flag bit — the filter was effectively a no-op.
    """
    try:
        from Quartz import (
            kCGEventFlagMaskCommand,
            kCGEventFlagMaskControl,
            kCGEventFlagMaskShift,
        )
        return kCGEventFlagMaskCommand | kCGEventFlagMaskControl | kCGEventFlagMaskShift
    except ImportError:
        return 0x00100000 | 0x00040000 | 0x00020000


_MASK_BLOCKERS = _build_mask_blockers()


class HotkeyEdge(str, Enum):
    PRESS = "press"
    RELEASE = "release"


@dataclass(frozen=True)
class HotkeyEvent:
    edge: HotkeyEdge
    flags: int
    keycode: int


class HotkeyWatcher:
    """Watches the Right Option key globally.

    Usage::

        q: queue.Queue[HotkeyEvent] = queue.Queue()
        watcher = HotkeyWatcher(q)
        watcher.start()
        ...
        watcher.stop()

    The queue is fed only with isolated Right Option presses/releases —
    presses held together with Cmd/Ctrl/Shift are filtered out so we
    don't steal system shortcuts.
    """

    def __init__(
        self,
        output: queue.Queue,
        *,
        keycodes: tuple[int, ...] = K_VK_OPTION_KEYS,
    ) -> None:
        self._output = output
        self._keycodes = keycodes
        self._thread: Optional[threading.Thread] = None
        self._loop = None  # CFRunLoop ref
        self._tap = None
        self._stopping = threading.Event()
        # Last seen CGEventFlags — we use this to detect Option press/
        # release transitions on ``kCGEventFlagsChanged`` events.
        self._last_flags = 0

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("HotkeyWatcher already started")
        self._stopping.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="hotkey-tap", daemon=True
        )
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        if self._thread is None:
            return
        self._stopping.set()
        if self._loop is not None:
            try:
                from Quartz import CFRunLoopStop

                CFRunLoopStop(self._loop)
            except Exception as exc:  # noqa: BLE001
                logger.debug("CFRunLoopStop failed: %s", exc)
        self._thread.join(timeout=timeout)
        self._thread = None

    def is_alive(self) -> bool:
        """True if the tap thread is still running.

        The thread dies (a) cleanly on ``stop()``, (b) when
        ``CGEventTapCreate`` returns NULL (caught before the loop
        starts), or (c) when the callback raises — a crash in the
        callback kills the thread but leaves the daemon otherwise
        unaware. The state machine polls this and exits when the tap
        thread is gone.
        """
        return self._thread is not None and self._thread.is_alive()

    # -- internals --------------------------------------------------------

    def _run_loop(self) -> None:
        try:
            self._install_tap()
        except Exception as exc:  # noqa: BLE001
            logger.error("hotkey tap installation failed: %s", exc)
            self._output.put(_PERMISSION_ERROR)
            return

        from Quartz import (
            CFRunLoopGetCurrent,
            CFRunLoopRun,
            CFRunLoopRunInMode,
            kCFRunLoopDefaultMode,
        )

        self._loop = CFRunLoopGetCurrent()
        keys = ", ".join(f"0x{k:02x}" for k in self._keycodes)
        logger.info(
            "hotkey tap installed — watching Option key(s) [%s]", keys
        )

        # Pump the CFRunLoop forever, processing any queued events.
        # The tap was already enabled by ``_install_tap``; if macOS
        # later disables it (e.g. Input Monitoring not yet granted),
        # the user is expected to grant the permission via System
        # Settings and Quit + reopen the app, as macOS itself
        # suggests. We do not poll ``CGEventTapIsEnabled`` here —
        # re-enabling a disabled tap without permission is a no-op
        # anyway, and the noise in the log only confuses the user.
        while not self._stopping.is_set():
            CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.5, False)

        # stop() flips _stopping, exits the loop, then calls
        # CFRunLoopStop below as a belt-and-braces measure.
        logger.info("hotkey tap run loop exited")

    def _install_tap(self) -> None:
        from Quartz import (
            CGEventTapCreate,
            CGEventTapEnable,
            kCGEventKeyDown,
            kCGEventKeyUp,
            kCGEventFlagsChanged,
            kCGHIDEventTap,
            kCGHeadInsertEventTap,
            kCFRunLoopCommonModes,
            kCGEventTapOptionListenOnly,
        )

        # Plain Python function: PyObjC bridges it to a C trampoline.
        # DO NOT wrap in ``ctypes.CFUNCTYPE(c_void_p, ...)`` — the actual
        # callback receives Objective-C objects that won't marshal into
        # raw void pointers.
        #
        # CGEventTap callback signature is (proxy, type, event, userInfo).
        # The 4th argument is the ``userInfo`` pointer we passed as the
        # last arg to CGEventTapCreate (None). Omitting it raises
        # TypeError on every key event, which kills the callback but
        # leaves CFRunLoop spinning — making the daemon silently idle.
        #
        # We listen on keyDown, keyUp AND flagsChanged. The latter is
        # the only way to observe a lone Option / Shift / Ctrl / Cmd
        # press — modifier keys fire kCGEventFlagsChanged, not keyDown.
        def _callback(_proxy, event_type, event_ref, _userInfo):
            try:
                self._handle_event(event_type, event_ref)
            except Exception:  # noqa: BLE001
                # Never let the callback die: a dead callback + alive
                # CFRunLoop is the worst failure mode (silent daemon).
                logger.exception("hotkey callback error")
            # Return the event unmodified so it propagates further
            # (ListenOnly mode ignores the return value, but returning
            # the original event is the documented contract).
            return event_ref

        event_mask = (
            (1 << kCGEventKeyDown)
            | (1 << kCGEventKeyUp)
            | (1 << kCGEventFlagsChanged)
        )
        self._tap = CGEventTapCreate(
            kCGHIDEventTap,
            kCGHeadInsertEventTap,
            kCGEventTapOptionListenOnly,
            event_mask,
            _callback,
            None,
        )
        if self._tap is None:
            raise RuntimeError(
                "CGEventTapCreate returned NULL — Accessibility AND "
                "Input Monitoring (macOS 14+) must be granted to this "
                "Terminal in System Settings → Privacy & Security"
            )

        from Quartz import (
            CFMachPortCreateRunLoopSource,
            CFRunLoopAddSource,
            CFRunLoopGetCurrent,
        )

        source = CFMachPortCreateRunLoopSource(None, self._tap, 0)
        CFRunLoopAddSource(
            CFRunLoopGetCurrent(), source, kCFRunLoopCommonModes
        )
        CGEventTapEnable(self._tap, True)

    def _handle_event(self, event_type: int, event_ref) -> None:
        from Quartz import (
            CGEventGetFlags,
            CGEventGetIntegerValueField,
            kCGEventKeyDown,
            kCGEventKeyUp,
            kCGEventFlagsChanged,
            kCGKeyboardEventKeycode,
        )

        try:
            flags = int(CGEventGetFlags(event_ref))
        except Exception:  # noqa: BLE001
            flags = 0

        # Modifier transitions (Option / Shift / Ctrl / Cmd pressed
        # or released on their own) arrive as kCGEventFlagsChanged,
        # not kCGEventKeyDown. Branch on event_type.
        if event_type == kCGEventFlagsChanged:
            self._handle_flags_changed(event_ref, flags)
            return

        try:
            keycode = int(CGEventGetIntegerValueField(event_ref, kCGKeyboardEventKeycode))
        except Exception:  # noqa: BLE001
            return

        # Log every keyboard event we receive so the user can diagnose
        # a "nothing happens on Option" mystery — at DEBUG level this
        # prints for every keypress system-wide which is verbose but the
        # only way to see if events are reaching the tap at all.
        logger.debug(
            "tap event_type=%s keycode=0x%02x flags=0x%08x",
            event_type,
            keycode,
            flags,
        )

        if keycode == K_VK_ESCAPE:
            # Esc is the recording-cancel key. It fires a regular
            # keyDown (not flagsChanged) and is meaningful to the state
            # machine only while RECORDING — it is dropped otherwise.
            # Ignore Esc held together with Cmd/Ctrl/Shift so we don't
            # shadow system shortcuts.
            if event_type != kCGEventKeyDown:
                return
            if flags & _MASK_BLOCKERS:
                return
            logger.debug("escape captured (flags=0x%x)", flags)
            try:
                self._output.put_nowait(
                    HotkeyEvent(edge=HotkeyEdge.PRESS, flags=flags, keycode=keycode)
                )
            except queue.Full:
                logger.warning("hotkey queue full — dropping escape")
            return

        if keycode not in self._keycodes:
            return

        # Block if the user is composing a system shortcut that
        # legitimately uses the Option key in combination.
        if flags & _MASK_BLOCKERS:
            logger.debug(
                "option ignored: modifier held (Cmd/Ctrl/Shift, flags=0x%x)",
                flags,
            )
            return

        if event_type == kCGEventKeyDown:
            edge = HotkeyEdge.PRESS
        elif event_type == kCGEventKeyUp:
            edge = HotkeyEdge.RELEASE
        else:
            return

        logger.debug("option %s captured (flags=0x%x)", edge.value, flags)
        try:
            self._output.put_nowait(HotkeyEvent(edge=edge, flags=flags, keycode=keycode))
        except queue.Full:
            logger.warning("hotkey queue full — dropping %s", edge)

    # kCGEventFlagMaskAlternate (Option key) — see <CGEventTypes.h>.
    _OPTION_FLAG = 0x00080000

    def _handle_flags_changed(self, event_ref, flags: int) -> None:
        from Quartz import CGEventGetIntegerValueField, kCGKeyboardEventKeycode

        prev = self._last_flags
        self._last_flags = flags

        was_opt = bool(prev & self._OPTION_FLAG)
        is_opt = bool(flags & self._OPTION_FLAG)
        if was_opt == is_opt:
            # Some other modifier transitioned — not Option.
            return

        try:
            keycode = int(
                CGEventGetIntegerValueField(event_ref, kCGKeyboardEventKeycode)
            )
        except Exception:  # noqa: BLE001
            return

        if keycode not in self._keycodes:
            return

        logger.debug(
            "tap flagsChanged keycode=0x%02x flags=0x%08x (option was=%s is=%s)",
            keycode,
            flags,
            was_opt,
            is_opt,
        )

        # If the user is composing a system shortcut (Cmd/Ctrl/Shift
        # also held) we ignore the Option transition — they probably
        # aren't trying to use our hotkey.
        if flags & _MASK_BLOCKERS:
            logger.debug(
                "option transition filtered: Cmd/Ctrl/Shift held (flags=0x%x)",
                flags,
            )
            return

        edge = HotkeyEdge.PRESS if is_opt else HotkeyEdge.RELEASE
        logger.debug("option %s captured via flagsChanged", edge.value)
        try:
            self._output.put_nowait(
                HotkeyEvent(edge=edge, flags=flags, keycode=keycode)
            )
        except queue.Full:
            logger.warning("hotkey queue full — dropping %s", edge)


_PERMISSION_ERROR = HotkeyEvent(edge=HotkeyEdge.PRESS, flags=-1, keycode=-1)
