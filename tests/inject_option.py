"""Test helper: send a synthetic Right Option keyDown+keyUp pair to the
currently-running DictateMac via Quartz CGEventCreate + CGEventPost.

Run AFTER starting DictateMac.app — the tap will receive the event as
if the user pressed the key. We don't need to be the same process.
"""

from __future__ import annotations

import sys
import time

from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventPost,
    kCGHIDEventTap,
)


# kVK_RightOption = 0x3D (Apple HIToolbox/Events.h, no PyObjC binding).
K_VK_RIGHT_OPTION = 0x3D


def press_and_release_option() -> None:
    down = CGEventCreateKeyboardEvent(None, K_VK_RIGHT_OPTION, True)
    CGEventPost(kCGHIDEventTap, down)

    time.sleep(0.05)

    up = CGEventCreateKeyboardEvent(None, K_VK_RIGHT_OPTION, False)
    CGEventPost(kCGHIDEventTap, up)


if __name__ == "__main__":
    press_and_release_option()
    print("[inject] Right Option press+release sent", file=sys.stderr)
