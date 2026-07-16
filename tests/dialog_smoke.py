"""Standalone dialog smoke test — opens the API settings dialog
without the full menu bar / state machine plumbing. Used to diagnose
whether ``model_settings_dialog.ApiModelSettingsDialog`` itself is
broken or whether the menu bar wiring is the culprit.

Runs only on macOS because the implementation imports ``AppKit``.

Usage::

    ./.venv/bin/python tests/dialog_smoke.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from AppKit import NSApp, NSApplication, NSDate, NSRunLoop
from Foundation import NSAutoreleasePool

from dictate_mac.model_settings_dialog import ApiModelSettingsDialog


def main() -> int:
    pool = NSAutoreleasePool.alloc().init()

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(0)

    dlg = ApiModelSettingsDialog()

    from AppKit import NSObject
    import objc

    class AutoCloser(NSObject):
        @objc.selector
        def perform_(self, _sender):  # noqa: N802
            print("[auto-closer] stopping modal after 2s")
            NSApp.stopModalWithCode_(99)

    print("[main] constructing window (will pop modally) …")

    closer = AutoCloser.alloc().init()
    try:
        closer.performSelector_withObject_afterDelay_(
            b"perform:", None, 2.0
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[main] auto-closer scheduling failed: {exc!r}")

    dlg._build_window()
    print(f"[main] window frame: {dlg._window.frame()}")
    print(f"[main] eye button frame: {dlg._eye_button.frame()}")
    secure_frame = dlg._key_secure_field.frame()
    print(f"[main] secure key field frame: {secure_frame}")
    eye_frame = dlg._eye_button.frame()
    if (
        abs(eye_frame.origin.y - secure_frame.origin.y) < 5
        and abs(eye_frame.size.height - secure_frame.size.height) < 5
    ):
        print("[main] PASS: eye vertically aligned with secure field")
    else:
        print(
            f"[main] FAIL: eye y={eye_frame.origin.y} h={eye_frame.size.height}; "
            f"secure y={secure_frame.origin.y} h={secure_frame.size.height}"
        )

    eye_action = dlg._eye_button.action()
    print(f"[main] eye button action selector: {eye_action!r}")
    eye_target = dlg._eye_button.target()
    print(f"[main] eye button target: {eye_target!r}")
    responds = eye_target.respondsToSelector_(eye_action) if eye_target else False
    print(f"[main] target responds to selector? {responds}")

    print(
        f"[main] initial state: secure visible={not dlg._key_secure_field.isHidden()}, "
        f"plain visible={not dlg._key_plain_field.isHidden()}, "
        f"revealed={dlg._key_revealed}"
    )

    dlg._key_secure_field.setStringValue_("hunter2")
    print(f"[main] set secure field to 'hunter2'")
    print(f"[main]   dialog._key_value={dlg._key_value!r}, "
          f"secure={dlg._key_secure_field.stringValue()!r}, "
          f"plain={dlg._key_plain_field.stringValue()!r}")

    dlg._eye_button.performClick_(None)
    print(
        f"[main] after click 1 (reveal): "
        f"secure visible={not dlg._key_secure_field.isHidden()}, "
        f"plain visible={not dlg._key_plain_field.isHidden()}, "
        f"plain value={dlg._key_plain_field.stringValue()!r}, "
        f"revealed={dlg._key_revealed}"
    )
    dlg._eye_button.performClick_(None)
    print(
        f"[main] after click 2 (hide): "
        f"secure visible={not dlg._key_secure_field.isHidden()}, "
        f"plain visible={not dlg._key_plain_field.isHidden()}, "
        f"secure value={dlg._key_secure_field.stringValue()!r}, "
        f"revealed={dlg._key_revealed}"
    )
    dlg._eye_button.performClick_(None)
    print(
        f"[main] after click 3 (reveal): "
        f"secure visible={not dlg._key_secure_field.isHidden()}, "
        f"plain visible={not dlg._key_plain_field.isHidden()}, "
        f"plain value={dlg._key_plain_field.stringValue()!r}, "
        f"revealed={dlg._key_revealed}"
    )

    print("[main] now opening modal briefly …")
    result = dlg.show(
        endpoint="https://example.test/v1",
        api_key="SAMPLE_KEY_FOR_SMOKE_TEST",
        api_model_id="sample-model",
    )
    print(f"[main] show() returned: {result!r}")

    del pool
    return 0


if __name__ == "__main__":
    raise SystemExit(main())