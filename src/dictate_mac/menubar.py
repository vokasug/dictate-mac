"""Menu bar UI for dictate-mac — Phase 9 + Phase 15.

`MenubarApp` is the default entry point (``dictate-mac`` with no
subcommand). It owns a single ``NSStatusItem`` whose icon is the SF
Symbol ``waveform`` (template — adapts to light/dark menu bar).

Context menu (top → bottom)
---------------------------

* ``Status: <state>`` — disabled, refreshed every 0.5 s from the
  ``DictationMachine.state`` thread-safe property.
* separator
* ``Recognition language: <X>`` — clickable parent menu item; opens a
  submenu of ``Auto-detect`` + the 99 ISO-639-1 languages supported by
  Whisper. The currently-selected option is prefixed with a checkmark
  (``✓``). A click writes the choice to
  ``~/.config/dictate-mac/config.json`` and updates the in-memory
  settings — no restart required.
* separator
* ``Permissions (reset permission and restart app if not working):``
  — disabled header.
* ``Input Monitoring`` — clickable; opens the Input Monitoring pane of
  System Settings.
* ``Microphone`` — clickable; opens the Microphone pane.
* ``Accessibility`` — clickable; opens the Accessibility pane.
* separator
* ``Open log`` — clickable; opens the daemon log file
  (``~/Library/Logs/dictate-mac/dictate-mac.log`` in app-bundle mode)
  in the user's default app. In CLI mode logs go to stderr and the
  parent directory (or Console.app) is opened instead.
* ``About`` — clickable; opens https://github.com/vokasug/dictate-mac in the
  default browser.
* ``Restart`` — clickable; relaunches the ``.app`` bundle.
* ``Quit`` — clickable; ⌘Q, calls ``rumps.quit_application``.

The permission rows are clickable shortcuts — clicking opens the
matching System Settings pane where the user can flip the toggle.

Threading model
---------------

* **Main thread** — ``NSApp`` (rumps). Owns the status item, the menu,
  and the 0.5 s refresh timer. No long-running work.
* **Worker thread** — asyncio state machine. ``DictationMachine.run``
  drives the warmup → arm → IDLE ⇄ RECORDING → TRANSCRIBING → TYPING
  pump. Started by ``MenubarApp.run`` before rumps blocks the main
  thread.
* **CFRunLoop thread (hotkey)** — the existing ``HotkeyWatcher``,
  installed by ``DictationMachine`` once the warmup reaches ``READY``.

Communication between the main thread (menu bar) and the worker
thread (state machine) is one-way: the worker writes the current state
into a ``threading.Lock``-guarded field; the main thread reads it from
the 0.5 s timer. No new threading primitives.

The Recognition-language submenu lives entirely on the main thread
because both reads (the ``settings`` snapshot we already pass in) and
writes (atomic ``config.save``) are short, blocking, and
thread-safe-by-construction. The next recognition pulls the updated
value through the existing ``state.Settings`` machinery — no new locks.

Lifecycle
---------

1. ``MenubarApp.__init__`` builds the menu and the ``DictationMachine``.
2. ``MenubarApp.run`` spawns the worker thread, then calls
   ``super().run()`` which:
   - sets up the NSStatusItem (replaced with the SF Symbol via the
     ``before_start`` event),
   - starts the registered ``@rumps.timer`` callbacks,
   - blocks the main thread on the NSApp event loop.
3. ``Restart`` spawns an ``osascript`` helper that re-opens the
   bundle after a short delay, then calls ``rumps.quit_application``.
   The helper lives in a separate process so it survives our exit.
4. ``Quit`` (or ⌘Q) calls ``rumps.quit_application`` directly; the
   ``before_quit`` event handler then calls ``DictationMachine.stop``,
   the worker thread exits, and the process exits 0.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import threading
from typing import Optional

import rumps
from rumps import events

from dictate_mac import config as config_mod
from dictate_mac.config import (
    MODEL_KIND_API,
    MODEL_KIND_LOCAL,
    normalize_endpoint,
)
from dictate_mac.state import DictationMachine, Settings, State
from dictate_mac.transcriber import MODEL_REPO

logger = logging.getLogger("dictate_mac.menubar")

# SF Symbol name. Available on macOS 11+ (Big Sur). ``waveform`` is a
# generic voice / audio glyph. If the symbol is missing on an older
# macOS, NSImage returns nil and rumps falls back to the app title.
ICON_NAME = "waveform"
ICON_FALLBACK = "\u23AA"  # ⎪ vertical extension — used as title text fallback

STATUS_REFRESH_SECONDS = 0.5

# User-facing strings
LANG_PARENT_LABEL = "Recognition language:"
MODEL_HEADER = "Model (changing will restart app)"
PERMISSIONS_HEADER = (
    "Permissions (reset permissions and restart app if not working)"
)
ABOUT_URL = "https://github.com/vokasug/dictate-mac"

PERM_LABEL_INPUT_MONITORING = "Input Monitoring"
PERM_LABEL_MICROPHONE = "Microphone"
PERM_LABEL_ACCESSIBILITY = "Accessibility"

# macOS 13+ URL scheme that opens the matching System Settings pane.
# Used by the clickable permission rows.
URL_PRIVACY_INPUT_MONITORING = (
    "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"
)
URL_PRIVACY_MICROPHONE = (
    "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
)
URL_PRIVACY_ACCESSIBILITY = (
    "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
)

# Prefix glyph drawn next to the currently-selected language in the
# Recognition-language submenu. U+2713 = CHECK MARK (✓).
CHECK_GLYPH = "\u2713 "


# State value → human-readable status string for the menu.
_STATUS_LABELS: dict[State, str] = {
    State.STARTING: "Starting…",
    State.DOWNLOADING_MODEL: "Downloading whisper model…",
    State.LOADING_MODEL: "Loading whisper into RAM…",
    State.READY: "Ready — press Right Option to start and stop recording",
    State.RECORDING: "Recording…",
    State.TRANSCRIBING: "Transcribing…",
    State.TYPING: "Typing…",
    State.ERROR: "Error: see logs",
}


class MenubarApp(rumps.App):
    """NSStatusItem-driven dictation daemon.

    Subclass ``rumps.App`` because the rumps event-loop plumbing
    (``@rumps.timer``, ``before_start`` / ``before_quit`` events) is
    wired to instances of that class.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__(
            name="dictate-mac",
            title=ICON_FALLBACK,
            quit_button=None,  # we add Quit explicitly so it sits below About / Restart
        )
        # Disable the default template (we set the icon explicitly).
        self.template = False

        self._settings = settings
        self._machine = DictationMachine(settings=settings)
        self._worker: Optional[threading.Thread] = None
        self._worker_ready = threading.Event()

        # ---- status line (disabled) -----------------------------------
        self._status_item = rumps.MenuItem(
            f"Status: {_STATUS_LABELS[State.STARTING]}", callback=None
        )
        try:
            self._status_item._menuitem.setEnabled_(False)
        except Exception:  # noqa: BLE001
            pass

        # ---- recognition language submenu (Phase 15) ------------------
        # The parent row is clickable and opens a submenu populated
        # with ``Auto-detect`` + the 99 ISO-639-1 languages that
        # mlx-whisper supports. The currently-selected option carries a
        # leading checkmark.
        self._lang_items: dict[str, rumps.MenuItem] = {}
        self._lang_parent = rumps.MenuItem(
            self._format_lang_parent_title(settings.language)
        )
        for code, label in config_mod.menu_items():
            item = rumps.MenuItem(
                self._format_lang_item_title(code, label, settings.language),
                callback=self._make_lang_callback(code),
            )
            self._lang_items[code] = item
            self._lang_parent.add(item)

        # ---- model submenu -------------------------------------------
        # Disabled header + two clickable rows: a local mlx-whisper
        # row that switches back without prompting, and an API row that
        # opens the credentials dialog (see model_settings_dialog.py).
        # The API row's title carries the active endpoint host so the
        # user knows where audio will go without re-opening the dialog.
        self._model_header = rumps.MenuItem(MODEL_HEADER, callback=None)
        try:
            self._model_header._menuitem.setEnabled_(False)
        except Exception:  # noqa: BLE001
            pass

        self._model_local_item = rumps.MenuItem(
            self._format_model_local_title(settings.model_kind),
            callback=self._select_model_local,
        )
        self._model_api_item = rumps.MenuItem(
            self._format_model_api_title(settings.api_endpoint, settings.model_kind),
            callback=self._open_model_api_dialog,
        )

        # ---- permissions header (disabled) ----------------------------
        self._perms_header = rumps.MenuItem(PERMISSIONS_HEADER, callback=None)
        try:
            self._perms_header._menuitem.setEnabled_(False)
        except Exception:  # noqa: BLE001
            pass

        # ---- permission rows (clickable shortcuts) --------------------
        self._perm_input_monitoring = rumps.MenuItem(
            PERM_LABEL_INPUT_MONITORING,
            callback=self._open_input_monitoring_settings,
        )
        self._perm_microphone = rumps.MenuItem(
            PERM_LABEL_MICROPHONE,
            callback=self._open_microphone_settings,
        )
        self._perm_accessibility = rumps.MenuItem(
            PERM_LABEL_ACCESSIBILITY,
            callback=self._open_accessibility_settings,
        )

        # ---- Open log / About / Restart / Quit -----------------------
        self._open_log_item = rumps.MenuItem("Open log", callback=self._open_log_file)
        self._about_item = rumps.MenuItem("About", callback=self._open_about)
        self._restart_item = rumps.MenuItem("Restart", callback=self._restart_app)
        self._quit_item = rumps.MenuItem("Quit", callback=rumps.quit_application)
        try:
            from AppKit import NSCommandKeyMask

            self._quit_item._menuitem.setKeyEquivalent_("q")
            self._quit_item._menuitem.setKeyEquivalentModifierMask_(
                NSCommandKeyMask
            )
        except Exception as exc:  # noqa: BLE001 — key shortcut is a nice-to-have
            logger.debug("could not set ⌘Q key equivalent: %s", exc)

        # ---- assemble the menu ----------------------------------------
        # ``None`` inserts a separator in rumps menus.
        self.menu = [
            self._status_item,
            None,  # separator
            self._model_header,
            self._model_local_item,
            self._model_api_item,
            None,  # separator
            self._lang_parent,
            None,  # separator
            self._perms_header,
            self._perm_input_monitoring,
            self._perm_microphone,
            self._perm_accessibility,
            None,  # separator
            self._open_log_item,
            self._about_item,
            self._restart_item,
            self._quit_item,
        ]


        # NOTE: the permission rows are static shortcuts to System
        # Settings; their labels do not change at runtime.

        # Hook the lifecycle events. ``register`` returns the function
        # so we keep a reference (events uses a set of weakrefs in
        # some versions — explicit refs are safer).
        events.before_start.register(self._install_sf_symbol_icon)
        events.before_quit.register(self._on_before_quit)

    # -- lifecycle --------------------------------------------------------

    def run(self) -> None:  # type: ignore[override]
        """Spawn the worker, then hand control to rumps's NSApp loop."""
        self._worker = threading.Thread(
            target=self._run_machine,
            name="state-machine",
            daemon=True,
        )
        self._worker.start()
        logger.debug("menubar: state machine thread spawned")
        super().run()

    def _run_machine(self) -> None:
        """Worker-thread entry point. Runs the asyncio state machine."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._machine.run())
        finally:
            loop.close()

    def _install_sf_symbol_icon(self) -> None:
        """Replace the default status item image with the SF Symbol.

        Runs once from the ``before_start`` event, after
        ``NSStatusItem`` has been created by rumps. We build the
        ``NSImage`` from the system symbol and set it as a template
        image so it adapts to the menu bar's light/dark mode.

        Falls back to a Unicode title character if the SF Symbol
        cannot be loaded (e.g. macOS < 11, or the symbol name is
        mistyped). The status bar still gets *something* visible.
        """
        try:
            from AppKit import NSImage
        except ImportError:  # pragma: no cover — AppKit is always present on macOS
            logger.warning("AppKit unavailable; status item keeps text title")
            return

        image = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            ICON_NAME, None
        )
        if image is None:
            logger.warning(
                "SF Symbol %r not available on this macOS; "
                "using Unicode title fallback",
                ICON_NAME,
            )
            return
        image.setTemplate_(True)

        rumps_app = getattr(rumps.App, "*app_instance", None)
        nsapp = getattr(rumps_app, "_nsapp", None) if rumps_app else None
        status_item = getattr(nsapp, "nsstatusitem", None) if nsapp else None
        if status_item is None:  # pragma: no cover — defensive
            logger.warning("could not locate NSStatusItem; skipping icon swap")
            return
        status_item.setImage_(image)
        status_item.setTitle_("")
        logger.debug("menubar: installed SF Symbol %r as status icon", ICON_NAME)

    def _on_before_quit(self) -> None:
        """Stop the state machine and wait for the worker to exit.

        Fires from ``rumps.App.run``'s ``applicationWillTerminate_``
        notification. We set the machine's stop event and join the
        worker thread so the model unloads and the CFRunLoop tap
        shuts down cleanly. This is what the user sees as a clean
        "Quit".
        """
        logger.info("menubar: quit requested — stopping state machine")
        self._machine.stop()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        if self._worker is not None and self._worker.is_alive():
            logger.warning(
                "state machine worker did not exit in 2 s — detaching"
            )

    # -- menu actions -----------------------------------------------------

    def _open_url(self, url: str) -> None:
        """Open ``url`` in the user's default browser / app via NSWorkspace.

        Silently swallows all exceptions — opening a URL should never
        crash the menu bar process.
        """
        try:
            from AppKit import NSURL, NSWorkspace

            nsurl = NSURL.URLWithString_(url)
            if nsurl is None:
                logger.warning("NSURL.URLWithString_ returned None for %r", url)
                return
            NSWorkspace.sharedWorkspace().openURL_(nsurl)
        except Exception as exc:  # noqa: BLE001
            logger.debug("open URL %r failed: %s", url, exc)

    def _open_input_monitoring_settings(self, _sender) -> None:
        self._open_url(URL_PRIVACY_INPUT_MONITORING)

    def _open_microphone_settings(self, _sender) -> None:
        self._open_url(URL_PRIVACY_MICROPHONE)

    def _open_accessibility_settings(self, _sender) -> None:
        self._open_url(URL_PRIVACY_ACCESSIBILITY)

    def _open_about(self, _sender) -> None:
        self._open_url(ABOUT_URL)

    def _open_log_file(self, _sender) -> None:
        """Open the daemon log file in the user's default app.

        When running from a bundled ``.app``, logs go to
        ``~/Library/Logs/dictate-mac/dictate-mac.log`` (see
        :mod:`dictate_mac.logutils`). In CLI mode logs go to stderr
        and there is no file to open — we fall back to opening the
        log directory so the user can find where files would go.
        """
        from dictate_mac.logutils import LOG_FILE, is_app_bundle

        if is_app_bundle() and LOG_FILE.exists():
            self._open_url(LOG_FILE.as_uri())
            return

        if is_app_bundle():
            # Log file does not exist yet — open its parent directory.
            parent = LOG_FILE.parent
            if parent.exists():
                self._open_url(parent.as_uri())
                return

        # CLI / source-venv mode: logs go to stderr, no log file on disk.
        # Show a hint in Console.app instead so the user can still find
        # the running daemon's stderr output if captured by their shell.
        try:
            import subprocess

            subprocess.Popen(["open", "-a", "Console"])
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not open Console: %s", exc)

    def _restart_app(self, _sender) -> None:
        """Quit, then re-open the current ``.app`` bundle.

        We can't reliably relaunch ourselves in-process because
        ``rumps.quit_application()`` will tear down NSApp *before* any
        child process we spawn finishes its work. The trick: spawn an
        ``osascript`` helper that runs ``delay 0.5s`` then ``open``s
        the bundle path. osascript lives in a separate process and
        survives our exit, and the 0.5 s delay gives our process time
        to release the TCC-grant file locks before the new instance
        starts up.

        In CLI mode (``NSBundle.mainBundle()`` is undefined) we fall
        back to re-running the same ``sys.executable -m dictate_mac``.
        """
        try:
            app_path = self._current_app_path()
            if app_path:
                subprocess.Popen(
                    [
                        "osascript",
                        "-e",
                        (
                            'delay 0.5\n'
                            f'do shell script "open '
                            f'\\"{app_path}\\"'
                            '"'
                        ),
                    ]
                )
            else:
                # CLI mode — re-launch the same Python entry point.
                subprocess.Popen(
                    [
                        "osascript",
                        "-e",
                        (
                            "delay 0.5\n"
                            f'do shell script '
                            f'"\\"{os.path.abspath(sys_argv0())}\\"'
                            ' &"'
                        ),
                    ]
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not schedule relaunch: %s", exc)

        rumps.quit_application()

    @staticmethod
    def _current_app_path() -> str | None:
        """Return the absolute path of the running ``.app`` bundle, or
        ``None`` if we're running from a bare Python script."""
        try:
            from AppKit import NSBundle

            path = NSBundle.mainBundle().bundlePath()
            if path and path.endswith(".app"):
                return str(path)
        except Exception:  # noqa: BLE001
            pass
        return None

    # -- recognition language submenu (Phase 15) ------------------------

    @staticmethod
    def _format_lang_parent_title(code: str) -> str:
        return f"{LANG_PARENT_LABEL} {config_mod.display_name(code)}"

    @staticmethod
    def _format_lang_item_title(code: str, display: str, active: str) -> str:
        prefix = CHECK_GLYPH if code == active else "  "
        return f"{prefix}{display}"

    @staticmethod
    def _format_model_local_title(active_kind: str) -> str:
        prefix = CHECK_GLYPH if active_kind == MODEL_KIND_LOCAL else "  "
        return f"{prefix}Local ({MODEL_REPO})"

    @staticmethod
    def _format_model_api_title(endpoint: str, active_kind: str) -> str:
        prefix = CHECK_GLYPH if active_kind == MODEL_KIND_API else "  "
        base = normalize_endpoint(endpoint)
        if not base:
            return f"{prefix}API"
        return f"{prefix}API ({base})"

    def _make_lang_callback(self, code: str):
        def callback(_sender) -> None:
            self._set_language(code)

        return callback

    def _set_language(self, code: str) -> None:
        """Apply a new recognition-language choice.

        Updates the in-memory ``Settings`` (the worker thread picks it
        up on the next recognition without a model reload), persists
        it to ``~/.config/dictate-mac/config.json`` via atomic write,
        and refreshes the parent menu title and the per-item
        checkmarks.
        """
        if code == self._settings.language:
            return

        previous = self._settings.language
        self._settings.language = code
        try:
            self._persist_settings()
        except ValueError as exc:
            logger.warning("rejected language=%s: %s", code, exc)
            self._settings.language = previous
            return

        logger.info("recognition language switched: %s -> %s", previous, code)
        self._refresh_lang_menu()

    def _refresh_lang_menu(self) -> None:
        """Resync the Recognition-language submenu with current state.

        Idempotent: no-op when the parent label and item titles
        already match. Called synchronously from ``_set_language`` —
        the 0.5 s status timer no longer drives this refresh.
        """
        active = self._settings.language
        new_parent = self._format_lang_parent_title(active)
        if self._lang_parent.title != new_parent:
            self._lang_parent.title = new_parent
        for code, item in self._lang_items.items():
            display = config_mod.display_name(code)
            new_item = self._format_lang_item_title(code, display, active)
            if item.title != new_item:
                item.title = new_item

    # -- model submenu ----------------------------------------------------

    def _trigger_model_restart(self) -> None:
        """Quit + reopen the bundle so the new ASR backend takes effect.

        The local model is loaded once at startup and held in RAM;
        switching from local to API doesn't release it, and switching
        from API to local would need to trigger a fresh ~30-60 s
        download/load. The cleanest UX is to restart the bundle and
        let the new config take effect at boot.
        """
        logger.info("model switch: triggering restart for new backend")
        try:
            self._restart_app(None)
        except Exception as exc:  # noqa: BLE001
            logger.exception("model switch: restart failed: %s", exc)

    def _select_model_local(self, _sender) -> None:
        """Switch back to the local mlx-whisper model and restart."""
        if self._settings.model_kind == MODEL_KIND_LOCAL:
            return
        logger.info("model switched: %s -> local", self._settings.model_kind)
        self._settings.model_kind = MODEL_KIND_LOCAL
        self._persist_settings()
        self._refresh_model_menu()
        self._trigger_model_restart()

    def _open_model_api_dialog(self, _sender) -> None:
        """Open the API credentials dialog.

        Switching the active backend to API only happens after the
        dialog's OK button passes a ``GET /models`` check against the
        entered endpoint. On success the new config is persisted and
        the bundle is restarted so the new backend takes effect.
        """
        logger.info("api dialog: callback fired")
        try:
            from dictate_mac.model_settings_dialog import (
                ApiModelSettingsDialog,
                ApiModelSettingsResult,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("api dialog: import failed: %s", exc)
            return

        try:
            if not hasattr(self, "_api_dialog") or self._api_dialog is None:
                logger.info("api dialog: constructing new instance")
                self._api_dialog = ApiModelSettingsDialog()
            else:
                logger.info("api dialog: reusing existing instance")
            result = self._api_dialog.show(
                endpoint=self._settings.api_endpoint,
                api_key=self._settings.api_key,
                api_model_id=self._settings.api_model_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("api dialog: show() raised: %s", exc)
            return

        if isinstance(result, ApiModelSettingsResult):
            self._settings.api_endpoint = result.endpoint
            self._settings.api_key = result.api_key
            self._settings.api_model_id = result.api_model_id
            self._settings.model_kind = MODEL_KIND_API
            self._persist_settings()
            logger.info(
                "model switched: -> api (endpoint=%s, model_id=%s)",
                result.endpoint,
                result.api_model_id,
            )
            self._refresh_model_menu()
            self._trigger_model_restart()

    def _refresh_model_menu(self) -> None:
        """Resync the Model submenu with the active settings."""
        new_local = self._format_model_local_title(self._settings.model_kind)
        if self._model_local_item.title != new_local:
            self._model_local_item.title = new_local
        new_api = self._format_model_api_title(
            self._settings.api_endpoint, self._settings.model_kind
        )
        if self._model_api_item.title != new_api:
            self._model_api_item.title = new_api

    def _persist_settings(self) -> None:
        """Atomically persist the in-memory ``Settings`` to the config file."""
        try:
            config_mod.save(
                config_mod.PersistedSettings(
                    language=self._settings.language,
                    model_kind=self._settings.model_kind,
                    api_endpoint=self._settings.api_endpoint,
                    api_key=self._settings.api_key,
                    api_model_id=self._settings.api_model_id,
                )
            )
        except OSError as exc:
            logger.warning("could not persist settings: %s", exc)
        except ValueError as exc:
            logger.warning("rejected settings: %s", exc)
            raise

    # -- refresh timers ---------------------------------------------------

    @rumps.timer(STATUS_REFRESH_SECONDS)
    def _refresh_status(self, _sender) -> None:
        """Pull the latest state from the worker thread and update the menu.

        Runs on the main thread every 0.5 s. Reads the thread-safe
        ``DictationMachine.state`` snapshot — no locks needed in this
        direction. The Recognition-language and Model submenus are not
        touched here: those are refreshed synchronously from their own
        callbacks when the user actually changes them.
        """
        try:
            state = self._machine.state
        except Exception as exc:  # noqa: BLE001
            logger.debug("status refresh: %s", exc)
            state = None
        if state is not None:
            label = _STATUS_LABELS.get(state, state.value)
            new_title = f"Status: {label}"
            if self._status_item.title != new_title:
                self._status_item.title = new_title


def sys_argv0() -> str:
    """Helper used by the CLI-mode restart fallback."""
    import sys

    return sys.argv[0] if sys.argv else "dictate-mac"


def run_menubar(settings: Settings) -> int:
    """Module-level entry point used by ``cli.py``."""
    app = MenubarApp(settings=settings)
    app.run()
    return 0