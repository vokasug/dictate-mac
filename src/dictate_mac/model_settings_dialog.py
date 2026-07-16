"""Modal settings dialog for the API ASR backend.

Opened by :meth:`dictate_mac.menubar.MenubarApp._open_model_api_dialog`
when the user clicks the **API** row in the Model submenu. The dialog is
a free-floating ``NSWindow`` (no app main window exists in a
menubar-only ``LSUIElement=True`` app, which makes a sheet
impractical).

Three editable fields:

* **Endpoint** — ``NSTextField`` for the OpenAI-compatible base URL
  (``http(s)://<host>/v1``). Trailing ``/`` is stripped at save time.
* **API key** — TWO fields stacked at the same coordinates:
  ``NSSecureTextField`` (hidden when revealed) and ``NSTextField``
  (hidden when the field is in secure mode). The eye button to the
  right of the fields toggles which one is visible. The "real" key
  is kept in a Python attribute and copied into the visible field on
  each toggle. Tracking ``controlTextDidChange:`` keeps the two
  views in sync as the user types.

  Earlier attempts swapped the cell of a single field between
  ``NSSecureTextFieldCell`` and ``NSTextFieldCell`` — that is the
  Cocoa pattern from public libraries, but on this PyObjC version
  the secure field editor kept painting bullets on top of the new
  cell even after ``abortEditing``. Two separate fields dodge that
  class of bug entirely.
* **Model ID** — ``NSTextField`` for the model id the gateway should
  use (e.g. the gateway's published model id).

A red ``NSTextField`` between the model id row and the OK/Cancel
buttons shows error messages from the validation step; it is hidden
until validation produces something to say.

Modal lifecycle
---------------

``show()`` runs the dialog via ``NSApp.runModalForWindow_``. The
window stays on screen until one of three things happens:

* **OK with valid input** — the dialog validates the endpoint, key and
  model id by GET-ing ``{endpoint}/models`` with the bearer token and
  confirming the id appears in the response, then stops the modal with
  code 1. The OK button flips its label to ``Checking…`` while the
  network call runs (typically 100–500 ms; up to 5 s on timeout).
* **Cancel** — stops the modal with code 0. No save, no model change.
* **Window close button** — same path as Cancel.

The dialog instance is reused across menu clicks so the AppKit view
tree isn't rebuilt on every show.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSButton,
    NSClosableWindowMask,
    NSColor,
    NSImage,
    NSMakeRect,
    NSObject,
    NSSecureTextField,
    NSTextField,
    NSTitledWindowMask,
    NSWindow,
)

logger = logging.getLogger("dictate_mac.model_settings_dialog")


@dataclass(frozen=True)
class ApiModelSettingsResult:
    endpoint: str
    api_key: str
    api_model_id: str


class _ApiDialogActions(NSObject):
    """Target for the dialog's three buttons."""

    def init(self):
        self = objc.super(_ApiDialogActions, self).init()
        return self

    @objc.selector
    def toggleKeyReveal_(self, sender) -> None:  # noqa: N802
        try:
            logger.info("api dialog: toggleKeyReveal clicked")
            dialog = getattr(self, "dialog", None)
            if dialog is not None:
                dialog._toggle_key_reveal()
        except Exception as exc:  # noqa: BLE001
            logger.exception("api dialog: toggleKeyReveal raised: %s", exc)

    @objc.selector
    def onOk_(self, sender) -> None:  # noqa: N802
        try:
            logger.info("api dialog: onOk clicked")
            dialog = getattr(self, "dialog", None)
            if dialog is not None:
                dialog._on_ok()
        except Exception as exc:  # noqa: BLE001
            logger.exception("api dialog: onOk raised: %s", exc)

    @objc.selector
    def onCancel_(self, sender) -> None:  # noqa: N802
        try:
            logger.info("api dialog: onCancel clicked")
            dialog = getattr(self, "dialog", None)
            if dialog is not None:
                dialog._on_cancel()
        except Exception as exc:  # noqa: BLE001
            logger.exception("api dialog: onCancel raised: %s", exc)


class _ApiDialogDelegate(NSObject):
    """Window delegate that closes the modal on close-button click."""

    def init(self):
        self = objc.super(_ApiDialogDelegate, self).init()
        return self

    @objc.selector
    def windowShouldClose_(self, sender) -> bool:  # noqa: N802
        try:
            logger.info("api dialog: windowShouldClose")
            from AppKit import NSApp

            dialog = getattr(self, "dialog", None)
            if dialog is not None:
                dialog._result = None
                NSApp.stopModalWithCode_(0)
        except Exception as exc:  # noqa: BLE001
            logger.exception("api dialog: windowShouldClose raised: %s", exc)
        return False


class _ApiKeyFieldDelegate(NSObject):
    """NSControl delegate that mirrors text edits back to the dialog.

    The dialog owns the "real" API key value (the plaintext the user
    actually typed). Both the secure and the plain key field point
    here as their delegate; whenever either field receives a
    ``controlTextDidChange:`` notification we update the dialog's
    stored value and re-populate the *other* field, so the two views
    stay in sync even if the user types while the eye toggle is mid-
    transition.
    """

    def init(self):
        self = objc.super(_ApiKeyFieldDelegate, self).init()
        return self

    @objc.selector
    def controlTextDidChange_(self, notification) -> None:  # noqa: N802
        dialog = getattr(self, "dialog", None)
        if dialog is None:
            return
        try:
            field = notification.object()
            if field is None:
                return
            dialog._on_key_field_changed(field)
        except Exception as exc:  # noqa: BLE001
            logger.exception("api dialog: controlTextDidChange raised: %s", exc)


class ApiModelSettingsDialog:
    """Free-floating AppKit window for the API ASR credentials."""

    WINDOW_WIDTH = 460
    WINDOW_HEIGHT = 226
    PADDING = 14
    FIELD_HEIGHT = 22
    LABEL_WIDTH = 90
    ERROR_HEIGHT = 16
    EYE_BUTTON_WIDTH = 40
    GAP_BETWEEN_FIELDS = 10
    GAP_BETWEEN_ERROR_AND_FIELDS = 40
    GAP_BETWEEN_BUTTONS_AND_ERROR = 24
    BUTTON_HEIGHT = 24
    LABEL_FIELD_GAP = 6
    EYE_SYMBOL_POINT_SIZE = 16.0
    TOP_PADDING = 22
    HINT_TEXT_COLOR = "white"

    def __init__(self) -> None:
        self._window = None
        self._endpoint_field = None
        self._key_secure_field = None
        self._key_plain_field = None
        self._model_id_field = None
        self._eye_button = None
        self._error_label = None
        self._ok_button = None
        self._cancel_button = None
        self._key_value: str = ""
        self._key_revealed = False
        self._suppress_key_change = False
        self._result: ApiModelSettingsResult | None = None
        self._targets_obj = None
        self._delegate_obj = None
        self._key_field_delegate_obj = None
        self._config_path_hint = ""

    def show(
        self,
        *,
        endpoint: str = "",
        api_key: str = "",
        api_model_id: str = "",
    ) -> ApiModelSettingsResult | None:
        """Show the dialog modally and return the user's choice."""
        from AppKit import NSApp

        logger.info(
            "api dialog show: entered (endpoint=%r, key=%r, id=%r)",
            endpoint,
            "<set>" if api_key else "<empty>",
            api_model_id,
        )

        # Ensure the application's mainMenu has an Edit menu with
        # Cut/Copy/Paste/Delete/Select All so ⌘C/⌘V/⌘A/⌘X work inside
        # the modal's text fields. The menubar app is LSUIElement=True
        # (no visible menu bar) so this menu never shows on screen but
        # its items still resolve the key equivalents.
        self._ensure_edit_menu()

        if self._window is None:
            logger.info("api dialog show: window is None, calling _build_window")
            try:
                self._build_window()
            except Exception as exc:  # noqa: BLE001
                logger.exception("api dialog show: _build_window failed: %s", exc)
                raise
            logger.info("api dialog show: window built OK (%s)", self._window)

        try:
            self._reset_for_show(
                endpoint=endpoint, api_key=api_key, api_model_id=api_model_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("api dialog show: _reset_for_show failed: %s", exc)
            raise
        self._result = None

        # Make sure the window is key/main and the endpoint field is
        # first responder before entering the modal loop. Without
        # this, key events (including ⌘V) may be delivered to the
        # wrong responder and trigger the system error sound.
        try:
            self._window.makeKeyAndOrderFront_(None)
            NSApp.activateIgnoringOtherApps_(True)
            self._window.makeFirstResponder_(self._endpoint_field)
        except Exception as exc:  # noqa: BLE001
            logger.debug("api dialog: pre-modal first responder setup failed: %s", exc)

        logger.info("api dialog show: calling NSApp.runModalForWindow_")
        NSApp.runModalForWindow_(self._window)
        logger.info("api dialog show: runModalForWindow_ returned")
        self._window.orderOut_(None)

        if self._result is not None:
            logger.info(
                "api settings dialog: saved (endpoint=%s, model_id=%s)",
                self._result.endpoint,
                self._result.api_model_id,
            )
        else:
            logger.debug("api settings dialog: cancelled")
        return self._result

    def _reset_for_show(
        self, *, endpoint: str, api_key: str, api_model_id: str
    ) -> None:
        from dictate_mac.config import config_path, normalize_endpoint

        self._config_path_hint = str(config_path())
        canonical = normalize_endpoint(endpoint)
        self._suppress_key_change = True
        try:
            self._endpoint_field.setStringValue_(canonical or "")
            self._key_value = api_key or ""
            self._key_revealed = False
            self._key_secure_field.setStringValue_(self._key_value)
            self._key_plain_field.setStringValue_(self._key_value)
            self._key_secure_field.setHidden_(False)
            self._key_plain_field.setHidden_(True)
            self._update_eye_icon(revealed=False)
            self._model_id_field.setStringValue_(api_model_id or "")
        finally:
            self._suppress_key_change = False
        self._show_error(None)
        self._ok_button.setTitle_("OK")
        self._ok_button.setEnabled_(True)
        self._cancel_button.setEnabled_(True)

    def _on_key_field_changed(self, field) -> None:
        """Delegate callback when either key field's text changes."""
        if self._suppress_key_change:
            return
        text = field.stringValue() or ""
        self._key_value = text
        other = (
            self._key_plain_field if field is self._key_secure_field
            else self._key_secure_field
        )
        if other.stringValue() != text:
            self._suppress_key_change = True
            try:
                other.setStringValue_(text)
            finally:
                self._suppress_key_change = False

        # NSSecureTextField's echosBullets can lag the displayed bullet
        # count after a bulk paste — the secure cell's stringValue
        # is right (matches the field editor) but the rendered bullets
        # stick to the previous length until the cell is told to
        # re-render. Pushing the same string through the cell and
        # invalidating the display forces an immediate redraw with
        # the correct count.
        if field is self._key_secure_field and not self._key_revealed:
            try:
                field.cell().setEchosBullets_(True)
                field.cell().setStringValue_(text)
                field.setNeedsDisplay_(True)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "api dialog: secure cell redraw nudge failed: %s", exc
                )

    def _build_window(self) -> None:
        logger.info("api dialog build_window: entered")
        style = NSTitledWindowMask | NSClosableWindowMask
        rect = NSMakeRect(0, 0, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style, NSBackingStoreBuffered, False,
        )
        window.setTitle_("API transcription settings")
        window.setReleasedWhenClosed_(False)

        targets = _ApiDialogActions.alloc().init()
        targets.dialog = self
        delegate = _ApiDialogDelegate.alloc().init()
        delegate.dialog = self
        key_field_delegate = _ApiKeyFieldDelegate.alloc().init()
        key_field_delegate.dialog = self
        window.setDelegate_(delegate)

        content = window.contentView()
        x = self.PADDING
        field_x = x + self.LABEL_WIDTH + self.LABEL_FIELD_GAP
        eye_w = self.EYE_BUTTON_WIDTH
        field_w = self.WINDOW_WIDTH - field_x - self.PADDING - eye_w - 4

        button_w = 80
        button_h = self.BUTTON_HEIGHT

        buttons_y_origin = self.PADDING
        error_y_origin = (
            buttons_y_origin + button_h + self.GAP_BETWEEN_BUTTONS_AND_ERROR
        )
        model_id_y_origin = (
            error_y_origin + self.ERROR_HEIGHT + self.GAP_BETWEEN_ERROR_AND_FIELDS
        )
        api_key_y_origin = (
            model_id_y_origin + self.FIELD_HEIGHT + self.GAP_BETWEEN_FIELDS
        )
        endpoint_y_origin = (
            api_key_y_origin + self.FIELD_HEIGHT + self.GAP_BETWEEN_FIELDS
        )

        def make_label(text: str, y_top: int) -> NSTextField:
            r = NSMakeRect(
                x, y_top - self.FIELD_HEIGHT, self.LABEL_WIDTH, self.FIELD_HEIGHT
            )
            label = NSTextField.alloc().initWithFrame_(r)
            label.setStringValue_(text)
            label.setBezeled_(False)
            label.setDrawsBackground_(False)
            label.setEditable_(False)
            label.setSelectable_(False)
            return label

        def make_text_field(y_top: int):
            r = NSMakeRect(
                field_x, y_top - self.FIELD_HEIGHT, field_w, self.FIELD_HEIGHT
            )
            field = NSTextField.alloc().initWithFrame_(r)
            field.setBezeled_(True)
            field.setEditable_(True)
            field.setSelectable_(True)
            field.cell().setWraps_(False)
            field.cell().setLineBreakMode_(4)  # NSLineBreakByClipping
            return field

        def make_secure_field(y_top: int):
            r = NSMakeRect(
                field_x, y_top - self.FIELD_HEIGHT, field_w, self.FIELD_HEIGHT
            )
            field = NSSecureTextField.alloc().initWithFrame_(r)
            field.setBezeled_(True)
            field.setEditable_(True)
            field.setSelectable_(True)
            field.cell().setWraps_(False)
            field.cell().setLineBreakMode_(4)  # NSLineBreakByClipping
            return field

        content.addSubview_(make_label("Endpoint:", endpoint_y_origin))
        endpoint_field = make_text_field(endpoint_y_origin)
        content.addSubview_(endpoint_field)

        content.addSubview_(make_label("API key:", api_key_y_origin))
        key_secure = make_secure_field(api_key_y_origin)
        key_plain = make_text_field(api_key_y_origin)
        key_secure.setDelegate_(key_field_delegate)
        key_plain.setDelegate_(key_field_delegate)
        key_plain.setHidden_(True)
        content.addSubview_(key_secure)
        content.addSubview_(key_plain)

        eye_y_origin = api_key_y_origin - self.FIELD_HEIGHT
        eye_rect = NSMakeRect(
            field_x + field_w + 4, eye_y_origin, eye_w, self.FIELD_HEIGHT
        )
        eye_button = NSButton.alloc().initWithFrame_(eye_rect)
        eye_button.setBezelStyle_(1)
        eye_button.setImagePosition_(2)
        eye_button.setImageScaling_(2)
        eye_button.setTarget_(targets)
        eye_button.setAction_(b"toggleKeyReveal:")
        eye_button.setToolTip_("Show / hide the API key")
        content.addSubview_(eye_button)

        content.addSubview_(make_label("Model ID:", model_id_y_origin))
        model_id_field = make_text_field(model_id_y_origin)
        content.addSubview_(model_id_field)

        error_rect = NSMakeRect(
            x, error_y_origin,
            self.WINDOW_WIDTH - 2 * self.PADDING, self.ERROR_HEIGHT,
        )
        error_label = NSTextField.alloc().initWithFrame_(error_rect)
        error_label.setStringValue_("")
        error_label.setBezeled_(False)
        error_label.setDrawsBackground_(False)
        error_label.setEditable_(False)
        error_label.setSelectable_(False)
        error_label.setTextColor_(NSColor.systemRedColor())
        error_label.setHidden_(True)
        content.addSubview_(error_label)

        ok_rect = NSMakeRect(
            self.WINDOW_WIDTH - self.PADDING - 2 * button_w - 8,
            buttons_y_origin, button_w, button_h,
        )
        cancel_rect = NSMakeRect(
            self.WINDOW_WIDTH - self.PADDING - button_w,
            buttons_y_origin, button_w, button_h,
        )

        ok_button = NSButton.alloc().initWithFrame_(ok_rect)
        ok_button.setTitle_("OK")
        ok_button.setBezelStyle_(1)
        ok_button.setKeyEquivalent_("\r")
        ok_button.setTarget_(targets)
        ok_button.setAction_(b"onOk:")
        content.addSubview_(ok_button)

        cancel_button = NSButton.alloc().initWithFrame_(cancel_rect)
        cancel_button.setTitle_("Cancel")
        cancel_button.setBezelStyle_(1)
        cancel_button.setKeyEquivalent_("\x1b")
        cancel_button.setTarget_(targets)
        cancel_button.setAction_(b"onCancel:")
        content.addSubview_(cancel_button)

        # Tab chain: Endpoint -> API key (visible one) -> Model ID
        # -> OK -> Cancel -> wrap. The secure/plain key fields
        # alternate visibility, but only one is ever the first
        # responder at a time so the chain works regardless of which
        # is visible.
        endpoint_field.setNextKeyView_(key_secure)
        key_secure.setNextKeyView_(key_plain)
        key_plain.setNextKeyView_(model_id_field)
        model_id_field.setNextKeyView_(ok_button)
        ok_button.setNextKeyView_(cancel_button)
        cancel_button.setNextKeyView_(endpoint_field)

        self._window = window
        self._endpoint_field = endpoint_field
        self._key_secure_field = key_secure
        self._key_plain_field = key_plain
        self._model_id_field = model_id_field
        self._eye_button = eye_button
        self._error_label = error_label
        self._ok_button = ok_button
        self._cancel_button = cancel_button
        self._targets_obj = targets
        self._delegate_obj = delegate
        self._key_field_delegate_obj = key_field_delegate
        self._key_value = ""
        self._key_revealed = False
        self._update_eye_icon(revealed=False)

        window.center()
        logger.info("api dialog build_window: window centered")

    def _update_eye_icon(self, revealed: bool) -> None:
        symbol = "eye" if revealed else "eye.slash"
        image = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            symbol, None
        )
        if image is None:
            return
        image.setSize_(
            (self.EYE_SYMBOL_POINT_SIZE, self.EYE_SYMBOL_POINT_SIZE)
        )
        self._eye_button.setImage_(image)
        self._eye_button.setToolTip_(
            ("Hide" if revealed else "Show") + " API key"
        )

    def _show_hint(self, text: str) -> None:
        """Display non-error information (config path) in the error-label slot.

        The error label is reused for two distinct states: the
        default state shows the path of the config file in white so
        the user knows where their settings are persisted, and the
        validation-error state shows the red ``⚠ …`` message.
        """
        self._error_label.setHidden_(False)
        self._error_label.setStringValue_(text)
        self._error_label.setTextColor_(NSColor.whiteColor())

    def _show_error(self, message: str | None) -> None:
        """Show a red validation error, or fall back to the path hint."""
        if not message:
            self._show_hint(self._config_path_hint)
        else:
            self._error_label.setHidden_(False)
            self._error_label.setStringValue_("⚠ " + message)
            self._error_label.setTextColor_(NSColor.systemRedColor())

    def _ensure_edit_menu(self) -> None:
        """Install a hidden Edit menu on NSApp.mainMenu so standard
        text-editing shortcuts (⌘C, ⌘V, ⌘X, ⌘A, Delete) resolve.

        The app is an LSUIElement menubar app (no visible menu bar),
        so NSApp.mainMenu() is normally ``None`` — and without it,
        key-equivalent resolution for the standard edit actions falls
        through to the default responder chain, which is what produces
        the system error beep on ⌘V. Adding an Edit menu whose items
        carry the canonical selectors is the documented Cocoa fix.

        Idempotent: if an Edit menu already exists, do nothing.
        """
        from AppKit import NSApp, NSMenu, NSMenuItem

        try:
            main_menu = NSApp.mainMenu()
        except Exception:  # noqa: BLE001
            main_menu = None

        if main_menu is None:
            main_menu = NSMenu.alloc().init()
            try:
                NSApp.setMainMenu_(main_menu)
            except Exception as exc:  # noqa: BLE001
                logger.debug("api dialog: setMainMenu failed: %s", exc)
                return

        for i in range(main_menu.numberOfItems()):
            item = main_menu.itemAtIndex_(i)
            submenu = item.submenu()
            if submenu is not None and submenu.title() == "Edit":
                return  # already installed

        edit_menu = NSMenu.alloc().initWithTitle_("Edit")
        edit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Edit", None, ""
        )
        main_menu.addItem_(edit_item)
        main_menu.setSubmenu_forItem_(edit_menu, edit_item)

        for title, action, key in (
            ("Cut", "cut:", "x"),
            ("Copy", "copy:", "c"),
            ("Paste", "paste:", "v"),
            ("Delete", "delete:", "\b"),
            ("Select All", "selectAll:", "a"),
        ):
            menu_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                title, action, key
            )
            edit_menu.addItem_(menu_item)

        logger.info("api dialog: installed hidden Edit menu for ⌘C/⌘V/⌘A/⌘X")

    def _set_ok_busy(self, busy: bool) -> None:
        if busy:
            self._ok_button.setEnabled_(False)
        else:
            self._ok_button.setEnabled_(True)

    def _toggle_key_reveal(self) -> None:
        self._key_revealed = not self._key_revealed
        text = self._key_value
        self._suppress_key_change = True
        try:
            if self._key_revealed:
                self._key_secure_field.setHidden_(True)
                self._key_plain_field.setHidden_(False)
                self._key_plain_field.setStringValue_(text)
            else:
                self._key_plain_field.setHidden_(True)
                self._key_secure_field.setHidden_(False)
                self._key_secure_field.setStringValue_(text)
        finally:
            self._suppress_key_change = False
        self._update_eye_icon(revealed=self._key_revealed)

    def _on_ok(self) -> None:
        from AppKit import NSApp

        endpoint_raw = (self._endpoint_field.stringValue() or "").strip()
        api_key = self._key_value
        model_id = (self._model_id_field.stringValue() or "").strip()

        from dictate_mac.config import endpoint_scheme_ok, normalize_endpoint

        endpoint = normalize_endpoint(endpoint_raw)
        if not endpoint:
            self._show_error("Endpoint is empty")
            return
        if not endpoint_scheme_ok(endpoint):
            self._show_error("Endpoint must start with http:// or https://")
            return
        if not api_key:
            self._show_error("API key is empty")
            return
        if not model_id:
            self._show_error("Model ID is empty")
            return

        self._set_ok_busy(True)
        self._show_error(None)

        from dictate_mac.transcriber import check_api_model_available

        try:
            check_api_model_available(
                endpoint, api_key, model_id, timeout=5.0
            )
        except RuntimeError as exc:
            self._set_ok_busy(False)
            self._show_error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self._set_ok_busy(False)
            self._show_error(f"Unexpected error: {exc!r}")
            return

        self._result = ApiModelSettingsResult(
            endpoint=endpoint,
            api_key=api_key,
            api_model_id=model_id,
        )
        NSApp.stopModalWithCode_(1)

    def _on_cancel(self) -> None:
        from AppKit import NSApp

        self._result = None
        NSApp.stopModalWithCode_(0)