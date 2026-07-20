# AGENTS.md — dictate-mac

## Documentation rule

This file describes only the **current** state of the project. When planning,
executing, or refactoring work:

1. Plan in conversation or in a scratch file.
2. Execute the change.
3. Update `AGENTS.md` and `README.md` together so they describe the result.
4. Delete the plan and any intermediate "change history" once the result is
   documented. `AGENTS.md` must not contain phase logs, "Phase N" markers,
   past-bug stories, "we used to…", or work-in-progress commentary.

`README.md` is for end users and never carries work history, release
notes, or "what changed and when" sections. It describes what the app
does and how to use it today.

Both files are written in **English only** — no other language in prose,
inline notes, quoted test utterances, or user-facing strings.

Neither `AGENTS.md`, `README.md`, nor any source file committed to this
repository may contain **authentication data, secrets, or personal
information**. The following are forbidden in tracked code and docs:

* credentials of any kind — passwords, passphrases, API keys, OAuth
  client secrets, bearer tokens, access tokens, refresh tokens, private
  keys, SSH keys, GPG keys, certificate private keys, `.env` values,
  `HF_TOKEN` / `OPENAI_API_KEY` / `GITHUB_TOKEN` / `AWS_*` / etc.
* personal account data — real names, email addresses, phone numbers,
  physical addresses, payment info, social handles tied to a private
  individual.
* build-machine-specific identifiers — absolute filesystem paths
  containing a username (`/Users/<name>/...`, `C:\Users\<name>\...`,
  `/home/<name>/...`), hostnames, MAC addresses, internal IPs, build
  CI workspace paths.
* personalised URLs pointing at a private profile, a private repo,
  a private channel, or a private issue tracker.
* placeholder markers that still leak the real value once a reader
  grep-replaces the placeholder back to the original — if a value
  has to be reproduced in a docstring example, use a generic shape
  (`'<build-time venv path>'`, `'<bundle id>'`, `'<your-username>'`)
  that does not survive a literal find.

When examples in code or docs need a "your machine" path, write `~/`
or `<build-time ...>` and never the developer's actual home. When a
URL is required, link the public project resource (the repo, the
release page) — not the maintainer's personal account.

If a previously committed value is found to violate this rule,
remove it in a follow-up commit; do not rewrite history. Add a
sanity grep to the build or test step whenever a new file type or
directory is added under version control, so the next contributor
catches the leak at commit time rather than after a public release.

---

## 1. What this project is

`dictate-mac` is a macOS-only voice dictation daemon with two ASR
backends selected at runtime:

- The user presses **Right Option** to start recording and presses it
  again to stop. Pressing **Esc** while recording cancels it: the
  buffer is discarded, the *Pop* end sound plays, nothing is
  transcribed or typed.
- Audio is captured at 16 kHz mono, trimmed with `silero-vad`, and
  transcribed with one of:
  - **Local** — `mlx-whisper` (`mlx-community/whisper-large-v3-turbo`)
    running in-process. ~1.5 GB resident in RAM.
  - **API** — POST to an OpenAI-compatible
    `/v1/audio/transcriptions` endpoint with a user-provided bearer
    token and model id. No in-process ASR model.
- Recognized text is typed character-by-character into the currently
  focused window via `Quartz.CGEventKeyboardSetUnicodeString` — no
  clipboard paste, so it works inside Citrix Workspace sessions that
  do not forward the macOS clipboard.

The default entry point is `dictate-mac` (no subcommand) — a `rumps`
menu bar app. `dictate-mac daemon` is a CLI-only mode for SSH/CI/tmux.

Switching between the two ASR backends is a **restart-required**
operation: the local model is loaded once at startup and held in RAM,
so flipping backends mid-session would either keep the old model
wasted in RAM (Local → API) or block on a fresh ~30-60 s
download/load (API → Local). On a backend switch the daemon re-execs
itself so the new mode takes effect at boot with no model download
when API was just selected.

Python 3.13 only (no 3.14 wheel for `mlx`); Apple Silicon only; ≥ 8 GB
RAM recommended for the local backend.

## 2. Layout

```
dictate-mac/
├── pyproject.toml          # uv-managed project + runtime deps
├── setup.py                # py2app build for DictateMac.app
├── build.sh                # runs py2app, regenerates the .icns
├── AGENTS.md / README.md   # this pair — current state only
├── assets/
│   ├── DictateMac.icns     # generated icon
│   ├── icon/make_icon.py   # pure-stdlib PNG→icns encoder
│   └── _bundling/          # stubs py2app copies into DictateMac.app
│       ├── silero_vad/     # numpy+onnxruntime re-implementation
│       │   ├── __init__.py # load_silero_vad, get_speech_timestamps, …
│       │   ├── model.py
│       │   └── utils_vad.py
│       ├── mlx_whisper/
│       │   └── timing.py   # raises — we never call add_word_timestamps
│       └── torchaudio/     # Python stub used by the silero-vad path
├── src/dictate_mac/
│   ├── __init__.py         # __version__ (currently 0.4.0)
│   ├── __main__.py         # python -m dictate_mac
│   ├── cli.py              # argparse + daemon / warmup / selftest / menubar
│   ├── audio.py            # Recorder + silero-vad trim_silence
│   ├── transcriber.py      # ASR backends: mlx-whisper local + OpenAI-compatible API
│   ├── typer.py            # CGEvent Unicode injector (+ osascript fallback)
│   ├── hotkey.py           # Quartz CGEventTap on Right Option
│   ├── state.py            # asyncio state machine (DictationMachine)
│   ├── config.py           # persisted settings (XDG-style JSON, schema v2)
│   ├── menubar.py          # rumps NSStatusItem + Model/language submenus
│   ├── model_settings_dialog.py  # free-floating NSWindow for API credentials
│   ├── logutils.py         # .app vs CLI log routing
│   └── selftest.py         # headless smoke checks
├── tests/
│   ├── inject_option.py    # synthetic Right Option injector
│   └── dialog_smoke.py     # API dialog standalone smoke test
└── dist/DictateMac.app     # build artifact (~284 MB)
```

## 3. Module map

Each module below owns exactly the thing its name suggests. Don't
reach across module boundaries — use the public surface listed.

| Module                    | Owns                                                          | Public surface                                                        |
| ------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------- |
| `cli.py`                  | argparse, subcommand dispatch, app-bundle detection           | `main`, `cmd_daemon`, `cmd_warmup`, `cmd_selftest`, `cmd_menubar`      |
| `audio.py`                | mic capture + VAD trimming                                    | `Recorder`, `AudioConfig`, `trim_silence`, `has_speech`                |
| `transcriber.py`          | ASR backends: local mlx-whisper + OpenAI-compatible API       | `transcribe(audio, language, *, model_kind, api_*)`, `check_api_model_available`, `ensure_warm_async`, `warm`, `is_model_cached`, `model_loaded`, `_audio_to_wav_bytes` |
| `typer.py`                | keystroke injection into focused window                       | `type_text(text, backend, per_char_delay_ms)`, `type_text_quartz`, `type_text_osascript` |
| `hotkey.py`               | global Right Option watcher (+ Esc cancel)                    | `HotkeyWatcher(output_queue)`, `HotkeyEdge`, `HotkeyEvent`              |
| `state.py`                | orchestrates the loop: warm → arm → record → transcribe → type | `DictationMachine`, `Settings`, `State`, `SOUND_START`, `SOUND_END`     |
| `config.py`               | persisted settings (language + ASR backend + API credentials) | `AUTO`, `SUPPORTED_ISO_639_1`, `MODEL_KINDS`, `MODEL_KIND_LOCAL`, `MODEL_KIND_API`, `menu_items`, `display_name`, `load`, `save`, `PersistedSettings`, `normalize_endpoint`, `endpoint_scheme_ok`, `detect_system_primary_language` |
| `menubar.py`              | NSStatusItem + Model/language submenus                        | `run_menubar(settings)`, `MenubarApp`                                   |
| `model_settings_dialog.py`| API credentials modal — 3 fields, GET /models check on OK     | `ApiModelSettingsDialog`, `ApiModelSettingsResult`                      |
| `logutils.py`             | stderr vs `~/Library/Logs/dictate-mac/dictate-mac.log`        | `configure_logging`, `is_app_bundle`, `LOG_FILE`, `LOG_FORMAT`          |
| `selftest.py`             | headless smoke tests                                          | `run_all`, `cmd_selftest`, individual `test_*` functions               |

## 4. Architecture and threading

```
┌────────── dictate-mac daemon (Python 3.13) ──────────────────────┐
│                                                                   │
│   ┌────────────────────┐    ┌──────────────────┐                  │
│   │ Menu bar UI        │    │ Hotkey watcher   │                  │
│   │ rumps NSStatusItem │    │ Quartz CGEvent   │                  │
│   │ waveform + Status  │    │ tap (Right Opt)  │                  │
│   └─────────┬──────────┘    └────────┬─────────┘                  │
│             │ state (0.5 s poll)     │ events                     │
│             ▼                        ▼                            │
│            DictationMachine (asyncio worker thread)               │
│            recorder · silero-vad · mlx-whisper · typer            │
│                                                                   │
│                          │ text                                  │
│                          ▼                                       │
│       Quartz CGEvent Unicode keystroke typer                      │
│       (CGEventKeyboardSetUnicodeString)                           │
└───────────────────────────────────────────────────────────────────┘
```

Three threads at runtime:

| Thread        | Owner                              | Notes                                  |
| ------------- | ---------------------------------- | -------------------------------------- |
| Main          | `rumps.App`                        | Status item, menu, 0.5 s refresh timer |
| Worker        | `DictationMachine.run`             | Asyncio loop, state transitions        |
| CFRunLoop     | `HotkeyWatcher`                    | `CGEventTap`, Right Option             |

The Main thread reads `DictationMachine.state` (a `threading.Lock`-guarded
snapshot); the Worker writes it. The CFRunLoop thread pushes events to a
bounded `queue.Queue` (maxsize=32); the Worker drains it every 10 ms.

State transitions:

```
STARTING → DOWNLOADING_MODEL (first run only)
         → LOADING_MODEL
         → READY  (hotkey armed)
                ⇄ RECORDING → TRANSCRIBING → TYPING → READY
                RECORDING --Esc--> READY (buffer discarded, no ASR)
ERROR — terminal; hotkey disarmed
```

System sounds (best-effort via detached `afplay` threads):
`Ping.aiff` on start, `Pop.aiff` immediately after the recorder stops
(before VAD/ASR, so the second Option press gets instant feedback).

## 5. Key technical decisions

These are the design choices that shape the code. Touch them only if
the reason is still valid after you check the source.

- **mlx-whisper large-v3-turbo, MLX variant.** `mlx-community/whisper-
  large-v3-turbo` is the MLX re-serialisation of `openai/whisper-large-
  v3-turbo` — same weights, different runtime. ~1.5 GB in RAM. Stays
  resident until process exit.
- **OpenAI-compatible API backend as an alternative to the local
  model.** When `model_kind=api`, the same trimmed audio buffer is
  encoded as 16 kHz mono PCM WAV (`_audio_to_wav_bytes`) and POSTed to
  `{endpoint}/audio/transcriptions` with `Authorization: Bearer <key>`
  and `model=<model_id>` form fields. The same numpy/onnxruntime
  VAD trim runs upstream, so the gateway only ever receives speech.
  Audio never leaves the machine in local mode.
- **API credentials are validated by GET-ing `/models` on OK.** The
  dialog uses the same bearer token it is about to save and confirms
  the configured model id appears in the response's `data` array.
  401 / 403 → "Authentication failed"; 404 → "Models endpoint not
  found — confirm the URL ends with /v1"; missing id → "Model ID
  '…' not found". Validation runs synchronously on the main thread
  with a 5-second timeout; the OK button is disabled (`setEnabled_(False)`)
  during the check and re-enabled on failure. The button text never
  changes — only its enabled state does, so the user always sees
  the same label regardless of validation state. Cancel stays usable
  throughout.
- **Two fields stacked at the same coordinates for the API key eye
  toggle.** Showing / hiding the key is implemented by stacking an
  `NSSecureTextField` and a plain `NSTextField` on top of each other
  and toggling `setHidden_` rather than by swapping the cell type.
  Earlier attempts to swap the cell between `NSSecureTextFieldCell`
  and `NSTextFieldCell` — the documented Cocoa pattern — failed on
  this PyObjC version because the secure field editor kept painting
  bullets on top of the new cell even after `abortEditing`. The
  two-field approach dodges that class of bug entirely: each cell
  type uses its own native echo behaviour. A Python attribute
  (`_key_value`) holds the plaintext; an `NSControl` delegate on
  both fields keeps them in sync via `controlTextDidChange:`. The
  show/hide eye button toggles visibility and re-copies `_key_value`
  into the now-visible field.
- **Model switching requires an app restart.** Selecting **Local** in
  the Model menu (when currently on API) or vice-versa writes the
  new `model_kind` to the config file and immediately calls
  `MenubarApp._restart_app` — the same `osascript` helper the
  **Restart** menu item uses. The user is dropped back into the
  freshly-launched bundle with the new backend active. No restart is
  needed when only the language or API credentials change.
- **API mode skips the local-model load entirely.** When
  `model_kind="api"`, `_warmup` short-circuits: no
  `is_model_cached()` check, no `snapshot_download`, no
  `mlx_whisper.load_models`. ~1.5 GB of RAM is never allocated. The
  hotkey watcher still arms normally so the daemon stays
  responsive.
- **Hidden Edit menu for ⌘C / ⌘V in modal text fields.** A menubar
  app is `LSUIElement=True` and has no visible menu bar, so without
  intervention `⌘V` in a modal text field triggers the system
  error beep — there's no menu item to absorb the key equivalent
  and route it through the responder chain. The dialog installs a
  hidden Edit menu (`Cut`, `Copy`, `Paste`, `Delete`, `Select All`)
  on `NSApp.mainMenu()` on first show. The menu is never displayed
  but its items register the standard `cut:` / `copy:` / `paste:` /
  `delete:` / `selectAll:` selectors, which routes the key
  equivalents through to the first responder (the field editor).
  Idempotent on every dialog open.
- **Quartz CGEvent Unicode over `pynput`.** Direct call to
  `CGEventKeyboardSetUnicodeString` is the canonical macOS path for
  Unicode typing and forwards predictably through the Citrix ICA
  channel. `pynput` is reported to drop Cyrillic in some
  configurations.
- **silero-vad.** Strips leading/trailing silence before Whisper to
  avoid hallucinated text and to honour "type nothing if no speech".
  In the bundled `.app` the runtime path is a numpy+onnxruntime stub
  under `assets/_bundling/silero_vad/`.
- **PortAudio re-init on stale device snapshot.** PortAudio snapshots
  the audio device list at `Pa_Initialize` (once per process). When
  the topology changes under a long-running daemon — virtual devices
  appearing/disappearing (NoMachine-style drivers), a coreaudiod
  restart, a default-input switch — `Pa_OpenStream` on the stale
  default device fails with `paInternalError (-9986)` on every
  recording until the process restarts. `Recorder.start()` catches
  `sd.PortAudioError` on the first open, calls `sd._terminate()` +
  `sd._initialize()` to refresh the snapshot, and retries once. Only
  a second failure propagates to the state machine.
- **Esc cancels an active recording.** The hotkey tap also forwards
  plain `kCGEventKeyDown` Esc presses (keycode 0x35, no Cmd/Ctrl/
  Shift) into the event queue. The state machine honours them only
  in RECORDING: `Recorder.stop()` closes the stream, the buffer is
  discarded (no VAD/ASR/typing), the end sound plays, and the
  machine returns to READY. Esc in any other state is dropped.
- **Right Option only, with modifier filter.** Rarely used as a system
  shortcut. Presses that also hold Cmd/Ctrl/Shift are ignored so the
  daemon doesn't break system shortcuts. A global macro tool (e.g.
  Keyboard Maestro) bound to Right Option can intercept events before
  our tap sees them — symptom: no `tap flagsChg keycode=0x3d` lines
  in the DEBUG log.
- **`kCGEventFlagsChanged` in the event mask.** Modifier keys (Option
  / Shift / Ctrl / Cmd) fire as flag transitions, not as
  `kCGEventKeyDown`. Listening to keyDown+keyUp alone makes the daemon
  blind to a lone Option press. The handler tracks `prev_flags` to
  synthesise press/release events for the queue.
- **CGEventTap callback as plain Python.** PyObjC bridges a Python
  function to the C trampoline. Wrapping the callback in
  `ctypes.CFUNCTYPE(c_void_p, ...)` fails because the real callback
  receives Objective-C objects that won't marshal into raw void
  pointers. The signature must be `(proxy, type, event, userInfo)`
  with four arguments — omit the fourth and `TypeError` kills the
  callback while the CFRunLoop keeps spinning.
- **Persisted settings in JSON, XDG-style.** Six storage options were
  considered (NSUserDefaults, Library/Application Support, ~/.config,
  TOML, env var, hybrid); `~/.config/dictate-mac/config.json` was
  chosen because NSUserDefaults would write to two different plists
  depending on whether the daemon runs from the source venv or from
  the `.app` bundle. Atomic write via `os.replace`; `0o700` on the
  directory and `0o600` on the file. CLI subcommands never touch the
  file — only the menu bar reads/writes it. The menu bar source is
  `Foundation.NSLocale.preferredLanguages()` (PyObjC, ~1 ms,
  in-process).
- **Hot-apply language and model kind.** `mlx_whisper.transcribe`
  reads `language` per call. Switching between two languages from the
  menu updates the in-memory `Settings.language`, persists the
  choice, and the next recording uses the new value. Only the
  encoder's first-pass language-token prediction pays the one-time
  ~0.3 s switch cost; the model itself is not reloaded. The
  `model_kind` switch is similarly hot: the next recording after a
  menu switch uses the new backend with no warmup delay — the API
  path has no in-process model to load, the local path's model
  stays in RAM regardless of which mode is selected.
- **Rumps menu bar + py2app `.app`.** `rumps` is a PyObjC wrapper
  around `NSStatusItem` — small, idiomatic, runs `NSApp` on the main
  thread. `py2app` produces a real `.app` bundle with
  `CFBundleIdentifier=com.local.dictate-mac`, `LSUIElement=True`,
  and `NSMicrophoneUsageDescription`, so TCC grants (Microphone,
  Accessibility, Input Monitoring) attach to the bundle id rather
  than to Terminal.
- **API key never logged, never serialised into error messages.**
  `transcriber.check_api_model_available`, `_transcribe_api`, the
  modal OK handler, and `state._stop_and_process` keep the key out of
  every `RuntimeError` they raise and out of every `logger.warning`
  call. The dialog's error label shows only the category of failure
  and the endpoint — never the key. Selftest mock-fakes the API
  key as `SECRET_KEY_DO_NOT_LEAK` and explicitly asserts the
  substring never appears in any captured error.

## 6. Build pipeline (`./build.sh`)

1. Ensure `py2app` is in the active venv (install via `uv pip` if not).
2. Regenerate `assets/DictateMac.icns` from `assets/icon/make_icon.py`
   (pure-stdlib PNG encoder + `iconutil`).
3. Pre-patch the source venv (`_pre_patch_source_venv` in
   `setup.py`): swap the wheel's `mlx_whisper/timing.py`,
   `silero_vad/__init__.py`, `silero_vad/utils_vad.py`,
   `silero_vad/model.py` with the stubs under `assets/_bundling/`.
   This stops py2app's `modulegraph` from chasing scipy/numba/torch
   through upstream `from scipy import signal` /
   `import torch` at the top of those modules.
4. Run `setup.py py2app`.
5. Restore the source venv in a `finally:` block (always, even on
   build failure).
6. Run post-build steps on `dist/DictateMac.app`:
   - `_extract_native_runtime_libs` — rewrite `python313.zip` so
     `dlopen` can find `libportaudio.dylib` and `_sounddevice_data`.
   - `_install_torchaudio_stub` — wipe bundled `torchaudio/` and
     write a Python stub (no `libtorchaudio.abi3.so`). The real
     torchaudio wheel's `libtorchaudio` has an `install_name` chain
     that references unbundled wheel deps, so it can't `dlopen`
     inside the bundle regardless of its LC_RPATH — the stub is the
     only thing that works.
   - `_install_silero_vad_stub` — replace upstream silero-vad
     package with our numpy+onnxruntime reimplementation.
   - `_install_timing_stub` — replace `mlx_whisper.timing` with a
     stub that raises on `add_word_timestamps`.
   - `_strip_bundle_junk` — drop `torch/`, `numba/`, `llvmlite/`,
     `sympy/`, `networkx/`, `scipy/optimize/` and the rest of
     `scipy/` not reachable at runtime, `*.dSYM/`, stdlib `test/`,
     `numpy/tests/`, `numpy/_core/tests/`, `numpy/f2py/`,
     `numpy/typing/`, `numpy/_examples`, `numpy/include/`,
     `numpy/{strings,char,core,rec,ctypeslib,dtypes,exceptions,
     ma,polynomial}`, `mlx/include/`, all `*.pyi` stubs,
     `huggingface_hub/{cli,inference,serialization,hub_mixin}`,
     three unused `mlx_whisper` modules, `silero_vad.jit` and
     the four alt-opset `.onnx` variants, the top-level and
     `data/silero_vad/` mirrors of `silero_vad.onnx`,
     `Contents/Resources/include/` and `openssl.ca/`,
     duplicate `mlx.metallib`/`libmlx.dylib`/`libjaccl.dylib`.
    - `_rewrite_boot_script` — replace the build-time venv path in
      `__boot__.py` with `os.environ['RESOURCEPATH']` so the bundle
      uses its embedded Python framework.
     - `_strip_info_plist_paths` — rewrite
       `Info.plist`'s `PythonInfoDict.PythonExecutable` from the
       developer's build-time venv path to
       `@executable_path/../Frameworks/Python.framework/Versions/3.13/Python`
       so the developer's username doesn't leak into the shipped
       bundle's Info.plist. **Also re-signs the bundle with
       `codesign --force --deep --sign -`** because editing
       `Info.plist` after py2app sealed it breaks the embedded
       code-signature's `Info.plist` hash, and an unsigned-mismatch
       bundle gets silently denied TCC permission prompts at runtime
       (the Microphone dialog never appears and the recorder logs
       `peak=0.0000`).
    - `_strip_bundle_junk` also drops every `__pycache__/`
      directory under `Contents/Resources/` (both `lib/python3.13/`
      and the top-level mirror paths py2app creates). The `.pyc`
      headers embed the *build machine's* absolute source path —
      `/Users/<name>/.venv/...` — so leaving them in the bundle
      leaks the developer's home directory. Python recompiles on
      first import with the bundle's own paths, so dropping the
      caches is free at runtime.

Bundle id: `com.local.dictate-mac`. No code signing, no notarisation.
Final size: ~284 MB on disk.

## 7. Dependencies (`pyproject.toml`)

Python 3.13.x on Apple Silicon.

Runtime:
`mlx>=0.32`, `mlx-whisper`, `huggingface_hub>=0.20`,
`sounddevice>=0.4.6`, `silero-vad>=5.1`, `onnxruntime>=1.17`,
`pyobjc-framework-Quartz>=10.2`, `numpy>=1.26`, `rumps>=0.4.0`,
`requests>=2.31`.

Dev (build only): `py2app>=0.28`.

Installed snapshot: `mlx 0.32.0`, `mlx-whisper 0.4.3`,
`sounddevice 0.5.5`, `silero-vad 6.2.1`, `onnxruntime 1.27.0`,
`pyobjc-framework-Quartz 12.2.1`, `numpy 2.4.6`,
`huggingface-hub 1.23.0`, `rumps 0.4.0`, `requests 2.34.2`,
`py2app 0.28.10`.

## 8. CLI surface

`dictate-mac` (no subcommand) launches the menu bar app.

| Subcommand | Purpose                                                |
| ---------- | ------------------------------------------------------ |
| `daemon`   | Plain CLI daemon (no menu bar). Same code path.        |
| `warmup`   | Download + load model, optional mic sanity test.       |
| `selftest` | Headless smoke checks; `--no-mic` skips the mic step.  |

Common flags (must come before OR after the subcommand): `--quiet`,
`--log-level` (DEBUG|INFO|WARNING|ERROR|CRITICAL), `--output`
(quartz|osascript), `--language` (ISO-639-1 code or `auto`),
`--model-kind` (`local` or `api`), `--api-endpoint`, `--api-key`,
`--model-id`. The last four are meaningful only with
`--model-kind=api`; `cmd_daemon` refuses to start without the
three required values when `api` is selected. CLI runs do not read
or write the persisted config file — language and ASR settings are
taken from flags directly. The menu bar entry point sources its
settings from `~/.config/dictate-mac/config.json` and ignores all
of these flags.

## 9. Persisted config

Path: `$XDG_CONFIG_HOME/dictate-mac/config.json`
(default `~/.config/dictate-mac/config.json`).

Schema v2 contents:

```json
{
  "_v": 2,
  "language": "<iso-639-1 or 'auto'>",
  "model_kind": "local|api",
  "api_endpoint": "<openai-compatible base url>",
  "api_key": "<bearer token>",
  "api_model_id": "<model id the gateway should use>"
}
```

A v1 file (no `model_kind` / API fields) loads with `model_kind=
"local"` and empty API fields — the persisted `language` is
preserved untouched, and subsequent saves rewrite the file as v2.
`api_endpoint` is canonicalised at save time: surrounding whitespace
and a single trailing `/` are stripped via `normalize_endpoint`,
so storage and request building always operate on the same form.

Atomic write via `os.replace` from a sibling `.tmp`. Permissions:
`0o700` on the directory, `0o600` on the file. Corrupt files are
left untouched (the daemon logs the cause, falls back to a fresh
in-memory default, and waits for the user to repair by hand).
First-run detection is `Foundation.NSLocale.preferredLanguages()`;
no supported match maps to `"auto"`.

## 10. Testing

`dictate-mac selftest [--no-mic]` runs fourteen checks; optionally a
fifteenth mic roundtrip. Exits 0 if all PASS, 1 on any FAIL.

1. **model-load** — `mlx_whisper` weights load into RAM.
2. **vad-silence** — 1 s of zeros → VAD returns `[]`.
3. **vad-speech-like** — AM-modulated noise → VAD returns something or
   `[]` (silero-vad is tuned for real speech; both are fine).
4. **asr-smoke** — `transcribe()` returns a string of any length.
5. **typer-dispatch** — the `type_text` router picks `quartz` / `osascript`
   / default correctly **without** injecting real keystrokes.
6. **config-v1-migration** — a v1 `config.json` loads as
   `model_kind="local"` with empty API fields and the persisted
   language preserved.
7. **config-invalid-endpoint** — malformed endpoints (`ftp://…`,
   bare host, etc.) are rejected by `PersistedSettings.is_valid`.
8. **config-api-required-when-api** — `model_kind="local"` accepts
   empty API fields; `model_kind="api"` rejects partial ones.
9. **audio-wav-roundtrip** — numpy → 16-bit PCM WAV → numpy, with
   amplitude preserved to within one LSB.
10. **api-transcribe-headers** — mocked `POST /v1/audio/transcriptions`
    confirms the URL, multipart `model` field, `Authorization:
    Bearer` header and (when configured) `language` field are all
    wired correctly.
11. **api-transcribe-auto-language** — when `language="auto"`, the
    `language` field is omitted from the form so the gateway falls
    back to its own detection.
12. **api-models-check** — mocked `GET /v1/models` handles 200+id,
    200-missing-id, 401, 404 and 500 correctly, and never leaks the
    fake API key string into any error.
13. **recorder-portaudio-retry** — a mocked first `sd.InputStream`
    open raising `PortAudioError -9986` triggers one
    `sd._terminate()`/`sd._initialize()` cycle and a successful retry.
14. **hotkey-escape-event** — a synthetic Esc keyDown is queued as a
    cancel press; Cmd+Esc is filtered out.
15. **mic-roundtrip** *(unless `--no-mic`)* — record 1.5 s, run VAD + ASR,
    report durations.

`dictate-mac warmup --skip-mic-test` is the deterministic "is it
healthy" smoke for CI; ~3 s from cache.

`tests/inject_option.py` injects a synthetic Right Option pair via
`CGEventCreateKeyboardEvent` + `CGEventPost`. Run it AFTER
DictateMac.app has reached `[hotkey] ready` to walk the full
Recording → VAD → ASR → Typing → Ready cycle.

`tests/dialog_smoke.py` builds the API settings dialog standalone
(outside the menubar / state-machine plumbing) and exercises the
eye-toggle cell-swap, the visibility chain, and the action
selectors. Smoke-tested without entering the modal runloop (which
would block forever in a headless context).

## 11. Working rules

- Never `git commit` without an explicit user request.
- Never delete files without an explicit user confirmation.
- Don't reach for `pynput` — use raw Quartz `CGEvent`.
- Don't use the clipboard — Citrix blocks it.
- Don't add a `README.md` "What changed" or "Changelog" section.
  Don't add a "Phase N" marker or a numbered history to
  `AGENTS.md`. Either document the **current** reality or strip
  the section.
- All prose in `AGENTS.md` and `README.md` is English. No
  Russian, German, Spanish, or other scripts in prose, comments,
  or quoted test utterances.
- Reference code with `file_path:line_number` when pointing an
  agent at it.
- API credentials (endpoints, bearer tokens, model ids) **must
  never appear as real values** in source, comments, AGENTS.md,
  README.md, or test fixtures — even in examples and even when
  pulled from a public tool you happen to know. Use generic
  placeholders (`<endpoint>`, `<api-key>`, `<model-id>`,
  `https://<host>/v1`, etc.) instead. Selftest fakes the key as
  the explicit string `SECRET_KEY_DO_NOT_LEAK` and asserts it
  never surfaces in any error message.

## 12. Known limitations

- ~1.5 GB of RAM held permanently after first warmup **in local
  mode**. In API mode the warmup thread is skipped entirely and
  the local model is never loaded; switching from local to API via
  the menu triggers a restart that drops the previously-loaded
  weights.
- The bundled `.app` cannot transcode audio files: the
  `silero_vad.read_audio` and `torchaudio.load` paths raise
  `RuntimeError`. All production audio flows through
  PortAudio → `numpy.ndarray` → silero-vad → Whisper.
- Citrix Viewer needs **Send Unicode keyboard input** enabled
  (modern default). If Cyrillic drops, switch to
  `--output=osascript`.
- TCC permissions (Microphone, Accessibility, Input Monitoring)
  must be granted manually. Bundle id: `com.local.dictate-mac`.
  They survive re-installs of the same `.app`, but moving to
  another Mac or another bundle id requires a fresh grant.
- The `.app` is unsigned, ad-hoc, not notarised. Gatekeeper may
  prompt on first launch — right-click → Open.
- If Input Monitoring was not granted before launch, the
  CGEventTap stays disabled. Re-enable requires Quit + reopen.
- No code signing, no LaunchAgent, no auto-start at login.
- Switching the ASR backend (Local ↔ API) requires an app
  restart. The menu click triggers the restart automatically; the
  user is dropped back into the freshly-launched bundle with the
  new config active.
- Python 3.13.x only; no 3.14 wheel for `mlx`.
