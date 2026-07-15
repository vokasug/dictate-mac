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

`dictate-mac` is a macOS-only, fully-local voice dictation daemon:

- The user presses **Right Option** to start recording and presses it
  again to stop.
- Audio is captured at 16 kHz mono, trimmed with `silero-vad`, and
  transcribed with `mlx-whisper`
  (`mlx-community/whisper-large-v3-turbo`).
- Recognized text is typed character-by-character into the currently
  focused window via `Quartz.CGEventKeyboardSetUnicodeString` — no
  clipboard paste, so it works inside Citrix Workspace sessions that
  do not forward the macOS clipboard.
- The model downloads on the first run and stays in RAM until the
  process exits.

Default entry point is `dictate-mac` (no subcommand) — a `rumps` menu
bar app. `dictate-mac daemon` is a CLI-only mode for SSH/CI/tmux.

Python 3.13 only (no 3.14 wheel for `mlx`); Apple Silicon only; ≥ 8 GB
RAM recommended.

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
│   ├── __init__.py         # __version__ (currently 0.2.0)
│   ├── __main__.py         # python -m dictate_mac
│   ├── cli.py              # argparse + daemon / warmup / selftest / menubar
│   ├── audio.py            # Recorder + silero-vad trim_silence
│   ├── transcriber.py      # mlx-whisper lazy singleton + warmup
│   ├── typer.py            # CGEvent Unicode injector (+ osascript fallback)
│   ├── hotkey.py           # Quartz CGEventTap on Right Option
│   ├── state.py            # asyncio state machine (DictationMachine)
│   ├── config.py           # persisted language (XDG-style JSON)
│   ├── menubar.py          # rumps NSStatusItem, language submenu
│   ├── logutils.py         # .app vs CLI log routing
│   └── selftest.py         # headless smoke checks
├── tests/
│   └── inject_option.py    # synthetic Right Option injector
└── dist/DictateMac.app     # build artifact (~291 MB)
```

## 3. Module map

Each module below owns exactly the thing its name suggests. Don't
reach across module boundaries — use the public surface listed.

| Module              | Owns                                                          | Public surface                                                        |
| ------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------- |
| `cli.py`            | argparse, subcommand dispatch, app-bundle detection           | `main`, `cmd_daemon`, `cmd_warmup`, `cmd_selftest`, `cmd_menubar`      |
| `audio.py`          | mic capture + VAD trimming                                    | `Recorder`, `AudioConfig`, `trim_silence`, `has_speech`                |
| `transcriber.py`    | mlx-whisper model lifecycle                                   | `transcribe(audio, language)`, `ensure_warm_async`, `warm`, `is_model_cached`, `model_loaded` |
| `typer.py`          | keystroke injection into focused window                        | `type_text(text, backend, per_char_delay_ms)`, `type_text_quartz`, `type_text_osascript` |
| `hotkey.py`         | global Right Option watcher                                   | `HotkeyWatcher(output_queue)`, `HotkeyEdge`, `HotkeyEvent`              |
| `state.py`          | orchestrates the loop: warm → arm → record → transcribe → type | `DictationMachine`, `Settings`, `State`, `SOUND_START`, `SOUND_END`     |
| `config.py`         | persisted recognition language                                | `AUTO`, `SUPPORTED_ISO_639_1`, `menu_items`, `display_name`, `load`, `save`, `PersistedSettings`, `detect_system_primary_language` |
| `menubar.py`        | NSStatusItem UI                                               | `run_menubar(settings)`, `MenubarApp`                                   |
| `logutils.py`       | stderr vs `~/Library/Logs/dictate-mac/dictate-mac.log`        | `configure_logging`, `is_app_bundle`, `LOG_FILE`, `LOG_FORMAT`          |
| `selftest.py`       | headless smoke tests                                          | `run_all`, `cmd_selftest`, individual `test_*` functions               |

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
- **Quartz CGEvent Unicode over `pynput`.** Direct call to
  `CGEventKeyboardSetUnicodeString` is the canonical macOS path for
  Unicode typing and forwards predictably through the Citrix ICA
  channel. `pynput` is reported to drop Cyrillic in some
  configurations.
- **silero-vad.** Strips leading/trailing silence before Whisper to
  avoid hallucinated text and to honour "type nothing if no speech".
  In the bundled `.app` the runtime path is a numpy+onnxruntime stub
  under `assets/_bundling/silero_vad/`.
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
- **Persisted language in JSON, XDG-style.** Six storage options were
  considered (NSUserDefaults, Library/Application Support, ~/.config,
  TOML, env var, hybrid); `~/.config/dictate-mac/config.json` was
  chosen because NSUserDefaults would write to two different plists
  depending on whether the daemon runs from the source venv or from
  the `.app` bundle. Atomic write via `os.replace`; `0o700` on the
  directory and `0o600` on the file. CLI subcommands never touch the
  file — only the menu bar reads/writes it. The menu bar source is
  `Foundation.NSLocale.preferredLanguages()` (PyObjC, ~1 ms,
  in-process).
- **Hot-apply language.** `mlx_whisper.transcribe` reads `language`
  per call. Switching between two languages from the menu updates the
  in-memory `Settings.language`, persists the choice, and the next
  recording uses the new value. Only the encoder's first-pass
  language-token prediction pays the one-time ~0.3 s switch cost;
  the model itself is not reloaded.
- **Rumps menu bar + py2app `.app`.** `rumps` is a PyObjC wrapper
  around `NSStatusItem` — small, idiomatic, runs `NSApp` on the main
  thread. `py2app` produces a real `.app` bundle with
  `CFBundleIdentifier=com.local.dictate-mac`, `LSUIElement=True`,
  and `NSMicrophoneUsageDescription`, so TCC grants (Microphone,
  Accessibility, Input Monitoring) attach to the bundle id rather
  than to Terminal.

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
   - `_fixup_native_library_rpaths` — rewrite torchaudio's broken
     LC_RPATH with `install_name_tool -rpath` (the upstream wheel
     pins a CI build path).
   - `_install_torchaudio_stub` — wipe bundled `torchaudio/` and
     write a Python stub (no `libtorchaudio.abi3.so`).
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
      (mic dialog never appears, `peak=0.0000` in the log). This
      bit a 0.2.0 release candidate that hadn't been re-signed —
      see the conversation in AGENTS.md history if you need context.
    - `_strip_bundle_junk` also drops every `__pycache__/`
      directory under `Contents/Resources/` (both `lib/python3.13/`
      and the top-level mirror paths py2app creates). The `.pyc`
      headers embed the *build machine's* absolute source path —
      `/Users/<name>/.venv/...` — so leaving them in the bundle
      leaks the developer's home directory. Python recompiles on
      first import with the bundle's own paths, so dropping the
      caches is free at runtime.

Bundle id: `com.local.dictate-mac`. No code signing, no notarisation.
Final size: ~291 MB on disk.

## 7. Dependencies (`pyproject.toml`)

Python 3.13.x on Apple Silicon.

Runtime:
`mlx>=0.32`, `mlx-whisper`, `huggingface_hub>=0.20`,
`sounddevice>=0.4.6`, `silero-vad>=5.1`, `onnxruntime>=1.17`,
`pyobjc-framework-Quartz>=10.2`, `numpy>=1.26`, `rumps>=0.4.0`.

Dev (build only): `py2app>=0.28`.

Installed snapshot: `mlx 0.32.0`, `mlx-whisper 0.4.3`,
`sounddevice 0.5.5`, `silero-vad 6.2.1`, `onnxruntime 1.27.0`,
`pyobjc-framework-Quartz 12.2.1`, `numpy 2.4.6`,
`huggingface-hub 1.23.0`, `rumps 0.4.0`, `py2app 0.28.10`.

## 8. CLI surface

`dictate-mac` (no subcommand) launches the menu bar app.

| Subcommand | Purpose                                                |
| ---------- | ------------------------------------------------------ |
| `daemon`   | Plain CLI daemon (no menu bar). Same code path.        |
| `warmup`   | Download + load model, optional mic sanity test.       |
| `selftest` | Headless smoke checks; `--no-mic` skips the mic step.  |

Common flags (must come before OR after the subcommand): `--quiet`,
`--log-level` (DEBUG|INFO|WARNING|ERROR|CRITICAL), `--output`
(quartz|osascript), `--language` (ISO-639-1 code or `auto`). CLI
runs do not read or write the persisted config file — language is
taken from `--language` directly. The menu bar entry point sources
its language from `~/.config/dictate-mac/config.json` and ignores
`--language`.

## 9. Persisted config

Path: `$XDG_CONFIG_HOME/dictate-mac/config.json`
(default `~/.config/dictate-mac/config.json`).

Contents: `{"_v": 1, "language": "<iso-639-1 or 'auto'>"}`.

Atomic write via `os.replace` from a sibling `.tmp`. Permissions:
`0o700` on the directory, `0o600` on the file. Corrupt files are
left untouched (the daemon logs the cause, falls back to a fresh
in-memory default, and waits for the user to repair by hand).
First-run detection is `Foundation.NSLocale.preferredLanguages()`;
no supported match maps to `"auto"`.

## 10. Testing

- `dictate-mac selftest [--no-mic]` runs model-load, VAD
  silence/speech, ASR smoke, typer-dispatch router, optional mic
  roundtrip. Exits 0 if all PASS, 1 on any FAIL.
- `dictate-mac warmup --skip-mic-test` is the deterministic "is it
  healthy" smoke for CI; ~3 s from cache.
- `tests/inject_option.py` injects a synthetic Right Option pair
  via `CGEventCreateKeyboardEvent` + `CGEventPost`. Run it AFTER
  DictateMac.app has reached `[hotkey] ready` to walk the full
  Recording → VAD → ASR → Typing → Ready cycle.

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

## 12. Known limitations

- ~1.5 GB of RAM held permanently after first warmup.
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
- Python 3.13.x only; no 3.14 wheel for `mlx`.
