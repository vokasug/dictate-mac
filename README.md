# dictate-mac 🎙️

Local voice dictation for macOS. Press **Right Option** → speak → press
**Right Option** again. Recognized text is typed character-by-character
into the focused window — including Citrix Workspace sessions that don't
share the clipboard.

- Model: [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo)
  via its MLX build for Apple Silicon, **or** any
  OpenAI-compatible `/v1/audio/transcriptions` endpoint (your own
  gateway, a hosted service, …)
- Language: 100 ISO-639-1 codes Whisper supports, plus `Auto-detect`
- Local backend downloads on the first run and stays in RAM until you
  quit the app; API backend uploads only the recorded speech (after
  silence trimming) and holds nothing extra in memory
- Runs as a **menu bar app** with a one-line status and a Quit item —
  the default invocation `dictate-mac` launches it
- Optional: still runs as a CLI daemon (`dictate-mac daemon`) for SSH/CI

---

## Contents

- [Requirements](#requirements)
- [Install](#install)  — recommended
- [Run from source](#run-from-source)  — alternative
- [Build the `.app` from source](#build-the-app-from-source)  — advanced
- [macOS Permissions](#macos-permissions-privacy--security)
- [First Run](#first-run)
- [Verifying the install](#verifying-the-install)
- [Usage](#usage)
- [The menu bar](#the-menu-bar)
- [Recognition language](#recognition-language)
- [Model: local vs API](#model-local-vs-api)
- [How It Works](#how-it-works)
- [Installing on another Mac](#installing-on-another-mac)
- [Troubleshooting](#troubleshooting)
- [Uninstall](#uninstall)
- [Known Limitations](#known-limitations)
- [Known Limitations](#known-limitations)

---

## Requirements

To *run* dictate-mac:

- **Apple Silicon Mac** (M1 / M2 / M3 / M4)
- **macOS 13 Ventura** or newer
- **8 GB RAM** or more
- Any working microphone (the built-in one is fine)
- ~3 GB of free disk space — ~284 MB for the `.app`, ~1.5 GB for the
  Whisper STT weights downloaded on first launch (local backend only;
  API backend doesn't need anything extra)
- Internet access on the first launch (model download from
  Hugging Face)

To *run from source* you additionally need:

- **Python 3.13** — the simplest install path is
  [`uv`](https://github.com/astral-sh/uv); it manages the project
  venv and the runtime deps in one shot.

To *build* the `.app` from source you additionally need:

- The Xcode Command Line Tools (`xcode-select --install`) — needed
  for the `install_name_tool` calls in `setup.py` that rewrite broken
  LC_RPATH entries baked into PyPI wheels.
- `py2app` is pulled in by `uv pip install` against the `[dev]`
  group in `pyproject.toml`; you do not have to install it separately.

## Install

The recommended path: take the prebuilt `DictateMac.app` from the
GitHub release, copy it into `/Applications`, and open it. The
bundle is fully self-contained — Python 3.13, mlx + mlx_whisper,
the onnxruntime-backed silero-vad stub, rumps, the Metal shim, the
Whisper tokenizer — everything except the ~1.5 GB Whisper STT
weights, which download on the first launch from Hugging Face.

### 1. Download

[`DictateMac-v0.3.0-macos.zip`](https://github.com/vokasug/dictate-mac/releases/latest/download/DictateMac-v0.3.0-macos.zip)
from the [latest release](https://github.com/vokasug/dictate-mac/releases/latest).
Compressed download is ~110–140 MB; the extracted `.app` is ~284 MB.

If you want to verify the download, `DictateMac-v0.3.0-macos.zip.sha256`
ships alongside it on the release page:

```bash
shasum -a 256 DictateMac-v0.3.0-macos.zip
# compare with the contents of the .sha256 file from the same release
```

### 2. Install

Double-click the zip in Finder, or from a terminal:

```bash
open DictateMac-v0.3.0-macos.zip   # expands to ./DictateMac.app
mv DictateMac.app /Applications/   # optional — keeps the .app
                                  # alongside your other apps
open /Applications/DictateMac.app
```

You can also skip `/Applications/` and run the `.app` from any
folder — `/Applications/` just keeps the TCC permissions together
with the rest of your system apps.

> **First-launch Gatekeeper.** The `.app` is unsigned and
> ad-hoc-notarised. If macOS refuses to open it, right-click
> `DictateMac.app` in Finder → **Open** → confirm in the dialog.
> Subsequent launches are double-click as usual.

### 3. Grant the three permissions

macOS prompts for *Microphone* the very first time you press Right
Option; the menu bar stays on `Status: Error: see logs` until you
also grant *Accessibility* and *Input Monitoring* by hand in
**System Settings → Privacy & Security**. See
[macOS Permissions](#macos-permissions-privacy--security) for the
full one-time setup.

### 4. Pick the language

Click the menu bar icon → **Recognition language** → *Auto-detect*
or any of the 100 ISO-639-1 codes. The choice persists in
`~/.config/dictate-mac/config.json`; subsequent switches take effect
on the next recording (no model reload).

### 5. Use

Press **Right Option**, speak, press **Right Option** again. The
recognised text is typed character-by-character into the currently
focused window — including Citrix Workspace sessions that don't
forward the macOS clipboard.

## Run from source

If you want the development loop — instant edits to source with no
rebuild cycle, or a CLI daemon over SSH — install from PyPI-style:

```bash
cd ~/dictate-mac

uv venv --python 3.13 .venv
uv pip install -e .
dictate-mac --help
```

The CLI subcommands are usable straight from the venv:

```bash
dictate-mac            # menu bar app (default)
dictate-mac daemon     # CLI mode, no menu bar — for SSH/CI/tmux
dictate-mac warmup     # download + load the model, then exit
dictate-mac selftest   # headless smoke checks
```

> Installed versions of every runtime dep live in `AGENTS.md` § 7.

The tradeoff vs. the `.app` install: macOS TCC permissions
(Microphone, Accessibility, Input Monitoring) attach to the
**Python interpreter** running the script (or to Terminal/iTerm
when you use `daemon` from a shell), not to a stable bundle id. If
you run from the venv and later switch to the `.app`, grant the
permissions once more for `com.local.dictate-mac`.

## Build the `.app` from source

Most users will not need this section — the [Install](#install)
path above pulls a prebuilt `.app` from the GitHub release. Rebuild
from source only if you want to:

- run a private fork or an unreleased commit,
- ship your own signed/notarised variant,
- inspect what `py2app` actually produced.

```bash
./build.sh
open dist/DictateMac.app
```

`./build.sh` regenerates the `.icns` icon (pure-Python PNG encoder +
`iconutil`), then runs `py2app.build_app`. The result is fully
self-contained — everything except the Whisper STT weights (which
download on the first launch) is bundled:

- mlx + mlx-whisper with `mlx.metallib` (incl. the Apple Metal
  compute library), `libmlx.dylib`, `libjaccl.dylib`
- silero-vad backed by `onnxruntime` against the bundled
  `silero_vad.onnx` (only the ONNX model variant ships; the
  torch-jit / safetensors / alt-opset variants are stripped)
- sounddevice + the bundled PortAudio library
- rumps, NumPy, the onnxruntime inference engine, huggingface_hub
  with its `hf_xet` fast-downloader — `.app` is "double-click and
  forget" on any Apple-Silicon Mac
- com.local.dictate-mac bundle id (no code signing — local use only)

**Bundle size: ~284 MB.** The shrink from a naïve py2app build comes
from the stubs and the post-build strip pass in `setup.py`:

- `silero_vad` is replaced by a numpy + onnxruntime
  re-implementation so the bundle does not have to ship `torch`
  (~429 MB) and `libtorch_cpu.dylib` (~263 MB). Production audio
  goes PortAudio → `numpy.ndarray` → silero-vad → Whisper; the
  in-bundle stub never touches files.
- `torchaudio` is replaced by a tiny Python stub for the same
  reason — its wheel ships a native dylib with broken LC_RPATH.
- `mlx_whisper.timing` is replaced by a stub that raises on
  `add_word_timestamps`; we never call that path, and replacing
  the module before `setup()` runs stops `modulegraph` from
  chasing `scipy.signal` (which would otherwise drag the entire
  scipy tree into the bundle).
- `_strip_bundle_junk` deletes what's left: torch/numba/llvmlite/
  sympy/networkx, scipy/optimize and the rest of scipy that the
  runtime cannot reach, `*.dSYM/`, `numpy/tests/`, `numpy/_core/
  tests/`, `numpy/f2py/`, `numpy/typing/`, `numpy/_examples`,
  `numpy/include/`, `numpy/{strings,char,core,rec,ctypeslib,
  dtypes,exceptions,ma,polynomial}/`, `mlx/include/`, all
  `*.pyi` stubs, `huggingface_hub/{cli,inference,serialization,
  hub_mixin}`, three unused `mlx_whisper` modules,
  `silero_vad.jit` and the alt-opset `.onnx` variants, duplicate
  top-level / `data/silero_vad/` mirrors of `silero_vad.onnx`,
  duplicate `mlx.metallib` / `libmlx.dylib` / `libjaccl.dylib`,
  `Contents/Resources/include/`, every `__pycache__/` directory
  under `lib/python3.13/` and `Contents/Resources/` (Python
  re-compiles on first import with the bundle's own paths), and
  `openssl.ca/`.

After the build finishes, `setup.py`'s post-build hooks also
rewrite two locations so the developer's home directory does not
leak into the shipped bundle:

- `__boot__.py` — swapped to use `os.environ['RESOURCEPATH']`
  instead of the absolute build-time venv path.
- `Info.plist`'s `PythonInfoDict.PythonExecutable` — swapped to
  `@executable_path/../Frameworks/Python.framework/Versions/3.13/Python`.

For the day-to-day CLI loop without rebuilding, see
[Run from source](#run-from-source) above.

## macOS Permissions (Privacy & Security)

macOS requires three permissions **manually**. With the **self-contained
`DictateMac.app`** they are granted to **`com.local.dictate-mac`** —
the bundle id stays put even when you move the app between users or
rebuild. With the **menu bar app run from the source venv** they go to
the **Python interpreter** (`Python` / `Python.app`) running the script.
With the **`daemon` CLI** they are granted to **Terminal.app /
iTerm.app** instead.

| # | Permission             | Where                                              | Why                              |
| - | ---------------------- | -------------------------------------------------- | -------------------------------- |
| 1 | **Microphone**         | Privacy & Security → Microphone                    | Record your voice                |
| 2 | **Accessibility**      | Privacy & Security → Accessibility                 | Synthesize keystrokes            |
| 3 | **Input Monitoring**   | Privacy & Security → Input Monitoring (macOS 14+)  | Intercept the Right Option key   |

> After changing permissions, **quit and reopen** the app and restart
> `dictate-mac`. One-time per-machine grant; persists across
> re-installs.

### Quick diagnostic

Right after start, the menu item status reads `Status: Ready` and
logs end with `[hotkey] ready`. If the status freezes on
`Starting…` or stops on `Error: see logs`, revisit the table above.
CLI mode prints `[hotkey] permission denied` on startup in the same
situation.

### Permission shortcuts in the menu

The menu bar exposes one clickable row per TCC service the app uses.
Clicking the row opens the matching pane of **System Settings →
Privacy & Security** so the user can flip the toggle directly:

```
Status: Ready — press Right Option to start and stop recording
─────────────────
Model (changing will restart app)
✓ Local (mlx-community/whisper-large-v3-turbo)
  API (https://<your-host>/v1)
─────────────────
Recognition language: Auto-detect
─────────────────
Permissions (reset permissions and restart app if not working)
Input Monitoring
Microphone
Accessibility
─────────────────
Open log
About
Restart
Quit
```

After flipping a toggle in System Settings, **Quit + reopen** the
app (or use the `Restart` menu item). macOS does not retroactively
enable a CGEventTap; the tap only becomes functional after a fresh
launch with the toggle on.

## First Run

**Menu bar (recommended):**

```bash
dictate-mac
```

The icon (SF Symbol `waveform`) appears in the menu bar within a few
seconds. Click it — the menu shows `Status: Starting…` while the
whisper model is downloaded (if not cached) and loaded into RAM,
then `Status: Ready` once Right Option is armed.

Expected status flow on the first run:

```
Status: Starting…
Status: Downloading whisper model…
Status: Loading whisper into RAM…
Status: Ready
```

Subsequent runs (model already cached):

```
Status: Starting…
Status: Loading whisper into RAM…
Status: Ready
```

The menu bar app downloads the model (if needed) and loads it into
RAM on startup before the hotkey is armed. Right Option is **not**
armed until status reaches `Ready` — so the first press is never
missed by the model still being on disk. silero-vad's model loads
lazily on the first recording (a few hundred ms) so we can keep
`Ready` strictly bound to the Whisper load alone — independent of
the silero-vad stub's `onnxruntime` session init inside the .app
bundle. In **API mode** the model download + load is skipped
entirely, so a configured API backend goes from app launch to
`Status: Ready` in under a second even on a Mac that hasn't fetched
the mlx weights.

**CLI mode (SSH/CI/tmux):**

```bash
dictate-mac daemon
```

Same lifecycle, but logs go to stdout instead of the menu bar.

> The `warmup` subcommand is still available for explicit download +
> load on metered connections, or for running a microphone sanity
> check.

### Where the model comes from

- **Source**: [`mlx-community/whisper-large-v3-turbo`](https://huggingface.co/mlx-community/whisper-large-v3-turbo).
  This is the same weights as `openai/whisper-large-v3-turbo`, just
  re-serialised into MLX format for Apple Silicon — quality and
  behaviour are identical.
- **When it downloads**: automatically on the first launch, in the
  background, before Right Option is armed. Subsequent launches read
  from the cache and skip the download.
- **Where it's cached**:
  `~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/`
  (~1.5 GB).
- **Size & time**: ~1.5 GB, usually 30–60 s on a typical home
  connection. The progress shows up in the menu bar as
  `Status: Downloading whisper model…`.
- **Pre-loading on a metered link**: run `dictate-mac warmup` once
  while you still have unmetered access — the download happens, the
  model is loaded into RAM, and the next interactive launch goes
  straight to `Loading… → Ready`.

## Installing on another Mac

The bundled `.app` is self-contained:

- Python 3.13 framework is embedded at
  `Contents/Frameworks/Python.framework/`.
- All Python wheels (`mlx`, `mlx_whisper`, `numpy`, `onnxruntime`,
  silero-vad stub, `huggingface_hub`, …) and the `mlx.metallib`
  GPU shim are bundled under `Contents/Resources/lib/`.
- `libportaudio.dylib` is a universal binary (`x86_64 + arm64`),
  bundled outside the zip so `dlopen` finds it on both
  architectures.
- **Total bundle size: ~284 MB** on disk.

Copy the `.app` to the target machine and double-click it.
Requirements on the target machine:

- **Apple Silicon only.** The bundled `mlx` wheel is arm64-only;
  Intel Macs will fail at `import mlx` with a confusing symbol
  error. There is no Intel fallback in v1.
- **macOS 13 or newer** (whatever the build machine had; the project
  is built and tested on 26.5.2).
- **Internet access on the first launch** — the Whisper model is
  downloaded from HuggingFace into `~/.cache/huggingface/hub/` the
  first time the daemon starts. After that it is read from cache.
- **Grant the three TCC permissions again** — *Privacy & Security →
  Microphone*, *Accessibility*, *Input Monitoring*. The grant is
  bound to the bundle id (`com.local.dictate-mac`), not to the file
  path, so re-installing the same `.app` keeps the grant, but moving
  it to a different Mac requires a fresh prompt.

### Transferring the model offline

If you don't want to download ~1.5 GB on the target machine, copy
the HF cache directory from the source machine first:

```bash
# On the source Mac (one that has already run dictate-mac once):
tar czf whisper-model.tgz \
  ~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo

# Transfer whisper-model.tgz by any means (AirDrop, scp, USB, …).

# On the target Mac, BEFORE first launch:
mkdir -p ~/.cache/huggingface/hub
tar xzf whisper-model.tgz -C ~/.cache/huggingface/hub
```

The cache directory is self-contained (`blobs/`, `snapshots/`,
`refs/`, `trees/`) and `dictate-mac` will pick it up automatically —
no HF network request is made once the directory is in place, and
`Status: Downloading whisper model…` is skipped on the very first
launch.

## Verifying the install

If something looks off (model won't load, VAD always returns silence,
typer keystrokes vanish), run the headless self-test:

```bash
dictate-mac selftest            # also records 1.5s from the mic
dictate-mac selftest --no-mic   # skip the microphone roundtrip
```

If you installed via the `.app` rather than from a venv, the same
CLI subcommands are reachable via the bundle's executable:

```bash
/Applications/DictateMac.app/Contents/MacOS/DictateMac selftest --no-mic
```

It checks, in order:

1. **model-load** — `mlx_whisper` weights load into RAM. (Skipped in
   API mode — use `--model-kind=api` against a stub to verify the
   no-op warmup path separately.)
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
11. **api-transcribe-auto-language** — when the language is
    `Auto-detect`, the `language` form field is omitted so the
    gateway falls back to its own detection.
12. **api-models-check** — mocked `GET /v1/models` handles 200+id,
    200-missing-id, 401, 404 and 500 correctly, and never leaks the
    fake API key string into any error.
13. **mic-roundtrip** *(unless `--no-mic`)* — record 1.5 s, run VAD + ASR,
    report durations.

Exit code is 0 if all checks pass, 1 otherwise. Each check prints a
PASS/FAIL line with a one-line detail.

## Usage

| Action                       | What happens                                       |
| ---------------------------- | -------------------------------------------------- |
| **Press Right Option**       | Recording starts (you'll hear a short Ping sound)  |
| Speak                        | Your voice is captured                             |
| **Press Right Option again** | Recording stops; recognition runs; text is typed   |

### The hotkey

- Use the **right** Option key specifically — not the left one.
- If Cmd / Ctrl / Shift is held at the same time, the press is ignored so
  system shortcuts aren't broken.
- Minimum speech length: **0.3 s** — shorter recordings produce no output.
- **Keyboard Maestro users:** if a macro is bound to the Option key, KM
  may intercept the event before dictate-mac sees it. Either disable the
  macro for testing or quit Keyboard Maestro.

### Where the text goes

Text is typed into the **currently focused** window. Works in: TextEdit,
Pages, Notes, browsers (address bar + form fields), Slack, Telegram,
VSCode, terminals, **Citrix Workspace / Citrix Viewer**.

### Audio feedback

- Recording start → system sound *Ping* (instant, on the first Option press)
- Recording stop → system sound *Pop* (instant, on the second Option press —
  fires *before* VAD/ASR/typing so you don't wait 1–3 s for acknowledgement)

Mute or change them in macOS Sound settings if you don't want them.

### Spacing

- dictate-mac appends a single space after every recognised phrase so the
  next dictation doesn't run into the previous one. If you don't want a
  space (e.g. when dictating continuous prose where you'll add punctuation
  manually), just type Backspace after the text appears.

## The menu bar

After `dictate-mac` (no arguments) launches, the icon (SF Symbol
`waveform`, template — automatically adapts to dark/light menu bar)
appears in the top-right of the menu bar.

Click the icon to open the menu:

| Item                       | Behaviour                                           |
| -------------------------- | --------------------------------------------------- |
| `Status: <state>`          | Disabled, read-only. Updated every 0.5 s.          |
| `Model (changing will restart app)` | Disabled, descriptive header.                        |
| `Local (mlx-community/whisper-large-v3-turbo)` | Clickable. Switches the active ASR backend to the local model and restarts the app so the change takes effect. Persists to `~/.config/dictate-mac/config.json`. |
| `API (<endpoint>)`         | Clickable. Opens the API credentials dialog (Endpoint / API key / Model ID). On a successful OK the dialog GETs `<endpoint>/models` with the bearer token to confirm the model id, persists the values, switches the active backend to API, and restarts the app. See [Model: local vs API](#model-local-vs-api) for details. |
| `Recognition language: <X>`| Clickable. Opens a submenu of `Auto-detect` and the 100 languages Whisper supports; the active option is prefixed with `✓`. Selecting one persists the choice and applies it to the **next** recording (no restart). See [Recognition language](#recognition-language) for details. |
| `Permissions …` header     | Disabled, descriptive label (no trailing colon).   |
| `Input Monitoring`         | Opens System Settings → Privacy & Security → Input Monitoring. |
| `Microphone`               | Opens System Settings → Privacy & Security → Microphone. |
| `Accessibility`            | Opens System Settings → Privacy & Security → Accessibility. |
| `Open log`                 | Opens the daemon log file (`~/Library/Logs/dictate-mac/dictate-mac.log` for the bundled `.app`) in the user's default app. In CLI mode logs go to stderr — the parent directory (or Console.app) is opened instead. |
| `About`                    | Opens https://github.com/vokasug/dictate-mac in the default browser. |
| `Restart`                  | Quits the app and re-opens it (via detached `osascript`). |
| `Quit`                     | ⌘Q (AppKit auto-renders the shortcut on the right). Stops the tap, releases the model, exits cleanly. |

`<state>` reflects the live daemon state:

- `Starting…` — process launched, running import-time setup.
- `Downloading whisper model…` — first-run HF download (skipped on
  subsequent launches).
- `Loading whisper into RAM…` — model weights are being loaded
  (~30–60 s on cold start, ~2 s from cache).
- `Ready` — Right Option is armed. silero-vad is NOT pre-loaded
  here — it warms up lazily on the first recording (sub-second on M1
  Air from the bundled `.jit`).
- `Recording…` — recording is in progress.
- `Transcribing…` — Whisper is running on the captured clip.
- `Typing…` — recognised text is being injected keystroke by
  keystroke.
- `Error: see logs` — something went wrong; the menu and logs
  describe it; Right Option is disarmed until you quit and relaunch.

While launching, Right Option is **not** armed — the first press is
guaranteed to be captured by the hotkey watcher only after status
reaches `Ready`.

## Recognition language

The fourth row of the menu (`Recognition language: <X>`) is a clickable
label that opens a submenu of choices. The currently-selected option is
prefixed with a checkmark (`✓`); clicking another entry makes that the
new default. The choice is persisted between launches — no restart of
the app is needed, and the model itself is **not** reloaded (Whisper
reads `language` per-call, so the next recording picks up the new
value directly). For the API backend the chosen language is also
forwarded as a `language` form field on every `POST
/v1/audio/transcriptions` request, so the gateway skips its own
language detection.

The submenu lists `Auto-detect` first, then the 100 ISO-639-1 languages
that Whisper supports, sorted alphabetically by their English display
name (`Russian`, `English`, `German`, …).

## Model: local vs API

The second section of the menu (`Model`) picks between the in-process
mlx-whisper model and an OpenAI-compatible API backend. Both backends
share the same audio pipeline (PortAudio → silero-vad → trimmed buffer)
— the only difference is what happens in the **Transcribing** step.

* **Local** is the default. Clicking the row switches back; the
  daemon restarts itself so the change takes effect at boot. The
  mlx-whisper weights stay in RAM regardless of which mode is
  selected until that restart.
* **API** opens a free-floating window with three fields:
  - **Endpoint** — the OpenAI-compatible base URL, e.g.
    `https://<host>/v1`. A trailing `/` is stripped at save time.
  - **API key** — the bearer token. The field shows bullets by
    default; click the eye icon next to it to reveal the value.
  - **Model ID** — the model id the gateway should use, e.g.
    `<your-model-id>`. The exact set of valid ids depends on the
    gateway — see its docs.

  Pressing **OK** disables the button (the label stays **OK** the
  whole time) and sends `GET <endpoint>/models` with the bearer
  token. The check passes only if the response contains the
  configured `Model ID` in the `data` array. Failures are shown
  inline (red label below the model id field) — the dialog stays
  open so you can fix the input; only a successful check closes the
  window and persists the values. On success the daemon restarts
  itself so the new backend (API, no local model load) takes effect
  at the next boot.

  When the dialog first opens, the white-text line under the
  fields shows the absolute path to `config.json` — useful when
  you want to inspect or hand-edit the saved credentials.

  The validation surfaces a categorised message for the most common
  failure modes:

  | Symptom                                 | Message                                                                 |
  | --------------------------------------- | ----------------------------------------------------------------------- |
  | wrong API key                           | `Authentication failed — check the API key (HTTP 401)`                  |
  | endpoint without `/v1` (or wrong host)  | `Models endpoint not found — confirm the URL ends with /v1 (current: …)`|
  | model id not offered by the gateway     | `Model ID '<id>' not found at <endpoint> (response listed N model(s))`  |
  | network unreachable / DNS / timeout     | `Could not reach <endpoint>: <reason>`                                 |

  The API key is **never** logged or written into any error message
  — only the endpoint, the HTTP status, and (when useful) a truncated
  response body. The whole config file (including the API key) sits
  at `~/.config/dictate-mac/config.json` with mode `0o600`. The
  key is stored as plain text — anyone with shell access to your
  home folder can read it. Treat the file the same way you treat a
  `.env` file: don't symlink it into a shared Dropbox folder, don't
  commit it to a dotfiles repo.

Switching between **Local** and **API** always triggers a self-restart.
Switching from API back to Local does **not** erase the saved
credentials — they remain on disk so re-enabling API later is a
single click. Reopening the dialog after a previous successful save
auto-populates all three fields (the key field shows bullets again,
not the plain value — click the eye to confirm).

The chosen recognition language is automatically forwarded as a
`language` form field on every `POST /v1/audio/transcriptions`
request when one is set in the menu. With `Auto-detect` selected the
field is omitted and the gateway falls back to its own detection.

CLI mode (`dictate-mac daemon`) takes the same four settings from
flags: `--model-kind={local,api}`,
`--api-endpoint=<url>`, `--api-key=<key>`,
`--model-id=<id>`. The CLI does **not** validate the endpoint at
startup — a typo surfaces as a runtime error in the log on the first
recording, exactly like with the local backend. In API mode the
daemon skips the local-model warmup entirely, so startup is
immediate even on a Mac where the mlx weights are not yet
downloaded.

### Where the choice is stored

The choice lives in a single JSON file at

```
$XDG_CONFIG_HOME/dictate-mac/config.json
```

with `$XDG_CONFIG_HOME` defaulting to `~/.config/`. On a typical Mac
that resolves to:

```
~/.config/dictate-mac/config.json
```

The file is written atomically (`os.replace` from a sibling `.tmp`),
with permissions `0o700` on the directory and `0o600` on the file. The
contents look like:

```json
{
  "_v": 2,
  "language": "ru",
  "model_kind": "api",
  "api_endpoint": "https://<your-host>/v1",
  "api_key": "<your-bearer-token>",
  "api_model_id": "<your-model-id>"
}
```

`_v` is the schema version (currently `2`); `language` is either an
ISO-639-1 code (`"ru"`, `"en"`, `"de"`, …) or the sentinel `"auto"`.
`model_kind` is `"local"` (default) or `"api"`. The remaining three
fields are only meaningful with `model_kind=api`.

Older configs written by v0.2.x (schema v1, only `language`) load
unchanged — the missing fields default to `local` and empty strings,
and the very next save rewrites the file as v2.

### First-run behaviour

When the menu bar app launches and the config file does not yet
exist, dictate-mac detects your macOS primary language and writes the
chosen value to disk. Detection calls one API:

- `Foundation.NSLocale.preferredLanguages()` (PyObjC, native macOS
  API, canonical BCP-47, ~1 ms, in-process).

The first entry that maps to a code Whisper supports (mirrored from
`mlx_whisper.tokenizer.LANGUAGES`) is written as-is; otherwise the
value `"auto"` is written (Whisper will then run its own language
detection on every recording, which adds ~0.3-0.8 s of overhead per
clip on M1).

### Corrupt-config policy

If the file is present but invalid — malformed JSON, a non-string
`language` field, an unknown `model_kind`, or a malformed API
endpoint — the menu bar logs a warning pointing at the cause, falls
back to a fresh in-memory default, and **does not overwrite the
file**. You can repair it by hand and the next launch will read it
normally.

### CLI subcommands never touch the config file

`dictate-mac daemon`, `dictate-mac warmup`, and `dictate-mac selftest`
take all their settings from CLI flags — `--language`, `--model-kind`,
`--api-endpoint`, `--api-key`, `--model-id` — and do not read or
write `~/.config/dictate-mac/config.json` under any circumstances.
This is intentional: CLI runs are deterministic, don't need
filesystem state, and won't race with the menu bar app.

Example:

```bash
# Local model, Russian, this CLI session only (config file untouched):
dictate-mac daemon --language=ru

# Auto-detect for this CLI session (default):
dictate-mac daemon

# API backend — endpoint/key/id must all be provided:
dictate-mac daemon \
  --model-kind=api \
  --api-endpoint=https://<your-host>/v1 \
  --api-key=<your-bearer-token> \
  --model-id=<your-model-id>
```

### Where the logs go

When you launch DictateMac.app from Finder, Launchpad, or
`open DictateMac.app` there is no terminal to write to — so logs go
to `~/Library/Logs/dictate-mac/dictate-mac.log` (truncated on every
launch). When you run `dictate-mac` from a terminal the logs still
go to stderr; the same `logutils` module decides based on whether
`sys.executable` is inside a `DictateMac.app` bundle.

## How It Works

```
┌────────── dictate-mac daemon (Python 3.13) ──────────────────────┐
│                                                                   │
│   ┌────────────────────┐    ┌──────────────────┐                  │
│   │ Menu bar UI        │    │ Hotkey watcher   │                  │
│   │ rumps NSStatusItem │    │ Quartz CGEvent   │                  │
│   │ waveform + Status  │    │ tap (Right Opt)  │                  │
│   └─────────┬──────────┘    └────────┬─────────┘                  │
│             │ state (0.5 s poll)     │ events                     │
│             │                        ▼                            │
│             │            ┌────────────────────────┐               │
│             │            │ State machine          │               │
│             │            │ (asyncio loop)         │               │
│             │            │ STARTING→DOWNLOADING   │               │
│             │            │ →LOADING→READY         │               │
│             │            │ ⇄RECORDING→TRANSCRIBING│               │
│             │            │ →TYPING→READY          │               │
│             │            └─────┬────┬─────────┬────┘               │
│             │                  │    │         │                     │
│             ▼                  ▼    ▼         ▼                     │
│       (status only)  sounddevice  silero-vad                       │
│                      16 kHz PCM  ONNX (~2 MB)                       │
│                              │                                   │
│                    ┌─────────┴─────────┐                         │
│                    ▼                   ▼                         │
│            mlx-whisper (local)   POST /v1/audio/transcriptions  │
│            ~1.5 GB RAM           bearer + model id (api)         │
│                                                                   │
│                          │ text                                  │
│                          ▼                                       │
│       ┌────────────────────────────────────────────┐             │
│       │ Quartz CGEvent Unicode keystroke typer     │             │
│       │ (CGEventKeyboardSetUnicodeString)          │             │
│       └────────────────────────────────────────────┘             │
└───────────────────────────────────────────────────────────────────┘
```

Three threads at runtime:

- **Main thread** — `NSApp` (rumps). Owns the menu bar icon and
  context menu; refreshes `Status:` from `State` every 0.5 s.
- **Worker thread (asyncio)** — the state machine, recorder,
  Whisper, typer.
- **CFRunLoop thread (hotkey)** — Quartz `CGEventTap`, pushes
  events into a thread-safe `queue.Queue`.

Communication is `queue.Queue` (hotkey events) plus a
`threading.Lock`-guarded `state` property — no extra threading
primitives.

The ASR backend is selected at startup from `~/.config/dictate-mac/
config.json`'s `model_kind` field. In **local** mode the daemon
downloads the whisper model (if not cached) and loads it into RAM
on startup, then arms the hotkey. Status reaches `Ready` only after
that completes. In **API** mode the local-model load is skipped,
`Status: Ready` arrives within a second, and audio is POSTed to
the configured gateway per recording. silero-vad's ONNX model loads
lazily on the first recording (a few hundred ms on M1) so the very
first recording pays a tiny extra cost and every subsequent one does
not. Quit releases everything cleanly. Local-mode recognitions take
~3 s on M1; API-mode recognitions take only the round-trip time to
the gateway (typically 0.3-1 s).

## Troubleshooting

### "Right Option press isn't detected"

1. On **macOS 14+** confirm **Input Monitoring** is granted to the
   process running dictate-mac. For the bundled `.app` the grant
   goes to `com.local.dictate-mac`; for `dictate-mac` run from the
   source venv it goes to the **Python interpreter**; for
   `dictate-mac daemon` it goes to Terminal.app / iTerm.app.
2. Click **Input Monitoring** in the menu bar — it opens the
   matching System Settings pane directly. Toggle the entry off
   and back on if it is already enabled (forces macOS to re-attach
   the CGEventTap).
3. Quit the app (menu → Quit ⌘Q) and relaunch. **Quit and reopen
   Terminal** if running CLI mode.
4. In CLI mode, run with `--log-level=DEBUG` for a verbose trace:
   ```bash
   dictate-mac daemon --log-level=DEBUG
   ```
5. If Right Option still does nothing, a system-level tool
   (Keyboard Maestro, Karabiner-Elements, AltTab, CuaDriver) is
   intercepting the key before our CGEventTap. Quit that tool
   and retry.

### "Russian text doesn't appear in Citrix"

In `Citrix Viewer → Preferences → Keyboard`, enable
**Send Unicode keyboard input** (default in modern versions). If that
doesn't help, try the alternate typer (CLI mode):

```bash
dictate-mac daemon --output=osascript
```

### "Model download fails"

Verify connectivity to Hugging Face:

```bash
curl -I https://huggingface.co/mlx-community/whisper-large-v3-turbo
```

Use a VPN or mirror if access is blocked. The progress is also visible
as `Status: Downloading whisper model…` in the menu bar.

### "`mlx` fails to install"

`mlx` ships wheels only for Python 3.10–3.13 on Apple Silicon. Confirm:

```bash
python3 --version   # must be 3.13.x
```

If you have 3.14, recreate the venv:

```bash
uv venv --python 3.13 .venv --force
uv pip install -e .
```

### "Too much RAM usage"

On an 8 GB Mac, avoid running Chrome with many tabs alongside the app.
The model permanently occupies ~1.5 GB after first launch
(loaded into RAM at startup, just before `Status: Ready`).

### "Status is stuck on Starting… / Downloading…"

When you launch from the bundled `DictateMac.app`, logs go to
`~/Library/Logs/dictate-mac/dictate-mac.log` (truncate-on-start) —
open it in Console.app or `tail -F` from a terminal:

```bash
tail -F ~/Library/Logs/dictate-mac/dictate-mac.log
```

When you launch from a terminal (`dictate-mac` or `dictate-mac daemon`),
logs already go to stderr — bump the verbosity with:

```bash
dictate-mac --log-level=DEBUG 2>&1 | tee /tmp/dictate-mac.log
```

Common causes:

- Hugging Face unreachable — see *Model download fails* above.
- macOS blocked network at first launch because the `.app` isn't
  code-signed (local-use only) — quit and reopen, allow in the
  Network prompt if asked.
- The Python interpreter in your venv is not 3.13.x — verify with
  `python3 --version`.
- The `.app` can't find its embedded Python (`dlopen` errors) — make
  sure you ran the build with the same `.venv` you used during
  `uv pip install -e .`. Rebuild with `./build.sh --clean`.

## Uninstall

```bash
# 1. Quit the .app (menu → Quit ⌘Q) or Ctrl-C in CLI mode.
# 2. Remove the project (and the .app if you copied it into /Applications)
rm -rf ~/dictate-mac
rm -rf /Applications/DictateMac.app      # only if you copied it there

# 3. Remove logs and optional model cache (~1.5 GB)
rm -rf ~/Library/Logs/dictate-mac
rm -rf ~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo
```

You can leave the granted Privacy & Security permissions in place —
they don't affect anything once the app is gone. To remove them too,
revoke `com.local.dictate-mac` (the bundled `.app` id) in
*Privacy & Security → Microphone / Accessibility / Input Monitoring*.

## Known Limitations

- ~1.5 GB of RAM is held permanently after the first startup **when
  using the local backend**. The API backend doesn't load any ASR
  weights into memory — its startup is instantaneous and can run on
  Mac models where the mlx weights aren't downloaded.
- Citrix Viewer requires **Send Unicode keyboard input** to be enabled
  (default in modern versions).
- macOS TCC permissions (Microphone, Accessibility, Input Monitoring)
  must be granted manually. With the bundled `DictateMac.app` they go
  to `com.local.dictate-mac`; with the menu-bar app run from the
  venv they go to the **Python interpreter**; with `dictate-mac daemon`
  they go to Terminal/iTerm. One-time per-machine grant.
- The app is launched manually in v1 — no LaunchAgent, no auto-start
  at login. Adding a Login Item is on the roadmap; not built in v1.
- No code signing. The bundled `.app` is unsigned and ad-hoc
  unnotarised; macOS Gatekeeper may prompt on first launch — right
  click → Open to bypass the quarantine.
- The bundled `.app` keeps silero-vad lazy-loaded (a sub-second hit
  on the *first* recording). The bundle ships a numpy + onnxruntime
  re-implementation of `silero_vad` at
  `Contents/Resources/lib/python3.13/silero_vad/`, so the runtime
  inference path runs through `onnxruntime` against the bundled
  `silero_vad.onnx` (2.2 MB). The real `silero_vad` wheel — which
  imports `torch` and `torchaudio` at module top-level — would have
  pulled in `libtorch_cpu.dylib` (263 MB) plus the rest of the
  429 MB torch tree, plus `libtorchaudio` (whose install-name
  chain hangs `dlopen`). The stub avoids both. Consequence: **the
  bundled `.app` cannot transcode audio files**
  (`silero_vad.read_audio("foo.wav")` and `torchaudio.load()` both
  raise `RuntimeError`). All audio in the bundle flows through
  PortAudio → `numpy.ndarray` → silero-vad → whisper — the only
  path we use.
- The hotkey tap stays disabled if Input Monitoring was not granted
  before launch. Grant it in System Settings → Privacy & Security →
  Input Monitoring, then Quit + reopen DictateMac (macOS itself
  prompts for the relaunch).
- Switching the ASR backend (**Local** ↔ **API**) always triggers a
  restart. The menu click is automatic; the user is dropped into the
  freshly-launched bundle with the new backend active.
- Python 3.13.x only; 3.14 is unsupported because `mlx` does not yet
  ship a wheel for it.