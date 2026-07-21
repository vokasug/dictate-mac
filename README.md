# dictate-mac 🎙️

Voice dictation for macOS with **two ASR backends** selectable at runtime:

- **Local** — `mlx-community/whisper-large-v3-turbo` runs in-process via
  MLX on Apple Silicon (~1.5 GB RAM, works fully offline).
- **API** — any OpenAI-compatible `/v1/audio/transcriptions` endpoint
  (your gateway, a hosted service, …). No ASR model loaded; only audio
  is uploaded per recording.

Press **Right Option** → speak → press **Right Option** again.
Recognized text is typed character-by-character into the focused
window — including Citrix Workspace sessions that don't share the
clipboard.

- 100 ISO-639-1 languages + `Auto-detect` — applies to both backends
- Settings (including API credentials) persist at
  `~/.config/dictate-mac/config.json` with mode `0o600`; the API key
  is never logged
- Menu bar app, default invocation `dictate-mac`; CLI daemon
  `dictate-mac daemon` for SSH/CI

---

## Contents

- [Requirements](#requirements)
- [Install](#install)
- [Run from source](#run-from-source)
- [Build the `.app` from source](#build-the-app-from-source)
- [Usage](#usage)
- [The menu bar](#the-menu-bar)
- [Recognition language](#recognition-language)
- [Choosing your backend: Local vs API](#choosing-your-backend-local-vs-api)
- [The config file](#the-config-file)
- [CLI subcommands](#cli-subcommands)
- [How it works](#how-it-works)
- [Verifying & troubleshooting](#verifying--troubleshooting)
- [Uninstall & known limitations](#uninstall--known-limitations)

---

## Requirements

To *run* dictate-mac:

- **Apple Silicon Mac** (M1 / M2 / M3 / M4)
- **macOS 13 Ventura** or newer
- **8 GB RAM** or more
- Any working microphone (the built-in one is fine)
- ~284 MB for the `.app` itself, plus an optional ~1.5 GB for the local
  Whisper weights (downloaded on first launch, only in **Local** mode;
  **API** mode needs no extra disk)
- Internet access only for the first local-model download — API mode
  needs none

To *run from source* you additionally need:

- **Python 3.13** — the simplest install path is
  [`uv`](https://github.com/astral-sh/uv); it manages the project
  venv and the runtime deps in one shot.

To *build* the `.app` from source you additionally need the Xcode
Command Line Tools (`xcode-select --install`).

## Install

The recommended path: take the prebuilt `DictateMac.app` from the
GitHub release, copy it into `/Applications`, and open it.

### 1. Download

[`DictateMac-v0.4.1-macos.zip`](https://github.com/vokasug/dictate-mac/releases/latest/download/DictateMac-v0.4.1-macos.zip)
from the [latest release](https://github.com/vokasug/dictate-mac/releases/latest).
Compressed download is ~110–140 MB; the extracted `.app` is ~284 MB.

Verify the download with the bundled `.sha256`:

```bash
shasum -a 256 DictateMac-v0.4.1-macos.zip
# compare with the contents of the .sha256 file from the same release
```

### 2. Install

Double-click the zip in Finder, or from a terminal:

```bash
open DictateMac-v0.4.1-macos.zip   # expands to ./DictateMac.app
mv DictateMac.app /Applications/   # optional — keeps the .app
                                  # alongside your other apps
open /Applications/DictateMac.app
```

> **First-launch Gatekeeper.** The `.app` is unsigned and
> ad-hoc-notarised. If macOS refuses to open it, right-click
> `DictateMac.app` in Finder → **Open** → confirm in the dialog.
> Subsequent launches are double-click as usual.

### 3. Grant the three permissions

macOS prompts for *Microphone* the very first time you press Right
Option; the menu bar stays on `Status: Error: see logs` until you
also grant *Accessibility* and *Input Monitoring* by hand in
**System Settings → Privacy & Security**. See the
[Permissions row in the menu](#the-menu-bar) for one-click shortcuts
to each pane, and [Troubleshooting](#verifying--troubleshooting) for
fixes if any are still missing.

### 4. Pick your backend

The default is **Local** — the in-process Whisper model. You don't
have to do anything; the app works offline out of the box.

To use an external service instead:

1. Click the menu icon → **Model (changing will restart app)** →
   **API**.
2. Fill in **Endpoint** (e.g. `https://<your-host>/v1`), **API key**,
   and **Model ID** the gateway should use.
3. Click **OK**. The dialog GETs `{endpoint}/models` to verify the
   endpoint and the model id. On success the values are saved and the
   daemon restarts so the API backend takes effect at boot.

To switch back, pick **Local** in the same menu. See
[Choosing your backend](#choosing-your-backend-local-vs-api) for
full details on validation, error messages, and the config file.

### 5. Pick the language

Click the menu bar icon → **Recognition language** → *Auto-detect*
or any of the 100 ISO-639-1 codes. The choice persists and applies
to the next recording (no model reload, applies to both backends).
For the API backend the chosen language is also forwarded as a
`language` form field on every request, so the gateway skips its own
detection.

### 6. Use

Press **Right Option**, speak, press **Right Option** again. The
recognised text is typed character-by-character into the currently
focused window — including Citrix Workspace sessions that don't
forward the macOS clipboard.

## Run from source

If you want the development loop — instant edits to source with no
rebuild cycle, or a CLI daemon over SSH — install from source:

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

The tradeoff vs. the `.app` install: macOS TCC permissions
(Microphone, Accessibility, Input Monitoring) attach to the **Python
interpreter** running the script (or to Terminal/iTerm when you use
`daemon` from a shell), not to a stable bundle id. If you run from
the venv and later switch to the `.app`, grant the permissions once
more for `com.local.dictate-mac`.

## Build the `.app` from source

Most users will not need this — the [Install](#install) path above
pulls a prebuilt `.app` from the GitHub release. Rebuild from source
only if you want to run a private fork, ship your own signed /
notarised variant, or inspect what `py2app` produced.

```bash
./build.sh
open dist/DictateMac.app
```

`./build.sh` regenerates the `.icns` icon and runs `py2app.build_app`.
The result is fully self-contained — everything except the ~1.5 GB
local Whisper weights (downloaded at first launch) is bundled.
**Bundle size: ~284 MB.** Dev-side details (stubs, post-build strip
pass, install_name / signature rewrites) live in
[`AGENTS.md` § 6](../blob/main/AGENTS.md#6-build-pipeline).

## Usage

| Action                       | What happens                                       |
| ---------------------------- | -------------------------------------------------- |
| **Press Right Option**       | Recording starts (you'll hear a short Ping sound)  |
| Speak                        | Your voice is captured                             |
| **Press Right Option again** | Recording stops; recognition runs; text is typed   |
| **Press Esc while recording**| Recording is cancelled — nothing is recognised or typed |

- **Use the right Option key specifically** — not the left one. If
  Cmd / Ctrl / Shift is held at the same time, the press is ignored
  so system shortcuts aren't broken.
- **Minimum speech length: 0.3 s** — shorter recordings produce no
  output.
- **Audio feedback** — Recording start plays *Ping*, recording stop
  plays *Pop* (instant, before VAD/ASR/typing). Cancelling with Esc
  also plays *Pop*. Mute in macOS Sound
  settings if you don't want them.
- **Spacing** — dictate-mac appends a single space after every
  recognised phrase so the next dictation doesn't run into the
  previous one. Press Backspace after the text if you don't want a
  space.
- **Where the text goes** — TextEdit, Pages, Notes, browsers
  (address bar + form fields), Slack, Telegram, VSCode, terminals,
  **Citrix Workspace / Citrix Viewer**.

## The menu bar

After `dictate-mac` (no arguments) launches, the icon (SF Symbol
`waveform`, template — adapts to dark/light menu bar) appears in the
top-right.

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

| Item                       | Behaviour                                           |
| -------------------------- | --------------------------------------------------- |
| `Status: <state>`          | Disabled, read-only. Updated every 0.5 s.            |
| `Model (changing will restart app)` | Disabled, descriptive header.                        |
| `Local (...)`              | Clickable. Switches to the local mlx-whisper backend and restarts the daemon. Persists to `~/.config/dictate-mac/config.json`. |
| `API (<endpoint>)`         | Clickable. Opens the API credentials dialog (Endpoint / API key / Model ID). On a successful OK the dialog GETs `<endpoint>/models` to verify the endpoint + model id, persists the values, switches the active backend to API, and restarts the daemon. |
| `Recognition language: <X>`| Clickable. Opens a submenu of `Auto-detect` and the 100 ISO-639-1 languages Whisper supports. Selecting one persists the choice and applies to the next recording (no restart). |
| `Permissions (… )` header  | Disabled, descriptive label.                        |
| `Input Monitoring`         | Opens System Settings → Privacy & Security → Input Monitoring. |
| `Microphone`               | Opens System Settings → Privacy & Security → Microphone. |
| `Accessibility`            | Opens System Settings → Privacy & Security → Accessibility. |
| `Open log`                 | Opens the daemon log file (`~/Library/Logs/dictate-mac/dictate-mac.log` for the bundled `.app`) in the default app for `.log`. In CLI mode logs go to stderr, so the parent directory (or Console.app) is opened instead. |
| `About`                    | Opens https://github.com/vokasug/dictate-mac in the default browser. |
| `Restart`                  | Quits the app and re-opens it (via detached `osascript`). |
| `Quit`                     | ⌘Q (AppKit auto-renders the shortcut on the right). |

`<state>` reflects the live daemon state:

- `Starting…` — process launched, running import-time setup.
- `Downloading whisper model…` — first-run HF download (skipped in
  API mode and on subsequent local launches).
- `Loading whisper into RAM…` — model weights are being loaded
  (~30–60 s cold, ~2 s from cache). Skipped in API mode.
- `Ready` — Right Option is armed. silero-vad is NOT pre-loaded here
  — it warms up lazily on the first recording.
- `Recording…` — recording is in progress.
- `Transcribing…` — ASR backend is running on the captured clip.
- `Typing…` — recognised text is being injected keystroke by
  keystroke.
- `Error: see logs` — something went wrong; the menu and logs
  describe it. A failed model download/load is retryable in place:
  the menu shows `Model download failed — press Right Option to
  retry` and the next Right Option press re-runs the warmup. Only
  permission errors require Quit + reopen.

## Recognition language

The fourth row of the menu (`Recognition language: <X>`) opens a
submenu of choices. The currently-selected option is prefixed with a
checkmark (`✓`); clicking another entry makes that the new default.
The choice is persisted between launches — no restart, no model
reload. For the API backend the language is forwarded as the
`language` form field on every `POST /v1/audio/transcriptions` request
so the gateway skips its own detection. With `Auto-detect` selected
the field is omitted.

The submenu lists `Auto-detect` first, then the 100 ISO-639-1
languages Whisper supports, sorted alphabetically by their English
display name (`Russian`, `English`, `German`, …).

## Choosing your backend: Local vs API

The two backends share the same audio pipeline (PortAudio →
silero-vad → trimmed buffer) — the only difference is what happens
in the **Transcribing** step.

### Local

- **What it does:** runs the in-process mlx-whisper
  (`mlx-community/whisper-large-v3-turbo`).
- **Cost:** ~1.5 GB RAM permanently after first warmup.
- **Disk:** ~1.5 GB Whisper weights downloaded on first launch to
  `~/.cache/huggingface/hub/`. Cached for subsequent launches.
- **Network:** none after the first download — works fully offline.
- **Speed:** ~3 s per recording on M1.
- **Restart on switch:** clicking **API** in the menu restarts the
  daemon (the new backend takes effect at boot).

### API

- **What it does:** POSTs 16 kHz mono WAV to
  `{endpoint}/audio/transcriptions` with `Authorization: Bearer
  <key>` and `model=<id>`. The same silero-vad trim runs upstream,
  so the gateway only ever receives speech.
- **Cost:** zero extra RAM (no model loaded).
- **Disk:** no ASR weights.
- **Network:** every recording.
- **Speed:** round-trip to gateway (typically 0.3–1 s).
- **Config:** the **API** menu row opens a dialog with three fields
  (Endpoint, API key, Model ID). On OK the dialog GETs
  `{endpoint}/models` with the same bearer token and verifies the
  configured Model ID appears in the response's `data` array.
  Failures surface as categorised messages:

  | Symptom                                 | Message                                                                 |
  | --------------------------------------- | ----------------------------------------------------------------------- |
  | wrong API key                           | `Authentication failed — check the API key (HTTP 401)`                  |
  | endpoint without `/v1` (or wrong host)  | `Models endpoint not found — confirm the URL ends with /v1 (current: …)`|
  | model id not offered by the gateway     | `Model ID '<id>' not found at <endpoint> (response listed N model(s))`  |
  | network unreachable / DNS / timeout     | `Could not reach <endpoint>: <reason>`                                 |

  The API key is **never** logged or written into any error message —
  only the endpoint, HTTP status, and a truncated response body.
- **Switching back to Local** does **not** erase the saved
  credentials — they stay on disk for re-enabling later.
- **Language forwarding:** when a language is set in the menu it's
  sent as the `language` form field on every API request, so the
  gateway skips its own detection. With `Auto-detect` the field is
  omitted.

### Switching and persistence

Switching backends always triggers a self-restart (~0.5 s) so the
new mode takes effect at boot. The menu click does this
automatically; you are dropped into the freshly-launched bundle with
the new backend active. The credentials dialog reopens pre-populated
on subsequent calls (the key field shows bullets, not the plain
value — click the eye to reveal).

## The config file

All settings (language, backend, API credentials) live in a single
JSON file at:

```
$XDG_CONFIG_HOME/dictate-mac/config.json
```

`$XDG_CONFIG_HOME` defaults to `~/.config/`, so on a typical Mac
that resolves to:

```
~/.config/dictate-mac/config.json
```

The file is written atomically (`os.replace` from a sibling `.tmp`),
with permissions `0o700` on the directory and `0o600` on the file.
Schema v2 contents look like:

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

- `_v` is the schema version (currently `2`).
- `language` is either an ISO-639-1 code (`"ru"`, `"en"`, …) or the
  sentinel `"auto"`.
- `model_kind` is `"local"` (default) or `"api"`.
- The remaining three fields are only meaningful with
  `model_kind=api`.

Older configs written by v0.2.x (schema v1, only `language`) load
unchanged — the missing fields default to `local` and empty strings,
and the very next save rewrites the file as v2.

If the file is present but invalid (malformed JSON, a non-string
`language` field, an unknown `model_kind`, or a malformed API
endpoint), the menu bar logs a warning, falls back to a fresh
in-memory default, and **does not overwrite the file**. You can
repair it by hand.

CLI subcommands (`daemon`, `warmup`, `selftest`) take their settings
from CLI flags and **do not** read or write this file.

## CLI subcommands

`dictate-mac` (no subcommand) launches the menu bar app. The
following subcommands are also available from both the menu-bar
binary and the source venv:

| Subcommand | Purpose                                                                  |
| ---------- | ------------------------------------------------------------------------ |
| `daemon`   | Plain CLI daemon (no menu bar). Same code path, logs to stderr.          |
| `warmup`   | Download the local model (if needed), load it, then exit.                |
| `selftest` | Headless smoke checks; `--no-mic` skips the mic roundtrip.              |

Common flags (before or after the subcommand): `--quiet`,
`--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`,
`--output {quartz,osascript}`, `--language {auto|<iso-639-1>}`,
`--model-kind {local,api}`, `--api-endpoint <url>`, `--api-key <key>`,
`--model-id <id>`. The last four are meaningful only with
`--model-kind=api`; `daemon` refuses to start without the three
required values when `api` is selected. CLI runs do not read or
write the persisted config file — language and ASR settings are
taken from flags directly.

Examples:

```bash
# Local backend, Russian, this CLI session only (config file untouched):
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

When the bundled `.app` is the entry point, the same subcommands are
reachable via the bundle's executable:

```bash
/Applications/DictateMac.app/Contents/MacOS/DictateMac selftest --no-mic
```

Logs go to `~/Library/Logs/dictate-mac/dictate-mac.log` (truncate-on-start)
when launched as a bundle, or to stderr when run from a terminal —
the same `logutils` module decides based on whether `sys.executable`
is inside a `DictateMac.app` bundle. The **Open log** menu item
opens the log file in your default `.log` viewer (Console.app).

## How it works

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
│            recorder · silero-vad · ASR backend · typer           │
│                              │                                   │
│                    ┌─────────┴─────────┐                         │
│                    ▼                   ▼                         │
│            mlx-whisper (local)   POST /v1/audio/transcriptions  │
│            ~1.5 GB RAM           bearer + model id (api)         │
│                                                                   │
│                          │ text                                  │
│                          ▼                                       │
│       Quartz CGEvent Unicode keystroke typer                      │
│       (CGEventKeyboardSetUnicodeString)                           │
└───────────────────────────────────────────────────────────────────┘
```

Three threads at runtime: **Main** (rumps / NSApp — menu + 0.5 s
status refresh), **Worker** (asyncio state machine — recorder + ASR
+ typer), **CFRunLoop** (CGEventTap for Right Option, pushes events
into a thread-safe `queue.Queue`). Communication is `queue.Queue`
plus a `threading.Lock`-guarded `state` property — no extra
primitives.

The ASR backend is selected at startup from the config file's
`model_kind` field. In **local** mode the daemon downloads the
Whisper model (if not cached) and loads it into RAM before arming
the hotkey; status reaches `Ready` only after that completes. In
**API** mode the local-model load is skipped, `Status: Ready`
arrives within a second, and audio is POSTed to the configured
gateway per recording. silero-vad's ONNX model loads lazily on
the first recording. Local-mode recognitions take ~3 s on M1;
API-mode recognitions take only the round-trip time to the gateway
(typically 0.3–1 s).

## Verifying & troubleshooting

If something looks off, run the headless self-test:

```bash
dictate-mac selftest            # also records 1.5s from the mic
dictate-mac selftest --no-mic   # skip the microphone roundtrip
```

If you installed via the `.app` rather than from a venv, the same
subcommands are reachable via the bundle's executable
(`/Applications/DictateMac.app/Contents/MacOS/DictateMac selftest`).
Exit code 0 if all 16 checks pass, 1 otherwise. Each check prints a
PASS/FAIL line with a one-line detail; the checks are
`model-load`, `vad-silence`, `vad-speech-like`, `asr-smoke`,
`typer-dispatch`, `ssl-certifi`, `config-v1-migration`,
`config-invalid-endpoint`, `config-api-required-when-api`,
`audio-wav-roundtrip`, `api-transcribe-headers`,
`api-transcribe-auto-language`, `api-models-check`, `warmup-retry`,
`recorder-portaudio-retry`, `hotkey-escape-event`, plus an optional
`mic-roundtrip`.

**FAQ:**

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Right Option press isn't detected | One of the three TCC permissions missing | Click **Input Monitoring** in the menu → grant. Quit + reopen the app. For the venv, the grant goes to the **Python interpreter** instead. |
| `Status: Error: see logs` after first Right Option press | Microphone not granted | macOS will prompt on first press; if you declined, open System Settings → Privacy & Security → Microphone and toggle dictate-mac on. |
| Russian text doesn't appear in Citrix | Citrix unicode input disabled | In Citrix Viewer → Preferences → Keyboard, enable **Send Unicode keyboard input** (default in modern versions). Or use `--output=osascript`. |
| Status stuck on `Starting…` / `Downloading…` | Hugging Face unreachable (local mode), or the bundle can't find embedded Python | `curl -I https://huggingface.co/mlx-community/whisper-large-v3-turbo`. If the model is downloaded but loading still fails, rebuild with `./build.sh --clean`. |
| `Model download failed — press Right Option to retry` | The first-run download failed (no network, DNS/VPN hiccup, HF outage) | Bring the network up and press **Right Option** — the warmup re-runs without an app restart. Check the log via **Open log** if it keeps failing. |
| `mlx` fails to install / load | Python 3.14 (no wheel) | Recreate venv: `uv venv --python 3.13 .venv --force && uv pip install -e .` |
| API mode returns HTTP 429 | The OpenAI-compatible gateway is rate-limiting | Some gateways (e.g. `whisper-large-v3-turbo`) have a 1-request-per-10-15-seconds cap on certain upstream providers. Switch the menu to a different model id, or to Local if the same model is acceptable. |
| Too much RAM usage | Local backend occupies ~1.5 GB permanently | API mode doesn't load any ASR weights; switch if RAM is tight. |

## Uninstall & known limitations

```bash
# 1. Quit the .app (menu → Quit ⌘Q) or Ctrl-C in CLI mode.
# 2. Remove the project (and the .app if you copied it into /Applications)
rm -rf ~/dictate-mac
rm -rf /Applications/DictateMac.app      # only if you copied it there

# 3. Remove logs and optional local model cache (~1.5 GB)
rm -rf ~/Library/Logs/dictate-mac
rm -rf ~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo
```

You can leave the granted Privacy & Security permissions in place —
they don't affect anything once the app is gone. To remove them too,
revoke `com.local.dictate-mac` in
*Privacy & Security → Microphone / Accessibility / Input Monitoring*.

**Known limitations:**

- ~1.5 GB RAM is held permanently after the first startup **when
  using the local backend**. The API backend doesn't load any ASR
  weights into memory; its startup is instantaneous and can run on
  Mac models where the mlx weights aren't downloaded.
- macOS TCC permissions (Microphone, Accessibility, Input Monitoring)
  must be granted manually. With the bundled `DictateMac.app` they
  go to `com.local.dictate-mac`; with the menu-bar app run from
  the venv they go to the **Python interpreter**; with
  `dictate-mac daemon` they go to Terminal/iTerm. One-time per-machine
  grant.
- Citrix Viewer requires **Send Unicode keyboard input** to be
  enabled (default in modern versions).
- The app is launched manually in v1 — no LaunchAgent, no
  auto-start at login.
- The `.app` is unsigned, ad-hoc, not notarised. macOS Gatekeeper
  may prompt on first launch — right-click → Open to bypass the
  quarantine.
- The hotkey tap stays disabled if Input Monitoring was not granted
  before launch. Grant it in System Settings → Privacy & Security →
  Input Monitoring, then Quit + reopen DictateMac.
- Switching the ASR backend (**Local** ↔ **API**) always triggers
  a restart. The menu click is automatic.
- Python 3.13.x only; 3.14 is unsupported because `mlx` does not yet
  ship a wheel for it.
- The bundled `.app` cannot transcode audio files
  (`silero_vad.read_audio("foo.wav")` and `torchaudio.load()` both
  raise `RuntimeError`). All audio in the bundle flows through
  PortAudio → `numpy.ndarray` → silero-vad → whisper — the only
  path we use. Dev-side rationale and how the stubs work around the
  upstream torch / torchaudio wheels lives in
  [`AGENTS.md`](../blob/main/AGENTS.md#6-build-pipeline).
