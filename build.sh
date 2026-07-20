#!/usr/bin/env bash
# Build DictateMac.app.
#
# Pipeline:
#   1. Make sure py2app is installed in the active venv.
#   2. Generate the icon (assets/iconset/*) as a waveform glyph.
#   3. Run setup.py py2app — produces dist/DictateMac.app. setup.py
#      then runs the full post-build pipeline:
#        - pre-patch the source venv with the assets/_bundling stubs
#          (restored afterwards even on failure)
#        - extract native libs (libportaudio, _sounddevice_data) out
#          of python313.zip
#        - install torchaudio / silero_vad / mlx_whisper.timing stubs
#          into the bundle
#        - strip unused packages, .pyi stubs, __pycache__, metadata
#        - rewrite __boot__.py to use RESOURCEPATH
#        - rewrite PythonExecutable in Info.plist to @executable_path
#          (removes the developer's venv path from the shipped bundle)
#        - re-sign the bundle ad-hoc with codesign
#      See AGENTS.md §6 for the full description.
#
# Usage:
#   ./build.sh           # full rebuild
#   ./build.sh --clean   # also wipe build/ and dist/ before building
set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-./.venv/bin/python}"
VENV_PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"

echo "== build.sh: using python at $VENV_PY"
"$VENV_PY" --version

if [ "${1:-}" = "--clean" ]; then
    echo "== build.sh: --clean — removing build/ and dist/"
    rm -rf build dist
fi

# 1) py2app in venv.
echo "== build.sh: ensuring py2app is installed"
if ! "$VENV_PY" -c "import py2app" >/dev/null 2>&1; then
    echo "   installing py2app via uv ..."
    # The venv was created by `uv` which leaves pip out. We can't reach
    # `python -m pip`. Detect uv and let it install; otherwise fall
    # back to ensurepip + pip.
    if command -v uv >/dev/null 2>&1; then
        UV_TORCH_BACKEND=cpu uv pip install --python "$VENV_PY" 'py2app>=0.28' >/dev/null
    else
        "$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1 \
            && "$VENV_PY" -m pip install 'py2app>=0.28'
    fi
fi

# 2) .icns via assets/icon/make_icon.py.
echo "== build.sh: (re)generating assets/DictateMac.icns"
"$VENV_PY" assets/icon/make_icon.py

# 3) Build.
echo "== build.sh: running py2app.build_app"
rm -rf build dist
"$VENV_PY" setup.py py2app

APP="dist/DictateMac.app"
if [ ! -d "$APP" ]; then
    echo "== build.sh: ERROR — $APP not produced" >&2
    exit 1
fi

echo
echo "Built $APP"
echo "Run with: open $APP"
echo "Logs at (when launched from Finder): ~/Library/Logs/dictate-mac/dictate-mac.log"
