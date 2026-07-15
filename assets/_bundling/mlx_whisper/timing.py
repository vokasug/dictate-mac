"""mlx_whisper.timing stub for the bundled DictateMac.app.

mlx_whisper.transcribe has ``from .timing import add_word_timestamps``
at its top level. The real ``timing.py`` imports ``numba`` and
``from scipy import signal`` at its own top level, which transitively
drags libllvmlite (129 MB), full scipy (78 MB), and the matching
``__pycache__`` plus dist-info into the bundle.

We never call ``add_word_timestamps`` — dictate-mac passes
``word_timestamps=False`` and only consumes the ``"text"`` field of
the transcribe-result dict. Replacing the body of ``add_word_timestamps``
with a stub that always raises keeps the import side effect-free while
breaking the dependency chain.
"""

from __future__ import annotations


def add_word_timestamps(*args, **kwargs):  # noqa: ANN001,ANN201
    """Stub for ``mlx_whisper.timing.add_word_timestamps``.

    Raises unconditionally — the daemon never enables ``word_timestamps``.
    If a future feature requires word timestamps, replace this stub
    with a downstream port of ``add_word_timestamps`` (the upstream
    implementation requires numba + scipy.signal which the bundle
    intentionally does not ship).
    """
    raise RuntimeError(
        "add_word_timestamps is not supported in DictateMac.app — "
        "use the venv-installed build for word-level timestamps."
    )
