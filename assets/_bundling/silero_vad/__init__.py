"""silero_vad package stub for the bundled DictateMac.app.

This replaces the upstream ``silero_vad`` runtime module path with a
pure-numpy / onnxruntime implementation. We do this so the bundle does
not have to ship ``torch`` (~429 MB on disk + the matching ~263 MB
``libtorch_cpu.dylib``), nor ``torchaudio``.

The public API mirrors the upstream wheel — only the call sites that
``dictate_mac.audio`` actually exercises are tested. Specifically:

* ``silero_vad.load_silero_vad`` — returns an OnnxWrapper-shaped model.
* ``silero_vad.get_speech_timestamps`` — same signature and same
  algorithm as upstream; uses numpy throughout.
* ``silero_vad.VADIterator`` — implemented; not currently used by the
  daemon but exposed for parity.
* ``silero_vad.save_audio``, ``silero_vad.read_audio`` — stubbed to
  raise; we feed silero-vad raw numpy arrays from PortAudio.
* ``silero_vad.collect_chunks``, ``drop_chunks`` — re-implemented in
  pure numpy (no torch).
"""

from .utils_vad import (  # noqa: F401
    get_speech_timestamps,
    save_audio,
    read_audio,
    VADIterator,
    collect_chunks,
    drop_chunks,
)
from .model import load_silero_vad  # noqa: F401
