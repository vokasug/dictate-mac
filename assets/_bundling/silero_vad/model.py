"""``load_silero_vad`` shim — selects the ONNX path only.

The upstream wheel defaults to ``onnx=False`` which loads the JIT model
via ``torch.jit.load``. We never call it that way: bundle-side we
**only** support ONNX (loaded through ``onnxruntime``). The function
defaults to ``onnx=True`` here because there is no other path.
"""

from __future__ import annotations

import importlib.resources as impresources
import logging
import os
import warnings

logger = logging.getLogger("dictate_mac.silero_vad_stub.model")


def load_silero_vad(onnx: bool = True, opset_version: int = 16):
    """Return a numpy/onnxruntime-backed silero-VAD model.

    Parameters mirror the upstream signature for parity. We force
    ``onnx=True`` (the JIT path requires torch, which the bundle does
    not ship). If the caller passes ``onnx=False``, we fall back to ONNX
    anyway and emit a warning — the JIT model would not load under the
    bundle.
    """
    if not onnx:
        warnings.warn(
            "silero_vad JIT path is not supported in DictateMac.app — "
            "falling back to ONNX (torch is not bundled).",
            stacklevel=2,
        )

    if opset_version != 16:
        # All other variants are stripped from the bundle (see setup.py
        # _strip_bundle_junk) so requesting opset != 16 would fail. We
        # silently coerce to the one variant we ship.
        opset_version = 16

    model_name = "silero_vad.onnx"
    package_path = "silero_vad.data"

    try:
        # Python 3.12+ stdlib importlib.resources path API.
        model_file_path = str(
            impresources.files(package_path).joinpath(model_name)
        )
    except Exception:  # pragma: no cover — defensive
        model_file_path = str(
            impresources.files(package_path).joinpath(model_name)
        )

    if not os.path.exists(model_file_path):  # pragma: no cover
        raise FileNotFoundError(
            f"silero_vad ONNX model not found at {model_file_path!r}; "
            "the bundle may have been built without `silero_vad.data` "
            "listed in `packages`."
        )

    logger.info("loading silero-vad ONNX model from %s", model_file_path)
    from .utils_vad import OnnxWrapper

    return OnnxWrapper(model_file_path, force_onnx_cpu=True)
