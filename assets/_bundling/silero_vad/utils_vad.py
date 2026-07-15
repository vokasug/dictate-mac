"""Numpy + onnxruntime re-implementation of silero-vad's runtime path.

Pure numpy throughout. We do not import ``torch`` or ``torchaudio`` —
those modules are not present in the bundled DictateMac.app (the
upstream wheel's top-level ``import torch`` / ``import torchaudio`` is
the entire reason this stub exists; see ``setup.py
``_install_silero_vad_stub``).

Public surface (re-)exported from ``silero_vad/__init__.py``:

* ``load_silero_vad`` — exposed by ``silero_vad.model``.
* ``get_speech_timestamps(audio, model, ...)`` — same algorithm as the
  upstream ``silero_vad`` wheel, transcribed verbatim from the
  Python implementation; numpy replaces every ``torch.*`` call with
  its numpy equivalent.
* ``OnnxWrapper`` — the model wrapper returned by ``load_silero_vad``.
* ``VADIterator`` — minimal streaming API for parity.
* ``collect_chunks``, ``drop_chunks`` — pure-numpy helpers used
  upstream by word-timestamp post-processing; provided here because
  the upstream ``__init__.py`` re-exports them.
* ``save_audio``, ``read_audio`` — stubbed to ``RuntimeError``; the
  daemon never asks silero-vad to load or save files (we always pass
  raw numpy arrays).
"""

from __future__ import annotations

import logging
import warnings
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger("dictate_mac.silero_vad_stub.utils_vad")

# silero-vad operates on 512-sample windows at 16 kHz, 256 at 8 kHz.
_WINDOW_16K = 512
_WINDOW_8K = 256

languages = ["ru", "en", "de", "es"]


# ---------------------------------------------------------------------------
# OnnxWrapper — matches the upstream class for ``get_speech_timestamps`` use.
# ---------------------------------------------------------------------------


class OnnxWrapper:
    """Stateful VAD model that runs through ``onnxruntime``.

    Usage matches the upstream OnnxWrapper::

        model = OnnxWrapper(path)
        prob = model(chunk, sampling_rate).item()

    ``chunk`` is ``np.ndarray`` of shape ``(batch, window_samples)``;
    the return value is a numpy scalar / ``(1, 1)`` ndarray whose
    ``.item()`` returns the speech probability for the whole chunk.
    """

    def __init__(self, path: str, force_onnx_cpu: bool = True):
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        if force_onnx_cpu and "CPUExecutionProvider" in onnxruntime.get_available_providers():
            providers = ["CPUExecutionProvider"]
        else:
            providers = onnxruntime.get_available_providers()

        self.session = onnxruntime.InferenceSession(
            path, providers=providers, sess_options=opts
        )
        self.reset_states()
        if "16k" in path:
            warnings.warn("This model supports only 16000 sampling rate!")
            self.sample_rates = [16000]
        else:
            self.sample_rates = [8000, 16000]

    def reset_states(self, batch_size: int = 1) -> None:
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context = np.zeros(0, dtype=np.float32)
        self._last_sr = 0
        self._last_batch_size = 0

    @staticmethod
    def _validate_input(x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if x.ndim > 2:
            raise ValueError(
                f"Too many dimensions for input audio chunk ({x.ndim})"
            )
        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[:, ::step]
            sr = 16000
        if sr not in (8000, 16000):
            raise ValueError(
                f"Supported sampling rates: 8000, 16000 (or multiples of 16000); got {sr}"
            )
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")
        return x, sr

    def __call__(self, x, sr: int) -> np.ndarray:
        x, sr = self._validate_input(
            np.asarray(x, dtype=np.float32), sr
        )
        num_samples = _WINDOW_16K if sr == 16000 else _WINDOW_8K
        batch_size = x.shape[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if self._last_sr and self._last_sr != sr:
            self.reset_states(batch_size)
        if self._last_batch_size and self._last_batch_size != batch_size:
            self.reset_states(batch_size)

        if self._context.size == 0:
            self._context = np.zeros((batch_size, context_size), dtype=np.float32)

        if x.shape[-1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {x.shape[-1]} "
                f"(Supported values: 256 for 8000 sample rate, "
                f"512 for 16000)"
            )

        x = np.concatenate([self._context, x], axis=1)
        if sr in (8000, 16000):
            ort_inputs = {
                "input": x,
                "state": self._state,
                "sr": np.array(sr, dtype="int64"),
            }
            out, state = self.session.run(None, ort_inputs)
            self._state = state
        else:
            raise ValueError(f"Unsupported sampling rate {sr}")

        self._context = x[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size
        # out has shape (batch, 1) — return as-is, callers use .item()
        # or .squeeze() depending on the path.
        return out

    def audio_forward(self, x: np.ndarray, sr: int) -> np.ndarray:
        """Run VAD on an entire utterance in fixed-size chunks.

        Used by silero-vad's higher-level helpers; not exercised by
        dictate-mac.
        """
        x, sr = self._validate_input(np.asarray(x, dtype=np.float32), sr)
        self.reset_states()
        num_samples = _WINDOW_16K if sr == 16000 else _WINDOW_8K
        if x.shape[1] % num_samples:
            pad_num = num_samples - (x.shape[1] % num_samples)
            x = np.pad(x, ((0, 0), (0, pad_num)), mode="constant")
        outs = []
        for i in range(0, x.shape[1], num_samples):
            outs.append(self(x[:, i : i + num_samples], sr))
        return np.concatenate(outs, axis=1)


# ---------------------------------------------------------------------------
# Validator — kept for parity but unused by the daemon.
# ---------------------------------------------------------------------------


class Validator:
    """Unused; retained so ``from silero_vad import Validator`` works."""

    def __init__(self, url: str, force_onnx_cpu: bool):
        self.onnx = url.endswith(".onnx")
        warnings.warn(
            "silero_vad.Validator is a stub in DictateMac.app", stacklevel=2
        )


# ---------------------------------------------------------------------------
# ``get_speech_timestamps`` — verbatim algorithm of upstream, numpy rewrite.
# ---------------------------------------------------------------------------


def _to_ndarray(audio) -> np.ndarray:
    if isinstance(audio, np.ndarray):
        return audio.astype(np.float32, copy=False)
    arr = np.asarray(audio)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def get_speech_timestamps(
    audio,
    model,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    return_seconds: bool = False,
    time_resolution: int = 1,
    visualize_probs: bool = False,
    progress_tracking_callback: Optional[Callable[[float], None]] = None,
    neg_threshold: Optional[float] = None,
    window_size_samples: int = 512,
    min_silence_at_max_speech: int = 98,
    use_max_poss_sil_at_max_speech: bool = True,
) -> List[dict]:
    """Mirror of upstream ``silero_vad.get_speech_timestamps``.

    Same parameter set and same return value (``list[{'start': N,
    'end': M}]`` in sample indices). Numpy replaces every ``torch.*``
    call from the upstream Python implementation; the algorithm is
    otherwise identical line-for-line.
    """
    audio = _to_ndarray(audio)

    # Squeeze empty outer dimensions (mirror upstream).
    while audio.ndim > 1 and audio.shape[0] == 1:
        audio = audio.squeeze(0)
    if audio.ndim > 1:
        raise ValueError(
            "More than one dimension in audio. Are you trying to "
            "process audio with 2 channels?"
        )

    # Downsample for sr > 16 kHz that is a multiple of 16 kHz.
    if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
        step = sampling_rate // 16000
        sampling_rate = 16000
        audio = audio[::step]
        warnings.warn(
            "Sampling rate is a multiply of 16000, casting to 16000 manually!",
            stacklevel=2,
        )
    else:
        step = 1

    if sampling_rate not in (8000, 16000):
        raise ValueError(
            "Currently silero VAD models support 8000 and 16000 (or "
            "multiply of 16000) sample rates"
        )

    window_size_samples = _WINDOW_16K if sampling_rate == 16000 else _WINDOW_8K
    model.reset_states()

    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = (
        sampling_rate * max_speech_duration_s
        - window_size_samples
        - 2 * speech_pad_samples
    )
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = (
        sampling_rate * min_silence_at_max_speech / 1000
    )

    audio_length_samples = int(audio.shape[0])

    speech_probs: list[float] = []
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample : current_start_sample + window_size_samples]
        if chunk.shape[0] < window_size_samples:
            pad = window_size_samples - chunk.shape[0]
            chunk = np.concatenate(
                [chunk, np.zeros(pad, dtype=np.float32)]
            )
        # Build the (1, window) input; ``model`` returns (1, 1).
        chunk_for_model = chunk[np.newaxis, :]
        speech_prob = float(model(chunk_for_model, sampling_rate).item())
        speech_probs.append(speech_prob)
        if progress_tracking_callback is not None:
            progress = current_start_sample + window_size_samples
            if progress > audio_length_samples:
                progress = audio_length_samples
            progress_tracking_callback(
                (progress / audio_length_samples) * 100.0
            )

    # ---- merge into speech chunks (verbatim algorithm) ----
    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)

    triggered = False
    speeches: list[dict] = []
    current_speech: dict = {}
    temp_end = 0
    prev_end = next_start = 0
    possible_ends: list[tuple[int, int]] = []

    for i, speech_prob in enumerate(speech_probs):
        cur_sample = window_size_samples * i

        if (speech_prob >= threshold) and temp_end:
            sil_dur = cur_sample - temp_end
            if sil_dur > min_silence_samples_at_max_speech:
                possible_ends.append((temp_end, sil_dur))
            temp_end = 0
            if next_start < prev_end:
                next_start = cur_sample

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech["start"] = cur_sample
            continue

        if triggered and (cur_sample - current_speech["start"] > max_speech_samples):
            if use_max_poss_sil_at_max_speech and possible_ends:
                prev_end, dur = max(possible_ends, key=lambda x: x[1])
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                next_start = prev_end + dur

                if next_start < prev_end + cur_sample:
                    current_speech["start"] = next_start
                else:
                    triggered = False
                prev_end = next_start = temp_end = 0
                possible_ends = []
            else:
                if prev_end:
                    current_speech["end"] = prev_end
                    speeches.append(current_speech)
                    current_speech = {}
                    if next_start < prev_end:
                        triggered = False
                    else:
                        current_speech["start"] = next_start
                    prev_end = next_start = temp_end = 0
                    possible_ends = []
                else:
                    current_speech["end"] = cur_sample
                    speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    possible_ends = []
                    continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = cur_sample
            sil_dur_now = cur_sample - temp_end

            if (
                not use_max_poss_sil_at_max_speech
                and sil_dur_now > min_silence_samples_at_max_speech
            ):
                prev_end = temp_end

            if sil_dur_now < min_silence_samples:
                continue
            else:
                current_speech["end"] = temp_end
                if (current_speech["end"] - current_speech["start"]) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                possible_ends = []
                continue

    if current_speech and (
        audio_length_samples - current_speech["start"]
    ) > min_speech_samples:
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence_duration // 2)
                )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - speech_pad_samples)
                )
        else:
            speech["end"] = int(
                min(audio_length_samples, speech["end"] + speech_pad_samples)
            )

    if return_seconds:
        audio_length_seconds = audio_length_samples / sampling_rate
        for speech_dict in speeches:
            speech_dict["start"] = max(
                round(speech_dict["start"] / sampling_rate, time_resolution), 0
            )
            speech_dict["end"] = min(
                round(speech_dict["end"] / sampling_rate, time_resolution),
                audio_length_seconds,
            )
    elif step > 1:
        for speech_dict in speeches:
            speech_dict["start"] *= step
            speech_dict["end"] *= step

    if visualize_probs:
        warnings.warn(
            "silero_vad.visualize is not implemented in the DictateMac stub",
            stacklevel=2,
        )

    return speeches


# ---------------------------------------------------------------------------
# Helpers / streaming API — parity-only.
# ---------------------------------------------------------------------------


def collect_chunks(tss: List[dict], audio: np.ndarray) -> np.ndarray:
    """Concatenate speech chunks by timestamp list (no torch)."""
    if not tss:
        return np.zeros(0, dtype=np.float32)
    chunks = [audio[int(s["start"]) : int(s["end"])] for s in tss]
    return np.concatenate(chunks).astype(np.float32, copy=False)


def drop_chunks(tss: List[dict], audio: np.ndarray) -> np.ndarray:
    """Inverse of ``collect_chunks`` (no torch)."""
    if not tss:
        return audio.astype(np.float32, copy=False)
    sorted_segments = sorted(
        ((int(s["start"]), int(s["end"])) for s in tss),
        key=lambda x: x[0],
    )
    pieces: list[np.ndarray] = []
    cursor = 0
    for start, end in sorted_segments:
        if cursor < start:
            pieces.append(audio[cursor:start])
        cursor = max(cursor, end)
    if cursor < audio.shape[0]:
        pieces.append(audio[cursor:])
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pieces).astype(np.float32, copy=False)


def read_audio(path: str, sampling_rate: int = 16000) -> np.ndarray:
    raise RuntimeError(
        "silero_vad.read_audio is stubbed in DictateMac.app — "
        "feed numpy.ndarray from PortAudio instead.",
    )


def save_audio(path: str, tensor, sampling_rate: int = 16000) -> None:
    raise RuntimeError(
        "silero_vad.save_audio is stubbed in DictateMac.app — "
        "use scipy.io.wavfile.write or soundfile instead.",
    )


class VADIterator:
    """Streaming VAD iterator. Preserved for parity; not used by the daemon."""

    def __init__(
        self,
        model,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ):
        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in (8000, 16000):
            raise ValueError(
                "VADIterator does not support sampling rates other than "
                "[8000, 16000]"
            )
        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self) -> None:
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, x, return_seconds: bool = False):
        window_size = _WINDOW_16K if self.sampling_rate == 16000 else _WINDOW_8K
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if x.shape[-1] != window_size:
            raise ValueError(
                f"chunk size {x.shape[-1]} != {window_size} for sr={self.sampling_rate}"
            )
        speech_prob = float(self.model(x, self.sampling_rate).item())
        self.current_sample += window_size

        if speech_prob >= self.threshold and self.temp_end:
            self.temp_end = 0

        if speech_prob >= self.threshold and not self.triggered:
            self.triggered = True
            start = max(0, self.current_sample - window_size - self.speech_pad_samples)
            return {"start": start} if return_seconds else {"start": int(start)}

        if speech_prob < (self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if (self.current_sample - self.temp_end) > self.min_silence_samples:
                self.triggered = False
                end = self.temp_end + self.speech_pad_samples
                self.temp_end = 0
                return {"end": end} if return_seconds else {"end": int(end)}

        return None


__all__ = [
    "OnnxWrapper",
    "Validator",
    "get_speech_timestamps",
    "collect_chunks",
    "drop_chunks",
    "read_audio",
    "save_audio",
    "VADIterator",
    "languages",
]
