"""Torch-backend counterpart of `madmom_infer/audio/` -- see `frontend.py`.

Only `frontend.py` exists at this phase (Phase 3a: spectrogram frontend).
There is no torch `signal.py`/`stft.py`/`filters.py`/`spectrogram.py` split
mirroring the numpy layout 1:1 -- the differentiable frontend composes all
of those stages into one module because the torch versions have no
independent API surface of their own (no `Signal`/`FramedSignal`-equivalent
classes are needed; see `frontend.py`'s and the parent package's docstrings
for why).

Reads: madmom_infer/torch/audio/frontend.py
"""

from madmom_infer.torch.audio.frontend import (
    SpectrogramFrontend,
    apply_filterbank,
    frame_signal,
    log_compress,
    rnn_downbeat_frontend,
    stft,
    temporal_difference,
)

__all__ = [
    "SpectrogramFrontend",
    "apply_filterbank",
    "frame_signal",
    "log_compress",
    "rnn_downbeat_frontend",
    "stft",
    "temporal_difference",
]
