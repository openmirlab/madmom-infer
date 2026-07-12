"""Optional torch backend -- Phase 3a: a differentiable spectrogram frontend.

This subpackage is the opt-in torch backend the numpy-backend docstrings
elsewhere in this repo anticipate (`madmom_infer/audio/signal.py`'s header
mentions a future `madmom_infer.torch.audio.*`). It is NOT imported by
`madmom_infer/__init__.py` -- `import madmom_infer` never touches this
module, so the core install has zero torch dependency. Importing
`madmom_infer.torch` (this module) itself IS the opt-in gate: it requires
`torch` to be installed and raises a clear `ImportError` with an install
hint otherwise, rather than an opaque `ModuleNotFoundError` deep inside a
submodule.

Scope (Phase 3a, see CLAUDE.md roadmap): a batched, differentiable,
device-agnostic reimplementation of the DSP feature-extraction chain
`madmom_infer.audio.{signal,stft,filters,spectrogram}` compose --
framing, STFT, filterbank application, log compression, and the temporal
difference feature `RNNDownBeatProcessor` stacks on top -- as torch
tensor ops with autograd support. It reuses the numpy side's *parameters*
(window arrays, filterbank matrices, bin frequencies, diff-frame counts)
computed via the existing numpy code and converted to tensors -- this
package does not reimplement filterbank/window construction, only the
tensor operations that need to be differentiable
(`madmom_infer/torch/audio/frontend.py`).

Explicitly NOT in scope here (see `madmom_infer/torch/audio/frontend.py`'s
module docstring for the full reasoning): Viterbi/DBN decoding (sequential,
discrete-state, no autograd/batching benefit) and the NN forward pass
(madmom's LSTMs use peephole connections `torch.nn.LSTM` does not support,
so a torch NN backend needs a custom cell -- left for a possible Phase 3b).
There is also no `madmom_infer.torch.audio.signal.Signal` counterpart to
the numpy `Signal` class: this frontend takes an already-mono, already
sample-rate-matched float waveform tensor directly, sidestepping file
loading/downmixing (out of scope for a differentiable-frontend package).

Reads: torch (guarded); read by: nothing in the numpy backend (one-way,
opt-in dependency only).
"""

try:
    import torch as _torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "madmom_infer.torch requires the optional 'torch' dependency, which "
        "is not installed. Install it with:\n\n"
        '    pip install "madmom-infer[torch]"\n\n'
        "or `pip install torch` directly (torch>=2.0.0). The numpy backend "
        "(`import madmom_infer`) does not need torch and is unaffected."
    ) from exc

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
