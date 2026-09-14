"""Optional torch backend -- a differentiable spectrogram frontend (Phase
3a) plus an additional NN-forward-pass backend (added afterward, see
`madmom_infer.torch.ml.nn`).

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

There is also no `madmom_infer.torch.audio.signal.Signal` counterpart to
the numpy `Signal` class: this frontend takes an already-mono, already
sample-rate-matched float waveform tensor directly, sidestepping file
loading/downmixing (out of scope for a differentiable-frontend package).

`madmom_infer.torch.ml.nn` (new subpackage) is the NN forward pass itself:
differentiable, GPU-capable `torch.nn.Module` twins of every class in
`madmom_infer.ml.nn.layers` (including a custom LSTM/GRU cell loop, since
madmom's peephole-connected LSTM has no `torch.nn.LSTM` equivalent),
built from an already-loaded numpy `NeuralNetwork`/`NeuralNetworkEnsemble`/
processor-graph via `to_torch` (re-exported here). This is purely
additive -- the numpy `madmom_infer.ml.nn` classes remain the reference
implementation and are never modified by this package. Still out of
scope: Viterbi/DBN decoding in torch (sequential, discrete-state -- no
autograd/batching benefit expected there, ever) and a fully wired
frontend-to-NN torch processor (frontend and NN conversion exist
independently for now; composing them into one end-to-end torch pipeline
is a later step).

Reads: torch (guarded), madmom_infer.torch.ml.nn (to_torch); read by:
nothing in the numpy backend (one-way, opt-in dependency only).
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
from madmom_infer.torch.ml.nn import to_torch

__all__ = [
    "SpectrogramFrontend",
    "apply_filterbank",
    "frame_signal",
    "log_compress",
    "rnn_downbeat_frontend",
    "stft",
    "temporal_difference",
    "to_torch",
]
