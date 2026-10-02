"""Backend dispatch -- the one module that owns the numpy-vs-torch decision
for madmom_infer's NN-backed processors, instead of it leaking into every
processor's own `__init__` (`RNNDownBeatProcessor`, `RNNBarProcessor`,
`RNNBeatProcessor`, `RNNOnsetProcessor`, `CNNOnsetProcessor`,
`CNNKeyRecognitionProcessor`, `DeepChromaProcessor`,
`CNNChordFeatureProcessor`, `RNNPianoNoteProcessor`, `CNNPianoNoteProcessor`).

This module owns explicit CPU/CUDA device resolution for all three entry
paths: shared pipelines, RNNBarProcessor, and direct adapters. None retains
existing placement; explicit auto chooses CUDA when available, otherwise CPU.
This module itself is torch-free at import time -- `import madmom_infer`
must never import torch (see `madmom_infer/torch/__init__.py`'s own
header). Resolution imports `madmom_infer.torch` lazily, so a caller asking for
`backend="torch"` without the optional `torch` extra installed gets that
package's own guarded `ImportError` (with its install hint) at the point
of use, not an opaque failure at `import madmom_infer` time.

Reads: (lazily) madmom_infer.torch (guarded torch import and CUDA availability),
madmom_infer.torch.features (build_pipeline,
TorchPipelineProcessor); read by: madmom_infer/features/{downbeats,beats,
onsets,chords,notes}.py, madmom_infer/audio/chroma.py, madmom_infer/features/
key.py, madmom_infer/api.py.
"""

BACKENDS = ("numpy", "torch")


def validate_backend(backend):
    """Raise `ValueError` unless `backend` is one of `BACKENDS`."""
    if backend not in BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}, expected one of {BACKENDS}")


def validate_torch_device(device):
    """Keep the importable v0.3.0 validator, delegating to the single owner."""
    resolve_torch_device(device)


def resolve_torch_device(device):
    """Return a validated torch device, preserving None's existing placement.

    Only explicit ``auto`` selects a default. Explicit CPU/CUDA choices are
    honored or rejected before checkpoint lookup; MPS and other accelerators
    are outside this backend's scope. Import through the optional backend
    gate to preserve its install hint without making core imports need torch.
    """
    if device is None:
        return None
    from .torch import _torch as torch

    if isinstance(device, str) and device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid torch device {device!r}; use 'cpu', 'cuda', 'cuda:N', or 'auto'") from exc
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError(
            f"device={str(resolved)!r} is not supported by madmom_infer's torch backend; "
            "use device=None, 'auto', 'cpu', or a CUDA device")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA device {str(resolved)!r} requested but CUDA is unavailable")
        count = torch.cuda.device_count()
        index = torch.cuda.current_device() if resolved.index is None else resolved.index
        if index < 0 or index >= count:
            raise ValueError(f"CUDA device index {index!r} is unavailable; found {count} devices")
        # Freeze the model's device: a later caller set_device() must not
        # redirect input tensors to a different GPU than the loaded module.
        resolved = torch.device("cuda", index)
    return resolved


def torch_pipeline_processor(name, device=None, **pipeline_kwargs):
    """Build a `madmom_infer.torch.features` pipeline by `name`
    (`build_pipeline`'s registry), resolve `device` before construction, and wrap
    it in a `TorchPipelineProcessor` so it can be dropped in wherever a
    numpy processor stage is expected."""
    device = resolve_torch_device(device)
    from madmom_infer.torch.features import TorchPipelineProcessor, build_pipeline

    module = build_pipeline(name, **pipeline_kwargs)
    return TorchPipelineProcessor(module, device=device)
