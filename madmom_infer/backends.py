"""Backend dispatch -- the one module that owns the numpy-vs-torch decision
for madmom_infer's NN-backed processors, instead of it leaking into every
processor's own `__init__` (`RNNDownBeatProcessor`, `RNNBarProcessor`,
`RNNBeatProcessor`, `RNNOnsetProcessor`, `CNNOnsetProcessor`,
`CNNKeyRecognitionProcessor`, `DeepChromaProcessor`,
`CNNChordFeatureProcessor`, `RNNPianoNoteProcessor`, `CNNPianoNoteProcessor`).

This module itself is torch-free at import time -- `import madmom_infer`
must never import torch (see `madmom_infer/torch/__init__.py`'s own
header). `torch_pipeline_processor` only imports `madmom_infer.torch`
lazily, inside the function body, so a caller who actually asks for
`backend="torch"` without the optional `torch` extra installed gets that
package's own guarded `ImportError` (with its install hint) at the point
of use, not an opaque failure at `import madmom_infer` time.

Reads: (lazily) madmom_infer.torch.features (build_pipeline,
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
    """Raise `ValueError` for torch devices outside this backend's scope."""
    if isinstance(device, str) and (
            device == "mps" or device.startswith("mps:")):
        raise ValueError(
            "device='mps' is not supported by madmom_infer's torch backend; "
            "use device=None, 'cpu', or a CUDA device")


def torch_pipeline_processor(name, device=None, **pipeline_kwargs):
    """Build a `madmom_infer.torch.features` pipeline by `name`
    (`build_pipeline`'s registry), move it to `device` (if given), and wrap
    it in a `TorchPipelineProcessor` so it can be dropped in wherever a
    numpy processor stage is expected."""
    validate_torch_device(device)
    from madmom_infer.torch.features import TorchPipelineProcessor, build_pipeline

    module = build_pipeline(name, **pipeline_kwargs)
    if device is not None:
        module = module.to(device)
    return TorchPipelineProcessor(module, device=device)
