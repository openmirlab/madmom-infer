"""Barrel for `madmom_infer.torch.features`: the 9 differentiable
audio-in-activations-out pipelines (`pipelines.py`) and the numpy-facing
`TorchPipelineProcessor` adapter (`adapter.py`).

Reads: madmom_infer/torch/features/{pipelines,adapter}.py; read by:
madmom_infer/torch/__init__.py.
"""

from .adapter import TorchPipelineProcessor
from .pipelines import (
    BeatsPipeline,
    ChordFeaturePipeline,
    ChromaPipeline,
    DownbeatsPipeline,
    KeyPipeline,
    NoteCNNPipeline,
    NoteRNNPipeline,
    OnsetCNNPipeline,
    OnsetRNNPipeline,
    build_pipeline,
)

__all__ = [
    "TorchPipelineProcessor",
    "build_pipeline",
    "DownbeatsPipeline",
    "BeatsPipeline",
    "OnsetRNNPipeline",
    "OnsetCNNPipeline",
    "KeyPipeline",
    "ChromaPipeline",
    "ChordFeaturePipeline",
    "NoteRNNPipeline",
    "NoteCNNPipeline",
]
