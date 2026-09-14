"""Torch NN-forward-pass backend -- barrel for `madmom_infer.torch.ml.nn`.

Re-exports `to_torch`/`ensemble_to_torch` (`convert.py`, the conversion
entry points) and the individual torch `nn.Module` layer classes
(`layers.py`, the differentiable, GPU-capable twins of
`madmom_infer.ml.nn.layers`'s numpy classes) so callers can do
`from madmom_infer.torch.ml.nn import to_torch` without reaching into the
submodules directly.

Reads: madmom_infer/torch/ml/nn/{layers,convert}.py; read by:
madmom_infer/torch/__init__.py.
"""

from .convert import ensemble_to_torch, to_torch
from .layers import (
    AverageLayer,
    BatchNormLayer,
    BidirectionalLayer,
    ConvolutionalLayer,
    DstackModule,
    EnsembleModule,
    FeedForwardLayer,
    GRULayer,
    LSTMLayer,
    MaxPoolLayer,
    PadLayer,
    ParallelGraphModule,
    RecurrentLayer,
    ReshapeLayer,
    SequentialGraphModule,
    StrideLayer,
    TransposeLayer,
)

__all__ = [
    "to_torch",
    "ensemble_to_torch",
    "FeedForwardLayer",
    "RecurrentLayer",
    "BidirectionalLayer",
    "LSTMLayer",
    "GRULayer",
    "ConvolutionalLayer",
    "MaxPoolLayer",
    "BatchNormLayer",
    "AverageLayer",
    "PadLayer",
    "StrideLayer",
    "TransposeLayer",
    "ReshapeLayer",
    "SequentialGraphModule",
    "ParallelGraphModule",
    "DstackModule",
    "EnsembleModule",
]
