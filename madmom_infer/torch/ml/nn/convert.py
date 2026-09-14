"""`to_torch`: converts an already-loaded numpy `madmom_infer.ml.nn` object
(a `NeuralNetwork`, a `NeuralNetworkEnsemble`, or a raw
`madmom_infer.processors.SequentialProcessor`/`ParallelProcessor` graph of
layers, as `notes_cnn.pkl` unpickles to -- see CLAUDE.md's Wave 4e) into an
equivalent `torch.nn.Module` from `madmom_infer.torch.ml.nn.layers`.

This module owns two decisions the pure per-layer math in `layers.py`
deliberately doesn't make:

1. **Weight extraction + dtype/flip/axis-shift conversion.** Every numpy
   layer's constructor-time arrays (`weights`, `bias`, `recurrent_weights`,
   `peephole_weights`, `beta`/`gamma`/`mean`/`inv_std`, `init`/`cell_init`)
   are copied into float32 torch tensors here, once, at conversion time --
   never at forward time. `ConvolutionalLayer.weights` (`(C_in, F_out, kt,
   kf)`, true-convolution kernel) is additionally permuted to `(F_out,
   C_in, kt, kf)` and flipped on its last two axes here, so
   `layers.ConvolutionalLayer` can hand the result straight to
   `F.conv2d` (a cross-correlation) with no per-call flip cost.
   `PadLayer.axes`/`AverageLayer.axis`/`TransposeLayer.axes` are shifted
   by the "and there's now a leading batch dimension" +1 offset here too
   (see `layers.py`'s module header for the batch convention this
   assumes) -- every module in `layers.py` receives pre-shifted axes, it
   never shifts them itself.
2. **`trainable`**: forwarded to every `layers.py` module constructor,
   controlling whether copied weights become `nn.Parameter`s (gradients
   flow into them, e.g. for fine-tuning) or plain buffers (the default --
   frozen, inference-only, but the INPUT still gets gradients since every
   op in `layers.py` is autograd-differentiable regardless).
3. **`fast_recurrent`**: forwarded only to eligible stacked LSTMs. It is an
   opt-in request for `triton_lstm.py`'s CUDA float32 no-grad gate fusion;
   unsupported and grad-enabled calls retain the eager implementation.

`to_torch` also accepts a plain Python list of `NeuralNetwork` instances
(an ensemble not already wrapped in a `NeuralNetworkEnsemble`) --
equivalent to `ensemble_to_torch`, provided both because either reads
naturally depending on what the caller already has in hand.

**Performance path (2026-09-14): ensemble/gate-stacked layers.** A
network (or, especially, an ensemble of E structurally-identical
networks -- `downbeats_blstm`/`beats_blstm`/`beats_lstm`/`onsets_brnn`/
`onsets_rnn`/`notes_brnn`, all pure `FeedForwardLayer`/`RecurrentLayer`/
`LSTMLayer`/`GRULayer`/`BidirectionalLayer` stacks, no CNN layers) is, by
default, converted to `.stacked.py`'s fused/ensemble-stacked modules
instead of `.layers.py`'s straightforward per-network loop -- see
`.stack_convert.py`'s `_try_stack_networks`/`_stack_layer_position` (the
eligibility check + module builder this file calls into) and
`stacked.py`'s own module header for why (measured torch-vs-numpy slowdown on the
recurrent path, `tools/bench_torch_backend.py`). Eligibility is decided
purely by walking each network's `.layers` list and checking every layer
against `_layer_signature` (returns `None` for any layer type that isn't
one of the 5 stackable ones -- covers every CNN layer, so
`onsets_cnn`/`chords_cnn_feat`/`key_cnn`/`notes_cnn` transparently keep
using `.layers.py`'s `EnsembleModule`/`NeuralNetworkModule` path
unchanged, which the implementation report found were already fast).
`_try_stack_networks` returns `None` (triggering that same fallback) if
any ensemble member's architecture doesn't match, so this is purely an
optimization -- never a behavior change a caller needs to opt into.

Reads: torch, madmom_infer.ml.nn (NeuralNetwork, NeuralNetworkEnsemble,
average_predictions), madmom_infer.ml.nn.layers (the numpy layer classes,
as isinstance targets only -- never imported for their behavior),
madmom_infer.processors (SequentialProcessor, ParallelProcessor),
madmom_infer.torch.ml.nn.layers (the torch modules being constructed for
the non-stackable/fallback path), madmom_infer.torch.ml.nn.stack_convert
(`_convert_activation`/`_tensor`/`_expected_unbatched_ndim` shared
helpers, plus `_try_stack_networks` -- the ensemble/gate-fused
performance path's entry point, see the "Performance path" section
above); read by: madmom_infer/torch/__init__.py (re-exports `to_torch`).
"""

from __future__ import annotations

import numpy as np
import torch

from madmom_infer.ml.nn import NeuralNetwork, NeuralNetworkEnsemble
from madmom_infer.ml.nn import layers as _np_layers
from madmom_infer.processors import ParallelProcessor, SequentialProcessor

from . import layers as _torch_layers
from .stack_convert import _convert_activation, _expected_unbatched_ndim, _tensor, _try_stack_networks

# -- per-layer-class conversion ------------------------------------------


def _convert_feedforward(layer, trainable):
    return _torch_layers.FeedForwardLayer(
        _tensor(layer.weights),
        _tensor(layer.bias),
        activation=_convert_activation(layer.activation_fn),
        trainable=trainable,
    )


def _convert_recurrent(layer, trainable):
    return _torch_layers.RecurrentLayer(
        _tensor(layer.weights),
        _tensor(layer.bias),
        _tensor(layer.recurrent_weights),
        _tensor(layer.init),
        activation=_convert_activation(layer.activation_fn),
        trainable=trainable,
    )


def _convert_bidirectional(layer, trainable):
    return _torch_layers.BidirectionalLayer(
        _convert_layer(layer.fwd_layer, trainable),
        _convert_layer(layer.bwd_layer, trainable),
    )


def _convert_gate(gate, trainable):
    # GRUCell (a Cell subclass, itself a Gate subclass) has no
    # `peephole_weights` attribute at all -- only Gate/Cell instances used
    # inside an LSTMLayer carry that (possibly-None) attribute.
    peephole = getattr(gate, "peephole_weights", None)
    return _torch_layers._GateParams(
        _tensor(gate.weights),
        _tensor(gate.bias),
        _tensor(gate.recurrent_weights),
        None if peephole is None else _tensor(peephole),
        activation=_convert_activation(gate.activation_fn),
        trainable=trainable,
    )


def _convert_lstm(layer, trainable):
    return _torch_layers.LSTMLayer(
        _convert_gate(layer.input_gate, trainable),
        _convert_gate(layer.forget_gate, trainable),
        _convert_gate(layer.cell, trainable),
        _convert_gate(layer.output_gate, trainable),
        _tensor(layer.init),
        _tensor(layer.cell_init),
        activation=_convert_activation(layer.activation_fn),
        trainable=trainable,
    )


def _convert_gru(layer, trainable):
    return _torch_layers.GRULayer(
        _convert_gate(layer.reset_gate, trainable),
        _convert_gate(layer.update_gate, trainable),
        _convert_gate(layer.cell, trainable),
        _tensor(layer.init),
        trainable=trainable,
    )


def _convert_conv(layer, trainable):
    # numpy weights: (C_in, F_out, kt, kf), true-convolution kernel.
    # -> (F_out, C_in, kt, kf) for conv2d, flipped for cross-correlation
    # == true-convolution equivalence (see layers.py's ConvolutionalLayer
    # docstring).
    w = _tensor(layer.weights).permute(1, 0, 2, 3)
    w = torch.flip(w, dims=[-2, -1]).contiguous()
    stride = layer.stride
    if stride is not None:
        if isinstance(stride, (int, np.integer)):
            stride = (int(stride), int(stride))
        else:
            stride = tuple(int(s) for s in stride)
    return _torch_layers.ConvolutionalLayer(
        w,
        _tensor(layer.bias),
        stride=stride,
        pad=layer.pad,
        activation=_convert_activation(layer.activation_fn),
        trainable=trainable,
    )


def _shift_axes(axes, ndim=None):
    """Shift 0-based numpy-layer axis indices by +1 to account for the
    batch dimension every torch module in `layers.py` assumes is already
    present at axis 0 (see that module's header)."""
    if axes is None:
        return None
    if isinstance(axes, int):
        return axes + 1
    return tuple(a + 1 for a in axes)


def _convert_maxpool(layer):
    axis = _shift_axes(layer.axis)
    return _torch_layers.MaxPoolLayer(size=layer.size, stride=layer.stride,
                                       axis=axis)


def _convert_batchnorm(layer, trainable):
    return _torch_layers.BatchNormLayer(
        _tensor(layer.beta),
        _tensor(layer.gamma),
        _tensor(layer.mean),
        _tensor(layer.inv_std),
        activation=_convert_activation(layer.activation_fn),
        trainable=trainable,
    )


def _convert_average(layer):
    return _torch_layers.AverageLayer(axis=_shift_axes(layer.axis),
                                       keepdims=layer.keepdims)


def _convert_pad(layer):
    return _torch_layers.PadLayer(layer.width, _shift_axes(layer.axes),
                                   value=layer.value)


def _convert_stride(layer):
    return _torch_layers.StrideLayer(layer.block_size)


def _convert_transpose(layer):
    axes = layer.axes
    if axes is None:
        # np.transpose(data, None) reverses all axes; the batched tensor
        # has one extra leading (batch) axis that must stay first.
        return _torch_layers.TransposeLayer(None)
    new_axes = (0,) + tuple(a + 1 for a in axes)
    return _torch_layers.TransposeLayer(new_axes)


def _convert_reshape(layer):
    # newshape's entries index sizes, not axes -- no batch-dim shift
    # needed, `layers.ReshapeLayer` itself prepends the real batch size.
    if layer.order != "C":
        raise NotImplementedError(
            "to_torch: ReshapeLayer with order != 'C' is not supported "
            f"(got order={layer.order!r})."
        )
    return _torch_layers.ReshapeLayer(tuple(layer.newshape))


_LAYER_CONVERTERS = {
    _np_layers.LSTMLayer: _convert_lstm,
    _np_layers.GRULayer: _convert_gru,
    _np_layers.BidirectionalLayer: _convert_bidirectional,
    _np_layers.RecurrentLayer: _convert_recurrent,
    _np_layers.ConvolutionalLayer: _convert_conv,
    _np_layers.FeedForwardLayer: _convert_feedforward,
    _np_layers.BatchNormLayer: _convert_batchnorm,
}

_LAYER_CONVERTERS_NO_TRAINABLE = {
    _np_layers.MaxPoolLayer: _convert_maxpool,
    _np_layers.AverageLayer: _convert_average,
    _np_layers.PadLayer: _convert_pad,
    _np_layers.StrideLayer: _convert_stride,
    _np_layers.TransposeLayer: _convert_transpose,
    _np_layers.ReshapeLayer: _convert_reshape,
}


def _convert_layer(layer, trainable):
    """Convert a single `madmom_infer.ml.nn.layers.Layer` instance.

    Order matters: subclasses (`LSTMLayer`/`GRULayer`/`BidirectionalLayer`/
    `RecurrentLayer`/`ConvolutionalLayer`) are checked before their common
    base classes (`RecurrentLayer` before nothing else needs it since
    `Gate`/`Cell` are never converted standalone -- see `_convert_gate`,
    called directly by `_convert_lstm`/`_convert_gru`, not through this
    dispatcher) and `ConvolutionalLayer` before `FeedForwardLayer` (its
    base class).
    """
    for cls, fn in _LAYER_CONVERTERS.items():
        if isinstance(layer, cls):
            return fn(layer, trainable)
    for cls, fn in _LAYER_CONVERTERS_NO_TRAINABLE.items():
        if isinstance(layer, cls):
            return fn(layer)
    raise TypeError(
        f"to_torch: don't know how to convert layer of type "
        f"{type(layer).__name__!r}."
    )


# -- graph nodes (SequentialProcessor / ParallelProcessor / numpy.dstack) -


def _is_dstack(node):
    return node is np.dstack or getattr(node, "__name__", None) == "dstack"


def _convert_graph_node(node, trainable):
    if isinstance(node, ParallelProcessor):
        return _torch_layers.ParallelGraphModule(
            [_convert_graph_node(p, trainable) for p in node.processors]
        )
    if isinstance(node, SequentialProcessor):
        return _torch_layers.SequentialGraphModule(
            [_convert_graph_node(p, trainable) for p in node.processors]
        )
    if isinstance(node, _np_layers.Layer):
        return _convert_layer(node, trainable)
    if _is_dstack(node):
        return _torch_layers.DstackModule()
    raise TypeError(
        f"to_torch: don't know how to convert processor-graph node of "
        f"type {type(node).__name__!r}."
    )


# `_expected_unbatched_ndim` (used below by `_convert_single`/`to_torch`)
# lives in `stack_convert.py` now -- imported at the top of this module,
# alongside `_convert_activation`/`_tensor`/`_try_stack_networks`.


# -- public API ------------------------------------------------------------


def _convert_single(obj, trainable, fast_recurrent=False):
    """Convert one ensemble member: either a `NeuralNetwork`, or (see
    `notes_cnn.pkl`, CLAUDE.md's Wave 4e) a raw `SequentialProcessor`/
    `ParallelProcessor` graph -- `NeuralNetworkEnsemble.load` wraps
    EITHER shape the same way (a size-N `ParallelProcessor` +
    `average_predictions`), so an ensemble's members are not guaranteed
    to all be bare `NeuralNetwork` instances."""
    if isinstance(obj, NeuralNetwork):
        stacked = _try_stack_networks(
            [obj], trainable, fast_recurrent=fast_recurrent
        )
        if stacked is not None:
            return stacked
        layer_modules = [_convert_layer(layer, trainable) for layer in obj.layers]
        ndim = _expected_unbatched_ndim(obj.layers)
        return NeuralNetworkModule(layer_modules, expected_unbatched_ndim=ndim)
    if isinstance(obj, (SequentialProcessor, ParallelProcessor)):
        graph_module = _convert_graph_node(obj, trainable)
        ndim = _expected_unbatched_ndim([obj])
        return ProcessorGraphModule(graph_module, expected_unbatched_ndim=ndim)
    raise TypeError(
        f"to_torch: don't know how to convert ensemble member of type "
        f"{type(obj).__name__!r}."
    )


def ensemble_to_torch(networks, trainable=False, fast_recurrent=False):
    """Convert a list of `madmom_infer.ml.nn.NeuralNetwork` (or raw
    processor-graph) instances into a single averaging-ensemble
    `torch.nn.Module` (mirrors `madmom_infer.ml.nn.average_predictions`
    for a length->=1 list of equal-shaped predictions -- no special-casing
    for 0-dimensional per-network outputs, since every target model
    family in this project always predicts at least one real
    time/feature axis).

    Tries the ensemble/gate-stacked performance path first (see this
    module's header) when every member is a plain `NeuralNetwork` --
    falls back to the original per-network `EnsembleModule` loop
    otherwise (mixed-type members, e.g. `notes_cnn`'s raw processor
    graphs, or a non-stackable/mismatched architecture)."""
    if all(isinstance(net, NeuralNetwork) for net in networks):
        stacked = _try_stack_networks(
            networks, trainable, fast_recurrent=fast_recurrent
        )
        if stacked is not None:
            return stacked
    modules = [
        _convert_single(nn, trainable, fast_recurrent=fast_recurrent)
        for nn in networks
    ]
    return _torch_layers.EnsembleModule(modules)


class NeuralNetworkModule(_torch_layers.nn.Module):
    """Torch equivalent of `madmom_infer.ml.nn.NeuralNetwork`: applies a
    list of converted layer modules in sequence, adding/removing the
    batch dimension exactly once (see `layers.py`'s module header for the
    "always batched internally" convention every sub-module assumes).

    `expected_unbatched_ndim` (2 or 3, computed by `to_torch` via
    `_expected_unbatched_ndim`) disambiguates an unbatched call from a
    batched one: a plain `(T, F)` network's unbatched input is 2-D, so a
    3-D call is batched (`(B, T, F)`); a network whose first
    `ConvolutionalLayer` needs > 1 input channel (e.g. `onsets_cnn`,
    which stacks 3 frame-size branches on the channel axis) has a 3-D
    unbatched input (`(T, F, C)`), so a 4-D call is batched.
    """

    def __init__(self, layer_modules, expected_unbatched_ndim=2):
        super().__init__()
        self.layer_modules = _torch_layers.nn.ModuleList(layer_modules)
        self.expected_unbatched_ndim = expected_unbatched_ndim

    def forward(self, x):
        if x.dim() == self.expected_unbatched_ndim:
            added_batch = True
            x = x.unsqueeze(0)
        elif x.dim() == self.expected_unbatched_ndim + 1:
            added_batch = False
        else:
            raise ValueError(
                f"NeuralNetworkModule: expected a tensor with "
                f"{self.expected_unbatched_ndim} dims (unbatched) or "
                f"{self.expected_unbatched_ndim + 1} dims (batched), got "
                f"{x.dim()} dims (shape {tuple(x.shape)})."
            )
        for layer in self.layer_modules:
            x = layer(x)
        if added_batch:
            x = x.squeeze(0)
            x = _squeeze_all(x)
        else:
            x = _squeeze_keep_batch(x)
        return x


class ProcessorGraphModule(_torch_layers.nn.Module):
    """Torch equivalent of a raw `madmom_infer.processors.
    SequentialProcessor`/`ParallelProcessor` graph of layers -- what
    `notes_cnn.pkl` unpickles to directly (see CLAUDE.md's Wave 4e). Same
    batch-dimension handling as `NeuralNetworkModule`, but does NOT call
    `.squeeze()` at the end (the raw graph's own `.process()` never does
    either -- only `NeuralNetwork.process` has that trailing squeeze).
    """

    def __init__(self, graph_module, expected_unbatched_ndim=2):
        super().__init__()
        self.graph_module = graph_module
        self.expected_unbatched_ndim = expected_unbatched_ndim

    def forward(self, x):
        if x.dim() == self.expected_unbatched_ndim:
            added_batch = True
            x = x.unsqueeze(0)
        elif x.dim() == self.expected_unbatched_ndim + 1:
            added_batch = False
        else:
            raise ValueError(
                f"ProcessorGraphModule: expected a tensor with "
                f"{self.expected_unbatched_ndim} dims (unbatched) or "
                f"{self.expected_unbatched_ndim + 1} dims (batched), got "
                f"{x.dim()} dims (shape {tuple(x.shape)})."
            )
        out = self.graph_module(x)
        if added_batch:
            out = out.squeeze(0)
        return out


def _squeeze_all(x):
    for d in reversed(range(x.dim())):
        if x.shape[d] == 1:
            x = x.squeeze(d)
    return x


def _squeeze_keep_batch(x):
    for d in reversed(range(1, x.dim())):
        if x.shape[d] == 1:
            x = x.squeeze(d)
    return x


def to_torch(obj, trainable=False, fast_recurrent=False):
    """Convert an already-loaded numpy `madmom_infer.ml.nn` object (or a
    plain list of `NeuralNetwork`s) into an equivalent `torch.nn.Module`.

    Accepts: `NeuralNetworkEnsemble` (-> an averaging ensemble module over
    its member networks), a plain `list`/`tuple` of `NeuralNetwork`
    instances (-> same, `ensemble_to_torch`), a single `NeuralNetwork`
    (-> `NeuralNetworkModule`), or a raw `SequentialProcessor`/
    `ParallelProcessor` graph of layers (`notes_cnn.pkl`'s shape --
    `NeuralNetwork.load`/`NeuralNetworkEnsemble.load` return this
    directly, see CLAUDE.md's Wave 4e) -> `ProcessorGraphModule`.

    `trainable=True` makes every copied weight/bias/etc. an
    `nn.Parameter` instead of a frozen buffer (see this module's header).

    `fast_recurrent=True` opts eligible stacked peephole LSTMs into a
    CUDA float32 no-grad Triton path. Unsupported calls keep the eager,
    differentiable implementation.
    """
    if isinstance(obj, (list, tuple)):
        return ensemble_to_torch(
            list(obj), trainable=trainable, fast_recurrent=fast_recurrent
        )
    if isinstance(obj, NeuralNetworkEnsemble):
        networks_processor = obj.processors[0]
        return ensemble_to_torch(list(networks_processor.processors),
                                  trainable=trainable,
                                  fast_recurrent=fast_recurrent)
    if isinstance(obj, NeuralNetwork):
        stacked = _try_stack_networks(
            [obj], trainable, fast_recurrent=fast_recurrent
        )
        if stacked is not None:
            return stacked
        layer_modules = [_convert_layer(layer, trainable) for layer in obj.layers]
        ndim = _expected_unbatched_ndim(obj.layers)
        return NeuralNetworkModule(layer_modules, expected_unbatched_ndim=ndim)
    if isinstance(obj, (SequentialProcessor, ParallelProcessor)):
        graph_module = _convert_graph_node(obj, trainable)
        ndim = _expected_unbatched_ndim([obj])
        return ProcessorGraphModule(graph_module, expected_unbatched_ndim=ndim)
    raise TypeError(
        f"to_torch: don't know how to convert object of type "
        f"{type(obj).__name__!r} -- expected a madmom_infer.ml.nn."
        "NeuralNetwork/NeuralNetworkEnsemble, a list of NeuralNetworks, "
        "or a madmom_infer.processors.SequentialProcessor/ParallelProcessor "
        "graph."
    )
