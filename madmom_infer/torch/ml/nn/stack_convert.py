"""Ensemble/gate-stacked conversion path -- decides whether a numpy
network/ensemble is eligible for `stacked.py`'s fused modules and, if so,
builds them; also hosts the couple of small numpy<->torch conversion
helpers (`_convert_activation`, `_tensor`) and the unbatched-shape
inference helper (`_expected_unbatched_ndim`) that both this module and
`convert.py` need -- kept here (rather than in `convert.py`) specifically
so `convert.py` can import from this module without a circular import
(this module never imports from `convert.py`).

**Eligibility** (`_try_stack_networks`, the only function `convert.py`
calls into): a network is stackable if every one of its `.layers` is a
`FeedForwardLayer`, `RecurrentLayer`, `LSTMLayer`, `GRULayer`, or
`BidirectionalLayer` of one of those (`_layer_signature` returns `None`
for anything else, e.g. any CNN layer -- see `stacked.py`'s module header
for why CNN pipelines don't need this path at all). An ensemble of E>=1
such networks is stackable if every member shares the exact same
architecture at every layer position (`_ensemble_stack_signature`); a
single (non-ensemble) network is just the E==1 case of the same check.

**Building** (`_stack_layer_position`/`_build_stacked_network`): walks
each layer position across all E (structurally-identical) members,
concatenates their numpy weights along a new leading `E` axis (fusing
LSTM's 4 gates / GRU's reset+update pair along the OUTPUT axis too, so
`stacked.py`'s modules can do one batched matmul per timestep instead of
one per gate per member), and constructs the matching `stacked.py`
module. `BidirectionalLayer` interleaves its fwd/bwd sub-layers into a
`2 * E`-wide stack before recursing (see `_stack_bidirectional`).

Reads: torch, numpy, madmom_infer.ml.nn.activations (identity-mapped to
torch equivalents), madmom_infer.ml.nn.layers (the numpy layer classes,
isinstance targets only), madmom_infer.processors (SequentialProcessor,
ParallelProcessor, for `_expected_unbatched_ndim`'s graph walk),
madmom_infer.torch.ml.nn.stacked (the modules being constructed); read
by: madmom_infer/torch/ml/nn/convert.py.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from madmom_infer.ml.nn import activations as _np_activations
from madmom_infer.ml.nn import layers as _np_layers
from madmom_infer.processors import ParallelProcessor, SequentialProcessor

from . import stacked as _torch_stacked

# -- activation function mapping (by numpy function IDENTITY) -----------

_ACTIVATION_MAP = {
    _np_activations.linear: (lambda x: x),
    _np_activations.tanh: torch.tanh,
    _np_activations.sigmoid: torch.sigmoid,
    _np_activations.relu: (lambda x: F.relu(x)),
    _np_activations.elu: (lambda x: F.elu(x, alpha=1.0)),
    _np_activations.softmax: (lambda x: F.softmax(x, dim=-1)),
}


def _convert_activation(activation_fn):
    """Map a numpy `activation_fn` (a function from
    `madmom_infer.ml.nn.activations`, or `None`) to its torch equivalent."""
    if activation_fn is None:
        return None
    try:
        return _ACTIVATION_MAP[activation_fn]
    except KeyError as exc:
        raise TypeError(
            f"to_torch: unknown activation function {activation_fn!r} -- "
            "not one of madmom_infer.ml.nn.activations's "
            "linear/tanh/sigmoid/relu/elu/softmax."
        ) from exc


def _tensor(array, dtype=torch.float32):
    return torch.as_tensor(np.asarray(array, dtype=np.float32), dtype=dtype)


# -- unbatched-input-shape inference (shared with convert.py) ------------


def _first_conv_channels(node):
    """Walk a `NeuralNetwork`'s `.layers` list, or a
    `SequentialProcessor`/`ParallelProcessor` graph, in call order and
    return the first `ConvolutionalLayer` encountered's input-channel
    count (`weights.shape[0]`), or `None` if the (sub)graph has no
    convolutional layer at all. Used to decide whether a network's
    top-level unbatched input is `(T, F)` (channel count 1, the channel
    axis is optional/implicit) or `(T, F, C)` (channel count > 1, the
    caller MUST supply the channel axis explicitly) -- see
    `convert.py`'s `NeuralNetworkModule`/`ProcessorGraphModule`
    docstrings.
    """
    if isinstance(node, _np_layers.ConvolutionalLayer):
        return node.weights.shape[0]
    if isinstance(node, (SequentialProcessor, ParallelProcessor)):
        for child in node.processors:
            found = _first_conv_channels(child)
            if found is not None:
                return found
        return None
    return None


def _expected_unbatched_ndim(layers):
    for layer in layers:
        channels = _first_conv_channels(layer)
        if channels is not None:
            return 3 if channels > 1 else 2
    return 2


# -- eligibility (architecture signatures) --------------------------------


def _sig_gate(gate):
    """Hashable (type, shape, has-peephole, activation-identity) signature
    for one `Gate`/`Cell`/`GRUCell` instance -- used only to compare
    architecture across ensemble members, never their actual weight
    values (which are expected, and required, to differ)."""
    peep = getattr(gate, "peephole_weights", None)
    return (
        tuple(gate.weights.shape),
        None if peep is None else tuple(peep.shape),
        gate.activation_fn,
    )


def _layer_signature(layer):
    """Hashable architecture signature for one numpy layer, or `None` if
    `layer`'s type isn't one of the 5 stackable ones (`FeedForwardLayer`,
    `RecurrentLayer`, `LSTMLayer`, `GRULayer`, `BidirectionalLayer`) --
    `None` propagates up through `_network_signature` to disqualify the
    whole network/ensemble from the stacked path, falling back to
    `convert.py`'s per-network `_convert_layer` path. Order matters, same
    reason as `_convert_layer`'s own dispatch: `ConvolutionalLayer` is
    excluded explicitly (it subclasses `FeedForwardLayer`); `LSTMLayer`/
    `GRULayer` are checked before their common base `RecurrentLayer`.
    """
    if isinstance(layer, _np_layers.LSTMLayer):
        return (
            "lstm",
            _sig_gate(layer.input_gate),
            _sig_gate(layer.forget_gate),
            _sig_gate(layer.cell),
            _sig_gate(layer.output_gate),
            layer.activation_fn,
        )
    if isinstance(layer, _np_layers.GRULayer):
        return (
            "gru",
            _sig_gate(layer.reset_gate),
            _sig_gate(layer.update_gate),
            _sig_gate(layer.cell),
        )
    if isinstance(layer, _np_layers.BidirectionalLayer):
        fwd_sig = _layer_signature(layer.fwd_layer)
        bwd_sig = _layer_signature(layer.bwd_layer)
        if fwd_sig is None or bwd_sig is None:
            return None
        return ("bidir", fwd_sig, bwd_sig)
    if isinstance(layer, _np_layers.ConvolutionalLayer):
        return None
    if isinstance(layer, _np_layers.RecurrentLayer):
        return (
            "rec",
            tuple(layer.weights.shape),
            tuple(layer.recurrent_weights.shape),
            layer.activation_fn,
        )
    if isinstance(layer, _np_layers.FeedForwardLayer):
        return ("ff", tuple(layer.weights.shape), layer.activation_fn)
    return None


def _network_signature(network):
    """`tuple` of `_layer_signature(...)` for every layer in `network`, or
    `None` if any layer isn't stackable."""
    sigs = [_layer_signature(layer) for layer in network.layers]
    if any(sig is None for sig in sigs):
        return None
    return tuple(sigs)


def _ensemble_stack_signature(networks):
    """The shared architecture signature for `networks` (a list of one or
    more `NeuralNetwork` instances), or `None` if any network is
    ineligible (has a non-stackable layer) or the networks don't all
    share the exact same architecture (different layer count/shapes/
    activations at some position)."""
    sigs = [_network_signature(net) for net in networks]
    if any(sig is None for sig in sigs):
        return None
    if len(set(sigs)) != 1:
        return None
    return sigs[0]


# -- building the stacked modules -----------------------------------------


def _stack_tensor(arrays):
    return torch.stack([_tensor(a) for a in arrays], dim=0)


def _stack_ff(layers, trainable):
    w = _stack_tensor([layer.weights for layer in layers])
    b = _stack_tensor([layer.bias for layer in layers])
    act = _convert_activation(layers[0].activation_fn)
    return _torch_stacked.StackedFeedForwardLayer(w, b, activation=act,
                                                   trainable=trainable)


def _stack_recurrent(layers, trainable):
    w = _stack_tensor([layer.weights for layer in layers])
    b = _stack_tensor([layer.bias for layer in layers])
    rw = _stack_tensor([layer.recurrent_weights for layer in layers])
    init = _stack_tensor([layer.init for layer in layers])
    act = _convert_activation(layers[0].activation_fn)
    return _torch_stacked.StackedRecurrentLayer(w, b, rw, init,
                                                 activation=act,
                                                 trainable=trainable)


def _stack_lstm(layers, trainable, fast_recurrent=False):
    hidden = layers[0].input_gate.weights.shape[1]
    has_peep = layers[0].input_gate.peephole_weights is not None
    w_list, b_list, rw_list = [], [], []
    peep_i_list, peep_f_list, peep_o_list = [], [], []
    for layer in layers:
        gates = (layer.input_gate, layer.forget_gate, layer.cell,
                 layer.output_gate)
        w_list.append(np.concatenate([g.weights for g in gates], axis=1))
        b_list.append(np.concatenate([g.bias for g in gates], axis=0))
        rw_list.append(
            np.concatenate([g.recurrent_weights for g in gates], axis=1)
        )
        if has_peep:
            peep_i_list.append(layer.input_gate.peephole_weights)
            peep_f_list.append(layer.forget_gate.peephole_weights)
            peep_o_list.append(layer.output_gate.peephole_weights)
    w = _stack_tensor(w_list)
    b = _stack_tensor(b_list)
    rw = _stack_tensor(rw_list)
    peep_i = _stack_tensor(peep_i_list) if has_peep else None
    peep_f = _stack_tensor(peep_f_list) if has_peep else None
    peep_o = _stack_tensor(peep_o_list) if has_peep else None
    init = _stack_tensor([layer.init for layer in layers])
    cell_init = _stack_tensor([layer.cell_init for layer in layers])
    act_i = _convert_activation(layers[0].input_gate.activation_fn)
    act_f = _convert_activation(layers[0].forget_gate.activation_fn)
    act_c = _convert_activation(layers[0].cell.activation_fn)
    act_o = _convert_activation(layers[0].output_gate.activation_fn)
    act_out = _convert_activation(layers[0].activation_fn)
    return _torch_stacked.StackedLSTMLayer(
        w, b, rw, hidden, peep_i, peep_f, peep_o, init, cell_init,
        act_i, act_f, act_c, act_o, act_out, trainable=trainable,
        fast_recurrent=fast_recurrent,
    )


def _stack_gru(layers, trainable):
    hidden = layers[0].reset_gate.weights.shape[1]
    w_ru_list, b_ru_list, rw_ru_list = [], [], []
    w_c_list, b_c_list, rw_c_list = [], [], []
    for layer in layers:
        rg, ug, cell = layer.reset_gate, layer.update_gate, layer.cell
        w_ru_list.append(np.concatenate([rg.weights, ug.weights], axis=1))
        b_ru_list.append(np.concatenate([rg.bias, ug.bias], axis=0))
        rw_ru_list.append(
            np.concatenate([rg.recurrent_weights, ug.recurrent_weights], axis=1)
        )
        w_c_list.append(cell.weights)
        b_c_list.append(cell.bias)
        rw_c_list.append(cell.recurrent_weights)
    w_ru = _stack_tensor(w_ru_list)
    b_ru = _stack_tensor(b_ru_list)
    rw_ru = _stack_tensor(rw_ru_list)
    w_c = _stack_tensor(w_c_list)
    b_c = _stack_tensor(b_c_list)
    rw_c = _stack_tensor(rw_c_list)
    init = _stack_tensor([layer.init for layer in layers])
    act_r = _convert_activation(layers[0].reset_gate.activation_fn)
    act_u = _convert_activation(layers[0].update_gate.activation_fn)
    act_c = _convert_activation(layers[0].cell.activation_fn)
    return _torch_stacked.StackedGRULayer(
        w_ru, b_ru, rw_ru, w_c, b_c, rw_c, hidden, init,
        act_r, act_u, act_c, trainable=trainable,
    )


def _stack_bidirectional(layers, trainable, fast_recurrent=False):
    # Interleave fwd/bwd sub-layers: slot 2*i = member i's fwd, 2*i+1 =
    # member i's bwd -- StackedBidirectionalLayer.forward relies on this
    # exact order (see its docstring).
    inner_layers = []
    for layer in layers:
        inner_layers.append(layer.fwd_layer)
        inner_layers.append(layer.bwd_layer)
    inner = _stack_layer_position(
        inner_layers, trainable, fast_recurrent=fast_recurrent
    )
    return _torch_stacked.StackedBidirectionalLayer(inner)


def _stack_layer_position(layers, trainable, fast_recurrent=False):
    """Build one `stacked.py` module covering all of `layers` (one numpy
    layer per ensemble member, all at the same position/architecture --
    guaranteed by `_ensemble_stack_signature` before this is ever
    called)."""
    sample = layers[0]
    if isinstance(sample, _np_layers.LSTMLayer):
        return _stack_lstm(
            layers, trainable, fast_recurrent=fast_recurrent
        )
    if isinstance(sample, _np_layers.GRULayer):
        return _stack_gru(layers, trainable)
    if isinstance(sample, _np_layers.BidirectionalLayer):
        return _stack_bidirectional(
            layers, trainable, fast_recurrent=fast_recurrent
        )
    if isinstance(sample, _np_layers.RecurrentLayer):
        return _stack_recurrent(layers, trainable)
    if isinstance(sample, _np_layers.FeedForwardLayer):
        return _stack_ff(layers, trainable)
    raise TypeError(
        f"_stack_layer_position: don't know how to stack layer type "
        f"{type(sample).__name__!r} (should have been excluded by "
        "_layer_signature)."
    )


def _build_stacked_network(networks, trainable, fast_recurrent=False):
    """Build a `StackedEnsembleModule` covering every layer position of
    `networks` (>= 1 structurally-identical `NeuralNetwork` instances --
    `_ensemble_stack_signature(networks)` must already be known non-`None`
    before calling this)."""
    num_layers = len(networks[0].layers)
    stacked_layers = [
        _stack_layer_position(
            [net.layers[i] for net in networks], trainable,
            fast_recurrent=fast_recurrent,
        )
        for i in range(num_layers)
    ]
    ndim = _expected_unbatched_ndim(networks[0].layers)
    return _torch_stacked.StackedEnsembleModule(
        stacked_layers, len(networks), expected_unbatched_ndim=ndim
    )


def _try_stack_networks(networks, trainable, fast_recurrent=False):
    """Return a `StackedEnsembleModule` for `networks` (a list of one or
    more `NeuralNetwork` instances) if they're all structurally identical
    and every layer is stackable, else `None` (the caller falls back to
    `convert.py`'s per-network/per-layer path)."""
    if _ensemble_stack_signature(networks) is None:
        return None
    return _build_stacked_network(
        networks, trainable, fast_recurrent=fast_recurrent
    )
