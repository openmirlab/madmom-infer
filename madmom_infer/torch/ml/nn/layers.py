"""Torch `nn.Module` equivalents of every class in `madmom_infer.ml.nn.layers`.

This is the differentiable, GPU-capable twin of the numpy layer classes --
same forward math, expressed as autograd-friendly torch ops instead of
numpy/scipy calls. It never constructs itself from a `.pkl` file directly;
`madmom_infer.torch.ml.nn.convert.to_torch` builds these from an
already-loaded numpy layer (via `madmom_infer.ml.nn.NeuralNetwork.load`)
and copies its weights in.

**Batch convention (load-bearing, read before editing):** every module
here operates on tensors that ALWAYS carry a leading batch dimension `B`
-- i.e. the numpy layer's own unbatched shape (`(T, F)`, `(T, F, C)`, a
per-timestep `(F,)` gate input, ...) with one extra dimension prepended.
The public entry points in `convert.py` (`NeuralNetworkModule`,
`ProcessorGraphModule`) are the ONLY place that adds this batch dimension
(via `unsqueeze(0)` when the caller passes an unbatched tensor) and removes
it again at the end (mirroring `NeuralNetwork.process`'s trailing
`.squeeze()`) -- every module in *this* file assumes it has already been
added. This is why axis/dim arguments baked into a module at construction
time (`PadLayer.axes`, `AverageLayer.axis`, `TransposeLayer.axes`,
`ReshapeLayer.newshape`) are pre-shifted by +1 in `convert.py` before being
passed in here: axis 0 in the numpy layer's own docstring is always axis 1
in this module's tensors (axis 0 is reserved for the always-present batch
dimension).

Weight convention: numpy's `FeedForwardLayer`/`RecurrentLayer`/`Gate`
family compute `np.dot(data, weights)` with `weights` shape `(in, out)` --
the torch equivalent is `x @ weights` (`torch.matmul`), no transpose
needed, since torch's batched matmul broadcasts a trailing `(in, out)`
matrix against a leading `(B, T, in)` tensor exactly like numpy's `dot`
broadcasts `(T, in) @ (in, out)`.

All weights/biases are registered as float32 buffers by default (matching
`NN_DTYPE = np.float32` in the numpy layers module) and cast to the input
tensor's dtype at forward time, so a caller can run a float64 precision
check or a float32/GPU pass through the exact same module instance without
re-converting it. `trainable=True` (passed down from `convert.to_torch`)
registers the same tensors as `nn.Parameter` instead of buffers.

No in-place tensor ops, no `.item()`/`.numpy()` calls anywhere in a
`forward()` -- every module here is meant to sit inside an autograd graph.

Reads: torch, torch.nn.functional; read by:
madmom_infer/torch/ml/nn/convert.py (the only place that constructs these).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _register(module, name, array, trainable):
    """Register `array` (already a torch tensor) on `module` as `name`,
    either as an `nn.Parameter` (`trainable=True`) or a float buffer."""
    if trainable:
        module.register_parameter(name, nn.Parameter(array))
    else:
        module.register_buffer(name, array)


def _to(tensor, x):
    """Cast a registered buffer/parameter to `x`'s dtype and device."""
    return tensor.to(dtype=x.dtype, device=x.device)


class FeedForwardLayer(nn.Module):
    """`activation(x @ weights + bias)`. Torch twin of
    `madmom_infer.ml.nn.layers.FeedForwardLayer`."""

    def __init__(self, weights, bias, activation=None, trainable=False):
        super().__init__()
        _register(self, "weights", weights, trainable)
        _register(self, "bias", bias, trainable)
        self.activation = activation

    def forward(self, x):
        w = _to(self.weights, x)
        b = _to(self.bias, x)
        out = torch.matmul(x, w) + b
        if self.activation is not None:
            out = self.activation(out)
        return out


class RecurrentLayer(nn.Module):
    """One-frame-at-a-time recurrent layer: `activation(x_t @ weights +
    bias + prev @ recurrent_weights)`, `prev` starting at `init`. Torch
    twin of `madmom_infer.ml.nn.layers.RecurrentLayer`. Vectorized across
    the batch dimension; sequential only over time (`T`)."""

    def __init__(self, weights, bias, recurrent_weights, init,
                 activation=None, trainable=False):
        super().__init__()
        _register(self, "weights", weights, trainable)
        _register(self, "bias", bias, trainable)
        _register(self, "recurrent_weights", recurrent_weights, trainable)
        _register(self, "init", init, trainable=False)
        self.activation = activation

    def forward(self, x):
        # x: (B, T, in) -> (B, T, out)
        w = _to(self.weights, x)
        b = _to(self.bias, x)
        rw = _to(self.recurrent_weights, x)
        init = _to(self.init, x)
        batch = x.shape[0]
        prev = init.unsqueeze(0).expand(batch, -1)
        # precompute the non-recurrent term for every frame at once
        ff = torch.matmul(x, w) + b
        outs = []
        for t in range(x.shape[1]):
            cur = ff[:, t, :] + torch.matmul(prev, rw)
            if self.activation is not None:
                cur = self.activation(cur)
            outs.append(cur)
            prev = cur
        return torch.stack(outs, dim=1)


class BidirectionalLayer(nn.Module):
    """Runs `fwd_layer` forward and `bwd_layer` on the time-reversed input,
    re-reverses, and concatenates on the feature axis. Torch twin of
    `madmom_infer.ml.nn.layers.BidirectionalLayer`."""

    def __init__(self, fwd_layer, bwd_layer):
        super().__init__()
        self.fwd_layer = fwd_layer
        self.bwd_layer = bwd_layer

    def forward(self, x):
        fwd = self.fwd_layer(x)
        bwd = self.bwd_layer(torch.flip(x, dims=[1]))
        bwd = torch.flip(bwd, dims=[1])
        return torch.cat([fwd, bwd], dim=-1)


class _GateParams(nn.Module):
    """Holds one gate/cell's parameters (weights, bias, recurrent_weights,
    optional peephole_weights) plus its activation. Shared building block
    for `LSTMLayer`/`GRULayer` -- not a standalone forward module (its
    call signature differs per use, see `Gate`/`Cell`/`GRUCell` in the
    numpy layers module), just a parameter container with a `_pre(...)`
    helper for the shared `x @ weights + bias (+ peephole) + prev @
    recurrent_weights` arithmetic.
    """

    def __init__(self, weights, bias, recurrent_weights, peephole_weights,
                 activation, trainable=False):
        super().__init__()
        _register(self, "weights", weights, trainable)
        _register(self, "bias", bias, trainable)
        _register(self, "recurrent_weights", recurrent_weights, trainable)
        self.has_peephole = peephole_weights is not None
        if self.has_peephole:
            _register(self, "peephole_weights", peephole_weights, trainable)
        self.activation = activation

    def pre(self, x_t, prev, state=None):
        w = _to(self.weights, x_t)
        b = _to(self.bias, x_t)
        rw = _to(self.recurrent_weights, x_t)
        out = torch.matmul(x_t, w) + b
        if self.has_peephole:
            peep = _to(self.peephole_weights, x_t)
            out = out + state * peep
        out = out + torch.matmul(prev, rw)
        return out

    def activate(self, x_t, prev, state=None):
        return self.activation(self.pre(x_t, prev, state))


class LSTMLayer(nn.Module):
    """Four-gate LSTM (input/forget/cell/output), driven one frame at a
    time, peephole connections on ig/fg/og. Torch twin of
    `madmom_infer.ml.nn.layers.LSTMLayer` -- a custom loop cell, not
    `torch.nn.LSTM` (which has no peephole support).

    Numpy semantics reproduced exactly: ig/fg peep at the PREVIOUS cell
    state, og peeps at the NEW (just-updated) cell state, and the layer's
    own output is `activation(state) * og` (`activation` defaults to
    `tanh`, matching `LSTMLayer.activation_fn`).
    """

    def __init__(self, input_gate, forget_gate, cell, output_gate,
                 init, cell_init, activation=None, trainable=False):
        super().__init__()
        self.input_gate = input_gate
        self.forget_gate = forget_gate
        self.cell = cell
        self.output_gate = output_gate
        _register(self, "init", init, trainable=False)
        _register(self, "cell_init", cell_init, trainable=False)
        self.activation = activation

    def forward(self, x):
        # x: (B, T, in) -> (B, T, hidden)
        batch = x.shape[0]
        prev = _to(self.init, x).unsqueeze(0).expand(batch, -1)
        state = _to(self.cell_init, x).unsqueeze(0).expand(batch, -1)
        outs = []
        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            ig = self.input_gate.activate(x_t, prev, state)
            fg = self.forget_gate.activate(x_t, prev, state)
            cell = self.cell.activate(x_t, prev)
            state = cell * ig + state * fg
            og = self.output_gate.activate(x_t, prev, state)
            out_t = self.activation(state) * og if self.activation is not None \
                else state * og
            outs.append(out_t)
            prev = out_t
        return torch.stack(outs, dim=1)


class GRULayer(nn.Module):
    """Cho et al. 2014-style GRU: reset/update gates plus a cell whose
    recurrent term is gated by the reset gate. Torch twin of
    `madmom_infer.ml.nn.layers.GRULayer` -- a custom loop cell, not
    `torch.nn.GRU` (different gate-ordering/formulation).
    """

    def __init__(self, reset_gate, update_gate, cell, init, trainable=False):
        super().__init__()
        self.reset_gate = reset_gate
        self.update_gate = update_gate
        self.cell = cell
        _register(self, "init", init, trainable=False)

    def forward(self, x):
        batch = x.shape[0]
        prev = _to(self.init, x).unsqueeze(0).expand(batch, -1)
        outs = []
        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            rg = self.reset_gate.activate(x_t, prev)
            ug = self.update_gate.activate(x_t, prev)
            # GRUCell: tanh(x @ W + b + reset_gate * (prev @ recurrent_weights))
            w = _to(self.cell.weights, x_t)
            b = _to(self.cell.bias, x_t)
            rw = _to(self.cell.recurrent_weights, x_t)
            cell_pre = torch.matmul(x_t, w) + b + rg * torch.matmul(prev, rw)
            cell = self.cell.activation(cell_pre)
            out_t = ug * cell + (1 - ug) * prev
            outs.append(out_t)
            prev = out_t
        return torch.stack(outs, dim=1)


def _kernel_margins_same_pad(kt, kf):
    """Left/right (top/bottom, left/right) zero-padding amounts that make
    a flipped-kernel `F.conv2d` reproduce `scipy.ndimage.convolve(...,
    mode='constant')`'s 'same'-shape output exactly (verified numerically
    against `madmom_infer.ml.nn.layers.ConvolutionalLayer(pad='same')` for
    both odd and even kernel sizes -- see this module's own test suite,
    `tests/test_torch_nn.py`). For an odd kernel this is the usual
    symmetric `(k - 1) // 2` on both sides; for an EVEN kernel
    `scipy.ndimage`'s default origin makes the padding asymmetric: `(k -
    1) // 2` on the low side, `k // 2` on the high side (one more zero on
    the high/right side than the low/left side).
    """
    pt_l, pt_r = (kt - 1) // 2, kt // 2
    pf_l, pf_r = (kf - 1) // 2, kf // 2
    return pt_l, pt_r, pf_l, pf_r


class ConvolutionalLayer(nn.Module):
    """2D convolution over `(B, T, F[, C])` data (channel axis optional,
    default 1 channel), one kernel per output feature map, per input
    channel, summed across channels, plus bias and activation.

    Torch twin of `madmom_infer.ml.nn.layers.ConvolutionalLayer`. numpy's
    `weights` array has shape `(C_in, F_out, kt, kf)` and the underlying
    op (`scipy.ndimage.convolve`) is a TRUE convolution (flips the
    kernel), not a cross-correlation -- `torch.nn.functional.conv2d` is a
    cross-correlation, so the kernel is flipped once at conversion time
    (see `convert.py`) before being stored here. `pad='valid'` is a plain
    `conv2d` with no padding (mathematically identical to numpy's
    full-mode-then-crop, since the cropped 'valid' region never touches
    the zero-padded boundary); `pad='same'` zero-pads asymmetrically for
    even kernels via `_kernel_margins_same_pad` (verified, not assumed --
    see that function's docstring). `stride` is applied by slicing the
    activated output, exactly like the numpy layer (not passed to
    `conv2d` itself), to keep this module's stride semantics identical to
    numpy's own `out[::stride[0], ::stride[1]]`.
    """

    def __init__(self, weights_flipped, bias, stride=None, pad="valid",
                 activation=None, trainable=False):
        super().__init__()
        # weights_flipped: (F_out, C_in, kt, kf), already flipped for
        # cross-correlation == true-convolution equivalence.
        _register(self, "weights", weights_flipped, trainable)
        _register(self, "bias", bias, trainable)
        self.stride = stride
        self.pad = pad
        self.activation = activation

    def forward(self, x):
        # x: (B, T, F) or (B, T, F, C) -> add a size-1 channel if missing
        added_channel = False
        if x.dim() == 3:
            x = x.unsqueeze(-1)
            added_channel = True
        # (B, T, F, C) -> (B, C, T, F) for conv2d
        x_chw = x.permute(0, 3, 1, 2)
        w = _to(self.weights, x)
        kt, kf = w.shape[-2], w.shape[-1]
        if self.pad == "valid":
            out = F.conv2d(x_chw, w, bias=None, padding=0)
        elif self.pad == "same":
            pt_l, pt_r, pf_l, pf_r = _kernel_margins_same_pad(kt, kf)
            x_chw = F.pad(x_chw, (pf_l, pf_r, pt_l, pt_r))
            out = F.conv2d(x_chw, w, bias=None, padding=0)
        else:
            raise NotImplementedError(
                f"`pad` must be 'valid' or 'same', got {self.pad!r}."
            )
        b = _to(self.bias, x).view(1, -1, 1, 1)
        out = out + b
        if self.activation is not None:
            out = self.activation(out)
        # (B, C_out, T', F') -> (B, T', F', C_out)
        out = out.permute(0, 2, 3, 1)
        if self.stride not in (None, 1, (1, 1)):
            out = out[:, :: self.stride[0], :: self.stride[1], :]
        # numpy's ConvolutionalLayer never drops the (always-present)
        # output-feature-map axis even when the input arrived without an
        # explicit input channel axis -- `added_channel` only controlled
        # whether we needed to *add* one before convolving, not the shape
        # of the result, so nothing to undo here.
        del added_channel
        return out


class MaxPoolLayer(nn.Module):
    """2D max pooling over `(B, T, F[, C])` data (`axis=None`, spatial
    mode) or a plain `max` reduction along one axis (`axis` set). Torch
    twin of `madmom_infer.ml.nn.layers.MaxPoolLayer`. Spatial mode uses
    `F.max_pool2d` with no padding -- verified numerically to match
    numpy's `maximum_filter` + centered slice for both even/odd
    `size`/`stride` and inputs not evenly divisible by `stride` (see
    `tests/test_torch_nn.py`).
    """

    def __init__(self, size=None, stride=None, axis=None):
        super().__init__()
        self.size = size
        self.stride = stride
        self.axis = axis  # already shifted by +1 if set, see convert.py

    def forward(self, x):
        if self.axis is not None:
            return torch.amax(x, dim=self.axis)
        added_channel = False
        if x.dim() == 3:
            x = x.unsqueeze(-1)
            added_channel = True
        x_chw = x.permute(0, 3, 1, 2)
        out = F.max_pool2d(
            x_chw,
            kernel_size=(self.size[0], self.size[1]),
            stride=(self.stride[0], self.stride[1]),
        )
        out = out.permute(0, 2, 3, 1)
        if added_channel:
            out = out.squeeze(-1)
        return out


class BatchNormLayer(nn.Module):
    """`(data - mean) * (gamma * inv_std) + beta`, then activation. Torch
    twin of `madmom_infer.ml.nn.layers.BatchNormLayer` (inference-mode
    batch norm using fixed, already-fitted statistics -- no running-stats
    update, matching the numpy layer exactly)."""

    def __init__(self, beta, gamma, mean, inv_std, activation=None,
                 trainable=False):
        super().__init__()
        _register(self, "beta", beta, trainable)
        _register(self, "gamma", gamma, trainable)
        _register(self, "mean", mean, trainable)
        _register(self, "inv_std", inv_std, trainable)
        self.activation = activation

    def forward(self, x):
        beta = _to(self.beta, x)
        gamma = _to(self.gamma, x)
        mean = _to(self.mean, x)
        inv_std = _to(self.inv_std, x)
        out = (x - mean) * (gamma * inv_std) + beta
        if self.activation is not None:
            out = self.activation(out)
        return out


class AverageLayer(nn.Module):
    """`torch.mean(data, dim=axis, keepdim=keepdims)`. Torch twin of
    `madmom_infer.ml.nn.layers.AverageLayer`; `axis` is pre-shifted by +1
    (batch-dim-aware) by `convert.py` before construction."""

    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        if self.axis is None:
            return torch.mean(x)
        return torch.mean(x, dim=self.axis, keepdim=self.keepdims)


class PadLayer(nn.Module):
    """Constant-value padding along the given (batch-shifted) axes,
    `width` on both sides of each. Torch twin of
    `madmom_infer.ml.nn.layers.PadLayer`, implemented via `torch.cat` (not
    `F.pad`, which only reaches the last N dims) so it works for any axis
    position, and stays autograd-friendly (no in-place writes)."""

    def __init__(self, width, axes, value=0.0):
        super().__init__()
        self.width = width
        self.axes = axes  # already shifted by +1, see convert.py
        self.value = value

    def forward(self, x):
        for ax in self.axes:
            shape = list(x.shape)
            shape[ax] = self.width
            pad_block = torch.full(shape, self.value, dtype=x.dtype,
                                    device=x.device)
            x = torch.cat([pad_block, x, pad_block], dim=ax)
        return x


class StrideLayer(nn.Module):
    """Re-arranges `(B, T, *rest)` into overlapping length-`block_size`
    windows along the time axis, flattened per window. Torch twin of
    `madmom_infer.ml.nn.layers.StrideLayer` (which delegates to
    `madmom_infer/utils.py`'s `segment_axis`); implemented with
    `Tensor.unfold` along the (batch-shifted) time axis, moved into
    `segment_axis`'s exact window-then-rest element ordering (verified
    numerically, see `tests/test_torch_nn.py`) before flattening.
    """

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        # x: (B, T, *rest) -> unfold along time (dim=1)
        unfolded = x.unfold(1, self.block_size, 1)  # (B, T', *rest, block)
        unfolded = torch.movedim(unfolded, -1, 2)  # (B, T', block, *rest)
        batch, new_t = unfolded.shape[0], unfolded.shape[1]
        return unfolded.reshape(batch, new_t, -1)


class TransposeLayer(nn.Module):
    """`torch.permute(data, axes)`. Torch twin of
    `madmom_infer.ml.nn.layers.TransposeLayer`; `axes` is pre-shifted
    (batch dim prepended and kept at position 0) by `convert.py`."""

    def __init__(self, axes):
        super().__init__()
        self.axes = axes

    def forward(self, x):
        return x.permute(*self.axes)


class ReshapeLayer(nn.Module):
    """`torch.reshape(data, newshape)`. Torch twin of
    `madmom_infer.ml.nn.layers.ReshapeLayer`; `newshape` is pre-adjusted
    by `convert.py` to keep the leading batch dimension intact (a `-1` in
    the numpy `newshape` is left as `-1`, since it already infers the
    right size; every other entry is used as-is since batching only adds
    a NEW leading dimension, it does not change any existing axis
    size)."""

    def __init__(self, newshape):
        super().__init__()
        self.newshape = newshape

    def forward(self, x):
        batch = x.shape[0]
        return x.reshape(batch, *self.newshape)


# -- processor-graph nodes (notes_cnn.pkl's raw SequentialProcessor/
# ParallelProcessor/numpy.dstack graph, see convert.py) -----------------


class SequentialGraphModule(nn.Module):
    """Applies a list of converted graph-node modules in order. Torch
    twin of `madmom_infer.processors.SequentialProcessor`'s `.process()`
    fold, used only for the raw (non-`NeuralNetwork`) processor graphs
    `madmom_infer.torch.ml.nn.convert.to_torch` builds for pickles like
    `notes_cnn.pkl` that unpickle to a bare `SequentialProcessor` graph."""

    def __init__(self, children):
        super().__init__()
        self.graph_children = nn.ModuleList(children)

    def forward(self, x):
        for child in self.graph_children:
            x = child(x)
        return x


class ParallelGraphModule(nn.Module):
    """Fans the same input out to every child module, returning a list of
    outputs (one per child) -- torch twin of
    `madmom_infer.processors.ParallelProcessor.process`."""

    def __init__(self, children):
        super().__init__()
        self.graph_children = nn.ModuleList(children)

    def forward(self, x):
        return [child(x) for child in self.graph_children]


class DstackModule(nn.Module):
    """Torch equivalent of the bare `numpy.dstack` function `notes_cnn.
    pkl`'s multi-task graph pickles as its final merge stage: stacks a
    list of same-shaped tensors along a new last axis (`numpy.dstack` on
    a list of 2-D `(T, F)` arrays produces `(T, F, len(list))`; with the
    always-present batch dimension, each branch's tensor is `(B, T, F)`
    and the result is `(B, T, F, len(list))`)."""

    def forward(self, xs):
        return torch.stack(xs, dim=-1)


class EnsembleModule(nn.Module):
    """Averages the outputs of several `NeuralNetworkModule`s run on the
    same input -- torch twin of `madmom_infer.ml.nn.NeuralNetworkEnsemble`
    (a `ParallelProcessor` of networks + `average_predictions`)."""

    def __init__(self, networks):
        super().__init__()
        self.networks = nn.ModuleList(networks)

    def forward(self, x):
        preds = [net(x) for net in self.networks]
        if len(preds) == 1:
            return preds[0]
        return torch.stack(preds, dim=0).mean(dim=0)
