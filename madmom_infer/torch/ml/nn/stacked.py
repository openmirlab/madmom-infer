"""Ensemble-stacked + gate-fused torch twins of the recurrent-family numpy
layers (`RecurrentLayer`/`LSTMLayer`/`GRULayer`/`BidirectionalLayer`/
`FeedForwardLayer`) -- the performance-optimized alternative to
`layers.py`'s straightforward per-network, per-gate loop.

This module is pure forward math (like `layers.py`); it never touches a
numpy layer directly. `stack_convert.py` is the only caller -- it walks
each ensemble/network's layers, decides eligibility, extracts and fuses
the numpy weights, and constructs these modules.

**Why this exists (the problem, measured).** `layers.py`'s
`RecurrentLayer`/`LSTMLayer`/`GRULayer` each drive their own Python-level
`for t in range(T)` loop, and `layers.EnsembleModule` runs E of those
loops one after another (once per ensemble member). For an 8-network
BLSTM ensemble (`downbeats_blstm`) over a 3000-frame clip, that is 8
networks x up to 3 bidirectional LSTM layers x 3000 timesteps x 8 small
matmuls per timestep (4 gates x {input projection, recurrent projection})
-- tens of thousands of tiny kernel launches, which is why torch (both
CPU and, especially, CUDA -- kernel-launch overhead per step dominates)
measured slower than numpy's tighter C loop despite doing the same FLOPs
(see `tools/bench_torch_backend.py`'s baseline table).

**What this module does about it**, per `convert.py`'s
`_try_stack_networks` (the only place that decides whether a given
network/ensemble is eligible -- eligibility requires every layer to be
one of the 5 types below; a CNN-containing network, e.g. `onsets_cnn`,
falls back to `layers.EnsembleModule` unchanged, since CNN pipelines are
already fast, see the module docstring in `convert.py`):

1. **Ensemble stacking**: E structurally-identical networks' weights are
   concatenated along a new leading `E` axis and the *same* input is
   broadcast across it, so all E networks run as ONE batched-matmul
   sequence instead of E separate ones (`torch.bmm` treats axis 0 as a
   batch dimension exactly like this). A single (non-ensemble) network is
   just the `E == 1` case of this -- it still benefits from gate fusion
   below, and the `E`-broadcast costs one `expand` (no copy until the
   first real op needs contiguous memory).
2. **Bidirectional fusion**: `StackedBidirectionalLayer` interleaves each
   ensemble member's forward/backward sub-layers into one `2 * E`-wide
   stack (slot `2*i`/`2*i+1` = member `i`'s fwd/bwd) and runs ONE time
   loop over both directions at once, instead of E separate
   forward-then-backward passes.
3. **Fused gates**: `StackedLSTMLayer` concatenates the 4 gates'
   (input/forget/cell/output) weight matrices along the output axis at
   conversion time, so each timestep does ONE input-projection matmul and
   ONE recurrent-projection matmul (not 4-of-each) -- peephole terms
   (elementwise) are added back in per-gate after splitting, preserving
   the exact ig/fg-peek-previous-state / og-peek-new-state order
   `layers.LSTMLayer` uses. `StackedGRULayer` fuses the reset/update gate
   pair the same way; the cell's own recurrent term is kept separate
   (it's gated by the reset gate's *output*, so it can't be fused with
   the reset/update projection itself) but its `x @ W + b` term -- like
   every other layer's non-recurrent term here -- is precomputed for
   every timestep at once outside the loop, since it doesn't depend on
   recurrence either.

None of this changes any number computed beyond float32-noise-scale
reordering of a handful of additions (e.g. peephole-then-recurrent vs.
recurrent-then-peephole) -- see `tests/test_torch_nn.py`'s
`test_stacked_*_matches_unstacked_reference` tests, which compare a
stacked module directly against `layers.py`'s unstacked reference at
float32 (1e-5) and float64 (1e-10) tolerance on identical weights/input.
Autograd is unaffected: every op here (`bmm`, `split`, elementwise,
`stack`, `cat`, `flip`, `expand`) is a standard differentiable torch op.

**Batch/ensemble tensor convention.** Every module here operates on `(E,
B, T, F)` tensors: `E` = ensemble size (broadcast dimension, `torch.bmm`'s
batch axis), `B` = the real (possibly-1) caller batch, `T` = time, `F` =
features. `convert.py`'s `_try_stack_networks` is the only place that
introduces `E`; the `E`/`B` distinction matters for `trainable=True`
(fine-tuning): a stacked layer's weight is ONE `nn.Parameter` of shape
`(E, ...)` covering all E ensemble members at once (not E separate
Parameters, unlike `layers.EnsembleModule`'s per-network modules) --
gradients still flow correctly to each member's own slice (autograd
tracks through the `torch.cat`-then-`bmm`-then-`split` chain elementwise),
but `list(module.parameters())` is shorter and each entry is bigger; see
`convert.py`'s module header and `tests/test_torch_nn.py`'s
`test_stacked_ensemble_trainable_gradients_reach_all_members` for how to
recover a per-member gradient (index the Parameter's `.grad` along dim 0).

Reads: torch, .layers (`_register`/`_to`, the shared buffer/parameter
registration + dtype-cast helpers); read by:
madmom_infer/torch/ml/nn/stack_convert.py (the only place that
constructs these), madmom_infer/torch/ml/nn/__init__.py (re-export).
"""

from __future__ import annotations

import torch
from torch import nn

from .layers import _register, _to


class StackedFeedForwardLayer(nn.Module):
    """`E` structurally-identical `FeedForwardLayer`s run as one batched
    matmul. Weights `(E, in, out)`, bias `(E, out)`. Input/output: `(E, B,
    T, in)` -> `(E, B, T, out)`."""

    def __init__(self, weights, bias, activation=None, trainable=False):
        super().__init__()
        _register(self, "weights", weights, trainable)
        _register(self, "bias", bias, trainable)
        self.activation = activation

    def forward(self, x):
        w = _to(self.weights, x)
        b = _to(self.bias, x)
        e, batch, t, in_dim = x.shape
        out = torch.bmm(x.reshape(e, batch * t, in_dim), w)
        out = out.reshape(e, batch, t, -1) + b.view(e, 1, 1, -1)
        if self.activation is not None:
            out = self.activation(out)
        return out


class StackedRecurrentLayer(nn.Module):
    """`E` structurally-identical `RecurrentLayer`s run in lockstep: one
    `bmm` for the (precomputed, non-recurrent) input projection over all
    `T` at once, then one `bmm` per timestep for the recurrent term.
    Weights `(E, in, out)`, `recurrent_weights` `(E, out, out)`, `init`
    `(E, out)`."""

    def __init__(self, weights, bias, recurrent_weights, init,
                 activation=None, trainable=False):
        super().__init__()
        _register(self, "weights", weights, trainable)
        _register(self, "bias", bias, trainable)
        _register(self, "recurrent_weights", recurrent_weights, trainable)
        _register(self, "init", init, trainable=False)
        self.activation = activation

    def forward(self, x):
        w = _to(self.weights, x)
        b = _to(self.bias, x)
        rw = _to(self.recurrent_weights, x)
        init = _to(self.init, x)
        e, batch, t, in_dim = x.shape
        out_dim = w.shape[-1]
        ff = torch.bmm(x.reshape(e, batch * t, in_dim), w)
        ff = ff.reshape(e, batch, t, out_dim) + b.view(e, 1, 1, out_dim)
        prev = init.view(e, 1, out_dim).expand(e, batch, out_dim)
        outs = []
        for step in range(t):
            cur = ff[:, :, step, :] + torch.bmm(prev, rw)
            if self.activation is not None:
                cur = self.activation(cur)
            outs.append(cur)
            prev = cur
        return torch.stack(outs, dim=2)


class StackedLSTMLayer(nn.Module):
    """`E` structurally-identical `LSTMLayer`s run in lockstep with BOTH
    the ensemble dimension and the 4 gates (input/forget/cell/output)
    fused: one batched input-projection `bmm` and one batched
    recurrent-projection `bmm` per timestep (not `E * 4 * 2`). Peephole
    terms (elementwise, cheap) are applied per-gate after splitting the
    fused pre-activation, in the same order as `layers.LSTMLayer`: ig/fg
    peek the PREVIOUS cell state, og peeks the NEW (just-updated) state.

    `w_ih`/`b_ih`/`w_hh` concatenate the 4 gates' own weights/bias along
    the output axis, in the fixed order `(ig, fg, cell, og)` -- built by
    `convert.py`'s `_stack_lstm`, split back out here via `.split(hidden,
    dim=-1)`.
    """

    def __init__(self, w_ih, b_ih, w_hh, hidden,
                 peep_i, peep_f, peep_o, init, cell_init,
                 act_i, act_f, act_c, act_o, act_out, trainable=False):
        super().__init__()
        _register(self, "w_ih", w_ih, trainable)
        _register(self, "b_ih", b_ih, trainable)
        _register(self, "w_hh", w_hh, trainable)
        self.hidden = hidden
        self.has_peephole = peep_i is not None
        if self.has_peephole:
            _register(self, "peep_i", peep_i, trainable)
            _register(self, "peep_f", peep_f, trainable)
            _register(self, "peep_o", peep_o, trainable)
        _register(self, "init", init, trainable=False)
        _register(self, "cell_init", cell_init, trainable=False)
        self.act_i, self.act_f, self.act_c, self.act_o, self.act_out = (
            act_i, act_f, act_c, act_o, act_out
        )

    def forward(self, x):
        e, batch, t, in_dim = x.shape
        h = self.hidden
        w_ih = _to(self.w_ih, x)
        b_ih = _to(self.b_ih, x)
        w_hh = _to(self.w_hh, x)
        ff = torch.bmm(x.reshape(e, batch * t, in_dim), w_ih)
        ff = ff.reshape(e, batch, t, 4 * h) + b_ih.view(e, 1, 1, 4 * h)
        prev = _to(self.init, x).view(e, 1, h).expand(e, batch, h)
        state = _to(self.cell_init, x).view(e, 1, h).expand(e, batch, h)
        if self.has_peephole:
            peep_i = _to(self.peep_i, x).view(e, 1, h)
            peep_f = _to(self.peep_f, x).view(e, 1, h)
            peep_o = _to(self.peep_o, x).view(e, 1, h)
        outs = []
        for step in range(t):
            pre = ff[:, :, step, :] + torch.bmm(prev, w_hh)
            ig_pre, fg_pre, cell_pre, og_pre = pre.split(h, dim=-1)
            if self.has_peephole:
                ig_pre = ig_pre + state * peep_i
                fg_pre = fg_pre + state * peep_f
            ig = self.act_i(ig_pre) if self.act_i is not None else ig_pre
            fg = self.act_f(fg_pre) if self.act_f is not None else fg_pre
            cell = self.act_c(cell_pre) if self.act_c is not None else cell_pre
            state = cell * ig + state * fg
            if self.has_peephole:
                og_pre = og_pre + state * peep_o
            og = self.act_o(og_pre) if self.act_o is not None else og_pre
            out_t = self.act_out(state) * og if self.act_out is not None else state * og
            outs.append(out_t)
            prev = out_t
        return torch.stack(outs, dim=2)


class StackedGRULayer(nn.Module):
    """`E` structurally-identical `GRULayer`s run in lockstep with the
    ensemble dimension AND the reset/update gate pair fused (one batched
    input-projection `bmm` + one batched recurrent-projection `bmm` per
    timestep for the gate pair). The cell's recurrent term stays separate
    (it's multiplied by the reset gate's OUTPUT, so it can't be fused with
    the reset/update projection itself -- exact numpy formula: `tanh(xW +
    b + rg * (prev @ Wrec))`), but its own `x @ W + b` term is
    precomputed for every timestep at once outside the loop too (it
    doesn't depend on recurrence either).
    """

    def __init__(self, w_ru, b_ru, rw_ru, w_c, b_c, rw_c, hidden, init,
                 act_r, act_u, act_c, trainable=False):
        super().__init__()
        _register(self, "w_ru", w_ru, trainable)
        _register(self, "b_ru", b_ru, trainable)
        _register(self, "rw_ru", rw_ru, trainable)
        _register(self, "w_c", w_c, trainable)
        _register(self, "b_c", b_c, trainable)
        _register(self, "rw_c", rw_c, trainable)
        self.hidden = hidden
        _register(self, "init", init, trainable=False)
        self.act_r, self.act_u, self.act_c = act_r, act_u, act_c

    def forward(self, x):
        e, batch, t, in_dim = x.shape
        h = self.hidden
        w_ru = _to(self.w_ru, x)
        b_ru = _to(self.b_ru, x)
        rw_ru = _to(self.rw_ru, x)
        w_c = _to(self.w_c, x)
        b_c = _to(self.b_c, x)
        rw_c = _to(self.rw_c, x)
        ff_ru = torch.bmm(x.reshape(e, batch * t, in_dim), w_ru)
        ff_ru = ff_ru.reshape(e, batch, t, 2 * h) + b_ru.view(e, 1, 1, 2 * h)
        ff_c = torch.bmm(x.reshape(e, batch * t, in_dim), w_c)
        ff_c = ff_c.reshape(e, batch, t, h) + b_c.view(e, 1, 1, h)
        prev = _to(self.init, x).view(e, 1, h).expand(e, batch, h)
        outs = []
        for step in range(t):
            pre_ru = ff_ru[:, :, step, :] + torch.bmm(prev, rw_ru)
            rg_pre, ug_pre = pre_ru.split(h, dim=-1)
            rg = self.act_r(rg_pre) if self.act_r is not None else rg_pre
            ug = self.act_u(ug_pre) if self.act_u is not None else ug_pre
            cell_pre = ff_c[:, :, step, :] + rg * torch.bmm(prev, rw_c)
            cell = self.act_c(cell_pre) if self.act_c is not None else cell_pre
            out_t = ug * cell + (1 - ug) * prev
            outs.append(out_t)
            prev = out_t
        return torch.stack(outs, dim=2)


class StackedBidirectionalLayer(nn.Module):
    """Wraps an inner `Stacked{Recurrent,LSTM,GRU}Layer` built with an
    ensemble dimension of `2 * E` (fwd/bwd interleaved: slot `2*i` is
    ensemble member `i`'s forward sub-layer, slot `2*i + 1` its backward
    sub-layer -- see `convert.py`'s `_stack_bidirectional`). Runs the
    forward AND backward directions of all `E` members in the SAME time
    loop (one call into `inner`), instead of `E` separate
    forward-then-backward passes.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        # x: (E, B, T, in)
        e = x.shape[0]
        bwd_in = torch.flip(x, dims=[2])
        stacked_in = torch.stack([x, bwd_in], dim=1)  # (E, 2, B, T, in)
        stacked_in = stacked_in.reshape(2 * e, *x.shape[1:])
        out = self.inner(stacked_in)  # (2E, B, T, out)
        out = out.reshape(e, 2, *out.shape[1:])
        fwd_out = out[:, 0]
        bwd_out = torch.flip(out[:, 1], dims=[2])
        return torch.cat([fwd_out, bwd_out], dim=-1)


class StackedEnsembleModule(nn.Module):
    """Top-level driver for a whole ensemble (or a single network treated
    as an ensemble of 1) once every layer has been converted to its
    Stacked* form: broadcasts the input across the ensemble dimension
    `E`, runs it through every stacked layer in sequence, and averages
    the final `(E, B, T, out)` result over `E` (mirrors
    `madmom_infer.ml.nn.average_predictions`'s sum/len -- identity for `E
    == 1`). Squeeze behavior at the end matches
    `convert.NeuralNetworkModule` exactly (drop all size-1 axes if the
    caller's input was unbatched, else keep the batch axis)."""

    def __init__(self, stacked_layers, num_networks, expected_unbatched_ndim=2):
        super().__init__()
        self.stacked_layers = nn.ModuleList(stacked_layers)
        self.num_networks = num_networks
        self.expected_unbatched_ndim = expected_unbatched_ndim

    def forward(self, x):
        if x.dim() == self.expected_unbatched_ndim:
            added_batch = True
            x = x.unsqueeze(0)
        elif x.dim() == self.expected_unbatched_ndim + 1:
            added_batch = False
        else:
            raise ValueError(
                f"StackedEnsembleModule: expected a tensor with "
                f"{self.expected_unbatched_ndim} dims (unbatched) or "
                f"{self.expected_unbatched_ndim + 1} dims (batched), got "
                f"{x.dim()} dims (shape {tuple(x.shape)})."
            )
        x = x.unsqueeze(0).expand(self.num_networks, *x.shape)
        for layer in self.stacked_layers:
            x = layer(x)
        x = x.mean(dim=0)
        if added_batch:
            x = x.squeeze(0)
            x = _squeeze_all(x)
        else:
            x = _squeeze_keep_batch(x)
        return x


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
