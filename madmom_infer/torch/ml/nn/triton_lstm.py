"""Optional Triton inference kernel for stacked peephole LSTMs.

This module owns the complete eligibility boundary for the fused recurrent
path. It retains cuBLAS ``torch.bmm`` for the recurrent projection and fuses
only the gate activations, peephole terms, and state updates which otherwise
become many small CUDA launches per timestep. Unsupported environments return
``None`` and let ``StackedLSTMLayer`` execute its differentiable eager path.

Triton is deliberately a soft runtime capability, not a package dependency.
Import and compilation failures are cached for the process so an unavailable
accelerator is never retried inside an inference loop. The fast path is limited
to CUDA float32, no-grad execution, standard sigmoid/tanh peephole LSTMs.

Reads: torch; optionally triton; read by:
madmom_infer/torch/ml/nn/stacked.py.
"""

from __future__ import annotations

import torch

try:  # Triton is not shipped on every platform supported by torch.
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except (ImportError, ModuleNotFoundError):  # pragma: no cover - platform dependent
    triton = None
    tl = None
    libdevice = None


_triton_failed = False


if triton is not None:

    @triton.jit
    def _lstm_gates_kernel(
        ff, recurrent, state, peep_i, peep_f, peep_o,
        output, next_prev, next_state,
        timestep, batch: tl.constexpr, frames: tl.constexpr,
        hidden: tl.constexpr, block: tl.constexpr,
    ):
        eb = tl.program_id(0)
        ensemble = eb // batch
        idx = tl.arange(0, block)
        mask = idx < hidden
        ff_base = (eb * frames + timestep) * (4 * hidden)
        recurrent_base = eb * (4 * hidden)
        state_base = eb * hidden
        peep_base = ensemble * hidden

        old_state = tl.load(state + state_base + idx, mask=mask, other=0.0)
        pi = tl.load(peep_i + peep_base + idx, mask=mask, other=0.0)
        pf = tl.load(peep_f + peep_base + idx, mask=mask, other=0.0)
        po = tl.load(peep_o + peep_base + idx, mask=mask, other=0.0)

        ig = tl.sigmoid(
            tl.load(ff + ff_base + idx, mask=mask, other=0.0)
            + tl.load(recurrent + recurrent_base + idx,
                      mask=mask, other=0.0)
            + old_state * pi
        )
        fg = tl.sigmoid(
            tl.load(ff + ff_base + hidden + idx, mask=mask, other=0.0)
            + tl.load(recurrent + recurrent_base + hidden + idx,
                      mask=mask, other=0.0)
            + old_state * pf
        )
        cell = libdevice.tanh(
            tl.load(ff + ff_base + 2 * hidden + idx,
                    mask=mask, other=0.0)
            + tl.load(recurrent + recurrent_base + 2 * hidden + idx,
                      mask=mask, other=0.0)
        )
        new_state = cell * ig + old_state * fg
        og = tl.sigmoid(
            tl.load(ff + ff_base + 3 * hidden + idx,
                    mask=mask, other=0.0)
            + tl.load(recurrent + recurrent_base + 3 * hidden + idx,
                      mask=mask, other=0.0)
            + new_state * po
        )
        new_output = libdevice.tanh(new_state) * og

        tl.store(next_state + state_base + idx, new_state, mask=mask)
        tl.store(next_prev + state_base + idx, new_output, mask=mask)
        tl.store(output + (eb * frames + timestep) * hidden + idx,
                 new_output, mask=mask)


def _eligible(layer, x):
    """Return whether ``layer``/``x`` satisfy the entire fast-path contract."""
    if triton is None or _triton_failed:
        return False
    if torch.is_grad_enabled() or x.device.type != "cuda" or x.dtype != torch.float32:
        return False
    if x.ndim != 4 or not layer.has_peephole:
        return False
    if not (
        layer.act_i is torch.sigmoid
        and layer.act_f is torch.sigmoid
        and layer.act_c is torch.tanh
        and layer.act_o is torch.sigmoid
        and layer.act_out is torch.tanh
    ):
        return False
    e, _batch, _frames, in_dim = x.shape
    h = layer.hidden
    tensors = (
        layer.w_ih, layer.b_ih, layer.w_hh, layer.peep_i, layer.peep_f,
        layer.peep_o, layer.init, layer.cell_init,
    )
    if any(tensor.device != x.device or tensor.dtype != x.dtype
           for tensor in tensors):
        return False
    return (
        layer.w_ih.shape == (e, in_dim, 4 * h)
        and layer.b_ih.shape == (e, 4 * h)
        and layer.w_hh.shape == (e, h, 4 * h)
        and layer.peep_i.shape == (e, h)
        and layer.peep_f.shape == (e, h)
        and layer.peep_o.shape == (e, h)
        and layer.init.shape == (e, h)
        and layer.cell_init.shape == (e, h)
    )


def try_fast_lstm(layer, x):
    """Return a fused output tensor, or ``None`` to request eager fallback."""
    global _triton_failed

    if not _eligible(layer, x):
        return None

    e, batch, frames, in_dim = x.shape
    h = layer.hidden
    if frames == 0:
        return x.new_empty((e, batch, 0, h))

    try:
        ff = torch.bmm(x.reshape(e, batch * frames, in_dim), layer.w_ih)
        ff = ff.reshape(e, batch, frames, 4 * h)
        ff = ff + layer.b_ih.view(e, 1, 1, 4 * h)

        # clone() is load-bearing for batch == 1: contiguous() would retain
        # a view of the registered initial-state buffer and double buffering
        # would mutate the model on every second timestep.
        prev = layer.init.view(e, 1, h).expand(e, batch, h).clone()
        state = layer.cell_init.view(e, 1, h).expand(e, batch, h).clone()
        output = torch.empty(
            (e, batch, frames, h), device=x.device, dtype=x.dtype
        )
        recurrent = torch.empty(
            (e, batch, 4 * h), device=x.device, dtype=x.dtype
        )
        next_prev = torch.empty_like(prev)
        next_state = torch.empty_like(state)
        block = triton.next_power_of_2(h)

        for timestep in range(frames):
            torch.bmm(prev, layer.w_hh, out=recurrent)
            _lstm_gates_kernel[(e * batch,)](
                ff, recurrent, state, layer.peep_i, layer.peep_f,
                layer.peep_o, output, next_prev, next_state,
                timestep=timestep, batch=batch, frames=frames, hidden=h,
                block=block,
            )
            prev, next_prev = next_prev, prev
            state, next_state = next_state, state
        return output
    except Exception:  # pragma: no cover - depends on compiler/driver failure
        _triton_failed = True
        return None


def _reset_failure_for_tests():
    global _triton_failed
    _triton_failed = False
