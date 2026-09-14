"""Small, shared, autograd-safe helper `nn.Module`s used by more than one
pipeline in `madmom_infer/torch/features/pipelines.py`.

Each helper mirrors one numpy pre/post-processing function found across
`madmom_infer/features/*.py` (edge-repeat padding, zero padding, and
superframe averaging). They operate on `(B, T, F)` tensors (batch always
present -- callers add/remove it, same convention as
`madmom_infer.torch.ml.nn.layers`).

Context-frame stacking (`DeepChromaProcessor`'s `_dcp_flatten` +
`FramedSignalProcessor(frame_size=15, hop_size=1)`) and CNN-branch
stacking (`CNNOnsetProcessor`'s `np.dstack`) are NOT re-implemented here --
they reuse `madmom_infer.torch.ml.nn.layers.StrideLayer` (already does
exactly the window-then-flatten `segment_axis(15, 1)` shape) and
`madmom_infer.torch.ml.nn.layers.DstackModule` respectively.

Reads: torch, madmom_infer.torch.ml.nn.layers (StrideLayer's window-then-
rest unfold ordering, reused by `SuperframeAverage`); read by:
madmom_infer/torch/features/pipelines.py.
"""

from __future__ import annotations

import torch
from torch import nn


class EdgeRepeatPad(nn.Module):
    """Pad `width` frames on both ends of the time axis (dim=1) by
    repeating the first/last frame -- torch twin of
    `madmom_infer.features.onsets._cnn_onset_processor_pad` (width 7) and
    `madmom_infer.features.notes._cnn_pad` (width 5)."""

    def __init__(self, width):
        super().__init__()
        self.width = int(width)

    def forward(self, x):
        if self.width == 0:
            return x
        first = x[:, :1].expand(-1, self.width, *x.shape[2:])
        last = x[:, -1:].expand(-1, self.width, *x.shape[2:])
        return torch.cat([first, x, last], dim=1)


class ZeroPad(nn.Module):
    """Pad `width` all-zero frames on both ends of the time axis (dim=1) --
    torch twin of `madmom_infer.features.chords._cnncfp_pad` (width 11)."""

    def __init__(self, width):
        super().__init__()
        self.width = int(width)

    def forward(self, x):
        if self.width == 0:
            return x
        pad_shape = (x.shape[0], self.width, *x.shape[2:])
        pad = torch.zeros(*pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([pad, x, pad], dim=1)


class SuperframeAverage(nn.Module):
    """Segment the time axis (dim=1) into overlapping `block_size`-frame
    windows (hop 1) and average each window over the (window, feature)
    axes -- torch twin of `madmom_infer.features.chords._cnncfp_superframes`
    (`segment_axis(data, 3, 1, axis=0)`) composed with `_cnncfp_avg`
    (`data.mean((1, 2))`). Uses the same `Tensor.unfold`-based window
    construction as `madmom_infer.torch.ml.nn.layers.StrideLayer`, but
    averages instead of flattening.
    """

    def __init__(self, block_size):
        super().__init__()
        self.block_size = int(block_size)

    def forward(self, x):
        # x: (B, T, F, [C, ...]) -> unfold along time -> puts the window
        # axis last: (B, T', F, [C, ...], block). numpy's `.mean((1, 2))`
        # on `segment_axis`'s `(T', block, F, [C, ...])` output reduces the
        # window axis AND the first "rest" axis (F) while keeping any
        # further axes (e.g. channels) intact -- the torch equivalent is
        # reducing dim=2 (F) and dim=-1 (window, wherever unfold put it).
        unfolded = x.unfold(1, self.block_size, 1)
        return unfolded.mean(dim=(2, -1))


def ensure_batched(waveform, unbatched_ndim=1):
    """Return `(batched_waveform, was_unbatched)`: adds a leading batch
    dimension if `waveform` has `unbatched_ndim` dims, otherwise assumes
    it already carries one (`unbatched_ndim + 1` dims). Every pipeline's
    `forward(waveform)` uses this to accept both an unbatched `(N,)`
    waveform and a batched `(B, N)` one, mirroring
    `madmom_infer.torch.ml.nn.convert.NeuralNetworkModule`'s own
    batch-dispatch convention.
    """
    if waveform.dim() == unbatched_ndim:
        return waveform.unsqueeze(0), True
    if waveform.dim() == unbatched_ndim + 1:
        return waveform, False
    raise ValueError(
        f"expected a tensor with {unbatched_ndim} dims (unbatched) or "
        f"{unbatched_ndim + 1} dims (batched), got {waveform.dim()} dims "
        f"(shape {tuple(waveform.shape)})."
    )


def unbatch(x, was_unbatched):
    """Undo `ensure_batched`'s added batch dimension if it added one."""
    return x.squeeze(0) if was_unbatched else x
