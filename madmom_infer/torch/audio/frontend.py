"""Differentiable spectrogram frontend -- torch reimplementation of the tensor
OPERATIONS in `madmom_infer.audio.{signal,stft,filters,spectrogram}`'s
framing -> STFT -> filterbank -> log-compress -> temporal-diff chain, batched
over `(B, num_samples)` waveforms and autograd-differentiable end-to-end.

**Single-owner-of-DSP-knowledge discipline (hard constraint, see
CLAUDE.md/pyproject.toml)**: this module does NOT reimplement filterbank
construction, window generation, bin-frequency computation, or the
temporal-difference frame-count formula. Every one of those is computed by
calling the existing numpy code and converting the result to a tensor:

- the analysis window comes from `np.hanning` (the same default
  `ShortTimeFourierTransformProcessor(window=np.hanning)` uses);
- the filterbank matrix comes from `madmom_infer.audio.filters.
  LogarithmicFilterbank`, built from bin frequencies computed by
  `madmom_infer.audio.stft.fft_frequencies`;
- `diff_frames` (the temporal-difference lag) comes from
  `madmom_infer.audio.spectrogram._diff_frames`;
- the number of frames a signal of a given length produces, and the
  numeric translation of a literal `origin` (`'center'`/`'left'`/`'right'`
  etc.), come from actually constructing a
  `madmom_infer.audio.signal.FramedSignal` over a same-length dummy array
  and reading its `.num_frames`/`.origin`/`.hop_size` -- not a
  reimplemented copy of that arithmetic.

Only the per-frame START OFFSET formula (`ref_sample = floor(index *
hop_size)`, `start = ref_sample - frame_size // 2 - origin`) is
re-expressed here, because it has to run as a vectorized index computation
for the torch gather below -- `madmom_infer.audio.signal.signal_frame`'s
python-loop version cannot be called per-frame without an O(num_frames)
Python loop. This is the one place this module re-derives (not reuses) a
numpy-side formula; `tests/test_torch_frontend.py`'s
`test_frame_signal_matches_framed_signal_getitem` cross-checks every frame
this produces against `FramedSignal.__getitem__` directly, frame by frame,
so any future drift between the two is caught immediately rather than
trusted by inspection.

**Precision, and why it deliberately does NOT mirror `STFT_DTYPE`/
`FILTER_DTYPE`.** The numpy backend hardcodes `complex64` STFT output and
`float32` filter coefficients (`madmom_infer/audio/stft.py`'s `STFT_DTYPE`,
`madmom_infer/audio/filters.py`'s `FILTER_DTYPE`) regardless of the input
signal's own dtype -- this keeps madmom's original output bit-identical,
including its float32-precision ceiling. This torch frontend instead runs
entirely in whatever dtype its buffers were built with (`dtype=` at
construction, `torch.float32` or `torch.float64`): a `float64` instance
computes the whole chain (window, filterbank matrix, FFT, magnitude,
matmul, log, diff) in double precision throughout, which is what makes
`torch.autograd.gradcheck` (which needs float64 for its finite-difference
comparison to be numerically meaningful) and a true "algorithm-exact"
float64 numpy comparison possible in the first place -- seeing the
hardcoded-float32 numpy classes as the only baseline would make a
~1e-12-tolerance comparison meaningless (see `tests/test_torch_frontend.py`
for the bespoke float64 numpy reference harness this is compared against,
and its docstring for why it is NOT simply `Spectrogram`/`FilteredSpectrogram`
/`LogarithmicSpectrogram` re-run). A `float32`-dtype instance of this
frontend, by contrast, IS directly comparable to the real, shipped numpy
`SequentialProcessor` chain, because at that precision the two coincide
(numpy's hardcoded ceiling is float32 too) -- see that same test file's
float32 parity tests, tolerance following `tests/test_spectrogram.py`'s
`assert_array_max_ulp` philosophy (BLAS/FFT reduction order differs across
libraries/devices, not a port bug).

**Assumed input**: a mono, float, already-resampled-to-`sample_rate`
waveform tensor shaped `(batch, num_samples)`. There is no int16-PCM
window-scaling special case here (`madmom_infer.audio.stft`'s module
header, trap 1) -- torch's natural audio representation is already
floating point, which is exactly the numpy path's "non-integer dtype: no
scaling needed" branch, so the two are directly comparable without extra
bookkeeping. Loading/downmixing/resampling (`madmom_infer.audio.signal.
SignalProcessor`) is out of scope for this frontend.

Excluded from Phase 3a entirely (see `madmom_infer/torch/__init__.py`):
Viterbi/DBN decoding (sequential, discrete -- no torch benefit) and the RNN
ensemble forward pass (madmom's LSTM layers have peephole connections
`torch.nn.LSTM` does not implement; a torch NN backend needs a custom cell,
left as a possible Phase 3b, not started).

Reads: torch, numpy, madmom_infer.audio.signal (FramedSignal, dummy-length
trick for num_frames/origin), madmom_infer.audio.stft (fft_frequencies),
madmom_infer.audio.filters (LogarithmicFilterbank + constants),
madmom_infer.audio.spectrogram (_diff_frames + constants); read by:
madmom_infer/torch/__init__.py, madmom_infer/torch/audio/__init__.py,
tests/test_torch_frontend.py
"""

import numpy as np
import torch
from torch import nn

from madmom_infer.audio.filters import (
    A4,
    FMAX,
    FMIN,
    NORM_FILTERS,
    NUM_BANDS,
    UNIQUE_FILTERS,
    LogarithmicFilterbank,
)
from madmom_infer.audio.signal import FramedSignal as _NumpyFramedSignal
from madmom_infer.audio.spectrogram import ADD, DIFF_RATIO, MUL, POSITIVE_DIFFS
from madmom_infer.audio.spectrogram import _diff_frames as _np_diff_frames
from madmom_infer.audio.stft import fft_frequencies

__all__ = [
    "SpectrogramFrontend",
    "apply_filterbank",
    "frame_signal",
    "log_compress",
    "rnn_downbeat_frontend",
    "stft",
    "temporal_difference",
]


# ---------------------------------------------------------------------------
# framing -- see module docstring for what is/isn't reused from numpy here
# ---------------------------------------------------------------------------
def _framing_plan(num_samples, frame_size, hop_size, origin, end):
    """Derive `(frame_size, hop_size, origin, num_frames)` via a real numpy
    `FramedSignal` over a same-length dummy array -- reuses `FramedSignal`'s
    own `origin`-literal translation and `num_frames` (ceil/floor 'normal'/
    'extend') formulas verbatim, rather than re-deriving them here.
    """
    dummy = _NumpyFramedSignal(
        np.empty(num_samples, dtype=np.float32),
        frame_size=frame_size, hop_size=hop_size, origin=origin, end=end,
    )
    return dummy.frame_size, dummy.hop_size, dummy.origin, dummy.num_frames


def _frame_index_map(num_samples, num_frames, frame_size, hop_size, origin):
    """Vectorized `(clipped_index, valid_mask)` arrays, shape `(num_frames,
    frame_size)`, for gathering frames out of a `(..., num_samples)` tensor.

    Re-expresses (does not reuse -- see module docstring) `madmom_infer.
    audio.signal.signal_frame`'s index arithmetic (`ref_sample = int(index *
    hop_size)`, `start = ref_sample - frame_size // 2 - origin`) in
    vectorized form, for the default (non-`'repeat'`) pad case: out-of-range
    positions are marked invalid (multiplied by 0 later) rather than
    repeating an edge sample -- the only pad mode `FramedSignal.__getitem__`
    ever uses (it never forwards a `pad=` argument).
    """
    frame_size = int(frame_size)
    t = np.arange(num_frames, dtype=np.float64)
    ref_sample = np.floor(t * float(hop_size)).astype(np.int64)
    start = ref_sample - (frame_size // 2) - int(origin)
    idx = start[:, None] + np.arange(frame_size, dtype=np.int64)[None, :]
    valid = (idx >= 0) & (idx < num_samples)
    idx_clipped = np.clip(idx, 0, max(num_samples - 1, 0))
    return idx_clipped, valid


def frame_signal(signal, frame_size, hop_size, origin=0, end="normal"):
    """Split `signal` (`..., num_samples`) into overlapping frames, matching
    `madmom_infer.audio.signal.FramedSignal`'s hop/origin semantics exactly
    (see module docstring). Returns a tensor shaped `(..., num_frames,
    frame_size)`, differentiable with respect to `signal`.
    """
    num_samples = signal.shape[-1]
    frame_size, hop_size, origin, num_frames = _framing_plan(
        num_samples, frame_size, hop_size, origin, end
    )
    idx, valid = _frame_index_map(num_samples, num_frames, frame_size, hop_size, origin)
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=signal.device)
    valid_t = torch.as_tensor(valid, dtype=signal.dtype, device=signal.device)
    frames = signal[..., idx_t] * valid_t
    return frames


# ---------------------------------------------------------------------------
# STFT
# ---------------------------------------------------------------------------
def _circular_shift(signal, frame_size, fft_size):
    """Port of `madmom_infer.audio.stft.stft`'s `circular_shift=True` branch,
    expressed as a concat instead of in-place buffer writes. Exact for the
    (only tested/used) `fft_size == frame_size`, even-`frame_size` case;
    inherits the same unspecified behavior upstream has for other
    combinations (see `madmom_infer/audio/stft.py`'s module header).
    """
    fft_shift = frame_size >> 1
    left = signal[..., fft_shift:]
    right = signal[..., :fft_shift]
    pad_len = fft_size - frame_size
    if pad_len > 0:
        middle = torch.zeros(
            *signal.shape[:-1], pad_len, dtype=signal.dtype, device=signal.device
        )
        return torch.cat([left, middle, right], dim=-1)
    return torch.cat([left, right], dim=-1)


def stft(frames, window, fft_size=None, circular_shift=False, include_nyquist=False):
    """Complex STFT of framed signal `frames` (`..., num_frames, frame_size`),
    matching `madmom_infer.audio.stft.stft`'s algorithm (window, fft size,
    circular shift). `window` is a real tensor of length `frame_size` (or
    `None`). Returns a complex tensor `(..., num_frames, num_bins)`,
    differentiable with respect to `frames` and `window`.
    """
    frame_size = frames.shape[-1]
    if fft_size is None:
        fft_size = frame_size
    num_fft_bins = fft_size >> 1
    if include_nyquist:
        num_fft_bins += 1

    signal = frames * window if window is not None else frames
    if circular_shift:
        signal = _circular_shift(signal, frame_size, fft_size)
        spectrum = torch.fft.fft(signal, dim=-1)
    else:
        spectrum = torch.fft.fft(signal, n=fft_size, dim=-1)
    return spectrum[..., :num_fft_bins]


# ---------------------------------------------------------------------------
# filterbank application / log compression / temporal difference
# ---------------------------------------------------------------------------
def apply_filterbank(spectrogram, filterbank):
    """`spectrogram @ filterbank`: `(..., num_bins)` x `(num_bins, num_bands)`
    -> `(..., num_bands)`, matching `FilteredSpectrogram`'s `np.dot`."""
    return torch.matmul(spectrogram, filterbank)


def log_compress(spectrogram, mul=MUL, add=ADD):
    """`log10(mul * spectrogram + add)`, matching `LogarithmicSpectrogram`."""
    return torch.log10(mul * spectrogram + add)


def temporal_difference(spectrogram, diff_frames, positive=POSITIVE_DIFFS):
    """Temporal first-order difference with `diff_frames` lag, full-length
    output (leading `diff_frames` rows are exactly 0) -- matches the net
    effect of `SpectrogramDifferenceProcessor.process`'s buffered,
    `reset=True`, `stack_diffs=None` call (see that class's docstring for
    the inf-buffer/`keep_dims=False` mechanics this reproduces the *result*
    of without the streaming-buffer machinery, which has no meaning for a
    whole-clip batched tensor call).
    """
    num_frames = spectrogram.shape[-2]
    if diff_frames >= num_frames:
        return torch.zeros_like(spectrogram)
    head_shape = list(spectrogram.shape)
    head_shape[-2] = diff_frames
    head = torch.zeros(*head_shape, dtype=spectrogram.dtype, device=spectrogram.device)
    tail = spectrogram[..., diff_frames:, :] - spectrogram[..., :num_frames - diff_frames, :]
    diff = torch.cat([head, tail], dim=-2)
    if positive:
        diff = torch.clamp_min(diff, 0)
    return diff


# ---------------------------------------------------------------------------
# SpectrogramFrontend: one frame-size branch of RNNDownBeatProcessor's
# pre-processing cascade, as a differentiable nn.Module
# ---------------------------------------------------------------------------
class SpectrogramFrontend(nn.Module):
    """Differentiable `frame -> STFT -> filterbank -> log -> [diff]` chain,
    equivalent to one `SequentialProcessor((FramedSignalProcessor,
    ShortTimeFourierTransformProcessor, FilteredSpectrogramProcessor,
    LogarithmicSpectrogramProcessor, [SpectrogramDifferenceProcessor]))`
    branch (see module docstring for precision/scope caveats).

    `forward(waveform)`: `waveform` is `(batch, num_samples)`, real, dtype
    matching this module's `dtype`. Returns `(batch, num_frames, num_bands)`
    if `include_diff=False`, or `(batch, num_frames, 2 * num_bands)` (log
    spectrogram concatenated with its temporal difference, matching
    `np.hstack` in `RNNDownBeatProcessor`) if `include_diff=True`.
    """

    def __init__(
        self, sample_rate=44100, frame_size=2048, fps=100, origin=0, end="normal",
        fft_size=None, circular_shift=False, include_nyquist=False,
        num_bands=NUM_BANDS, fmin=FMIN, fmax=FMAX, fref=A4,
        norm_filters=NORM_FILTERS, unique_filters=UNIQUE_FILTERS,
        log_mul=MUL, log_add=ADD, diff_ratio=DIFF_RATIO, diff_frames=None,
        positive_diffs=POSITIVE_DIFFS, include_diff=True,
        dtype=torch.float32,
    ):
        super().__init__()
        hop_size = float(sample_rate) / float(fps)
        self.sample_rate = sample_rate
        self.frame_size = int(frame_size)
        self.hop_size = hop_size
        self.origin = origin
        self.end = end
        self.fft_size = fft_size if fft_size is not None else self.frame_size
        self.circular_shift = circular_shift
        self.include_nyquist = include_nyquist
        self.log_mul = log_mul
        self.log_add = log_add
        self.positive_diffs = positive_diffs
        self.include_diff = include_diff

        # window: reused from numpy (np.hanning), not reimplemented
        window_np = np.hanning(self.frame_size)
        self.register_buffer("window", torch.as_tensor(window_np, dtype=dtype))

        # filterbank matrix + bin frequencies: reused from numpy
        num_fft_bins = self.fft_size >> 1
        if include_nyquist:
            num_fft_bins += 1
        bin_frequencies = fft_frequencies(num_fft_bins, sample_rate)
        filterbank = LogarithmicFilterbank(
            bin_frequencies, num_bands=num_bands, fmin=fmin, fmax=fmax, fref=fref,
            norm_filters=norm_filters, unique_filters=unique_filters,
        )
        self.num_bands = filterbank.num_bands
        self.register_buffer(
            "filterbank", torch.as_tensor(np.asarray(filterbank), dtype=dtype)
        )

        # diff_frames: reused from numpy (_diff_frames), not reimplemented
        if diff_frames is None:
            diff_frames = _np_diff_frames(
                diff_ratio, hop_size=hop_size, frame_size=self.frame_size,
                window=np.hanning,
            )
        self.diff_frames = int(diff_frames)

        self._index_cache = {}

    def _frame_indices(self, num_samples, device):
        key = (num_samples, device)
        cached = self._index_cache.get(key)
        if cached is not None:
            return cached
        frame_size, hop_size, origin, num_frames = _framing_plan(
            num_samples, self.frame_size, self.hop_size, self.origin, self.end
        )
        idx, valid = _frame_index_map(num_samples, num_frames, frame_size, hop_size, origin)
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
        valid_t = torch.as_tensor(valid, dtype=self.window.dtype, device=device)
        self._index_cache[key] = (idx_t, valid_t, num_frames)
        return idx_t, valid_t, num_frames

    def forward(self, waveform):
        if waveform.dim() != 2:
            raise ValueError(
                "waveform must be shaped (batch, num_samples), got %r"
                % (tuple(waveform.shape),)
            )
        if waveform.dtype != self.window.dtype:
            raise TypeError(
                "waveform dtype %s does not match this frontend's dtype %s"
                % (waveform.dtype, self.window.dtype)
            )
        idx_t, valid_t, _ = self._frame_indices(waveform.shape[-1], waveform.device)
        frames = waveform[:, idx_t] * valid_t

        spectrum = stft(
            frames, self.window, fft_size=self.fft_size,
            circular_shift=self.circular_shift, include_nyquist=self.include_nyquist,
        )
        magnitude = spectrum.abs()
        filtered = apply_filterbank(magnitude, self.filterbank)
        logspec = log_compress(filtered, mul=self.log_mul, add=self.log_add)

        if not self.include_diff:
            return logspec
        diff = temporal_difference(
            logspec, self.diff_frames, positive=self.positive_diffs
        )
        return torch.cat([logspec, diff], dim=-1)


# ---------------------------------------------------------------------------
# RNNDownBeatProcessor-equivalent multi-branch frontend
# ---------------------------------------------------------------------------
# The 3 frame-size/bands-per-octave branches below are exactly
# `madmom_infer.features.downbeats.RNNDownBeatProcessor.__init__`'s
# `frame_sizes = [1024, 2048, 4096]` / `num_bands = [3, 6, 12]` local
# variables (`madmom_infer/features/downbeats.py`) -- they are not module-
# level constants there (nothing to import), so they are restated here,
# cited to that exact call site rather than duplicated silently. Every
# OTHER parameter below (fmin/fmax/fref/norm_filters/unique_filters/log_mul/
# log_add/diff_ratio) is the actual shared numpy-side default constant
# (`FMIN`/`FMAX`/`A4`/`NORM_FILTERS`/`UNIQUE_FILTERS`/`MUL`/`ADD`/
# `DIFF_RATIO`), imported, not retyped -- except `positive_diffs`, which
# `RNNDownBeatProcessor` explicitly overrides to `True` (the module default
# `POSITIVE_DIFFS` is `False`), reproduced here as the same explicit
# literal, cited to that override.
_RNN_DOWNBEAT_FRAME_SIZES = (1024, 2048, 4096)
_RNN_DOWNBEAT_NUM_BANDS = (3, 6, 12)
_RNN_DOWNBEAT_POSITIVE_DIFFS = True  # downbeats.py:105 override, not POSITIVE_DIFFS default
_RNN_DOWNBEAT_FPS = 100  # downbeats.py:99, every branch
_RNN_DOWNBEAT_SAMPLE_RATE = 44100  # downbeats.py:94 (SignalProcessor)


class _RNNDownBeatSpectrogramFrontend(nn.Module):
    """3-branch `SpectrogramFrontend` stack + concat, differentiable
    equivalent of `RNNDownBeatProcessor`'s pre-processing cascade up to (but
    excluding) the RNN ensemble and DBN decoder -- see `rnn_downbeat_frontend`
    and the excluded-scope note in this module's docstring.
    """

    def __init__(self, sample_rate=_RNN_DOWNBEAT_SAMPLE_RATE, fps=_RNN_DOWNBEAT_FPS,
                 dtype=torch.float32):
        super().__init__()
        self.branches = nn.ModuleList([
            SpectrogramFrontend(
                sample_rate=sample_rate, frame_size=frame_size, fps=fps,
                num_bands=num_bands, fmin=FMIN, fmax=FMAX, fref=A4,
                norm_filters=NORM_FILTERS, unique_filters=UNIQUE_FILTERS,
                log_mul=MUL, log_add=ADD, diff_ratio=DIFF_RATIO,
                positive_diffs=_RNN_DOWNBEAT_POSITIVE_DIFFS, include_diff=True,
                dtype=dtype,
            )
            for frame_size, num_bands in zip(
                _RNN_DOWNBEAT_FRAME_SIZES, _RNN_DOWNBEAT_NUM_BANDS
            )
        ])

    def forward(self, waveform):
        return torch.cat([branch(waveform) for branch in self.branches], dim=-1)


def rnn_downbeat_frontend(
    sample_rate=_RNN_DOWNBEAT_SAMPLE_RATE, fps=_RNN_DOWNBEAT_FPS, dtype=torch.float32,
):
    """Build the differentiable pre-processing frontend equivalent to
    `madmom_infer.features.downbeats.RNNDownBeatProcessor`'s DSP cascade
    (everything up to, but excluding, the RNN ensemble and DBN decoder --
    see this module's docstring, "Excluded from Phase 3a entirely").

    `frontend(waveform)` where `waveform` is `(batch, num_samples)` mono
    float audio at `sample_rate` returns `(batch, num_frames, 314)`: the 3
    `(frame_size, bands_per_octave)` branches `(1024, 3)`, `(2048, 6)`,
    `(4096, 12)` build filterbanks with 21/45/91 TOTAL bands respectively
    (`num_bands` is bands PER OCTAVE, not a total count -- see
    `madmom_infer.audio.filters.LogarithmicFilterbank`'s docstring), each
    contributing `2 * total_bands` (log spectrogram + its temporal
    difference, `np.hstack` in `RNNDownBeatProcessor`):
    `2*21 + 2*45 + 2*91 = 42 + 90 + 182 = 314`, matching real madmom's
    known 314-dimensional `RNNDownBeatProcessor` feature vector.
    """
    return _RNNDownBeatSpectrogramFrontend(sample_rate=sample_rate, fps=fps, dtype=dtype)
