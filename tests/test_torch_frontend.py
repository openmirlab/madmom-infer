"""Tests for `madmom_infer.torch.audio.frontend` -- the Phase 3a differentiable
spectrogram frontend -- against the numpy backend it must stay parity-locked
to (CLAUDE.md's dual-backend philosophy).

`pytest.importorskip("torch")` at module scope: this whole file (and by
extension, this whole feature) skips cleanly on a torch-less install, so a
plain `uv run pytest` in a core-only environment stays green.

Two DIFFERENT numpy comparison baselines are used, deliberately, per
`madmom_infer/torch/audio/frontend.py`'s module docstring:

1. **float32**: the torch frontend built with `dtype=torch.float32` is
   compared directly against the REAL, shipped numpy processor chain
   (`_numpy_rnn_downbeat_preprocessor`, below -- the same
   `FramedSignalProcessor`/`ShortTimeFourierTransformProcessor`/
   `FilteredSpectrogramProcessor`/`LogarithmicSpectrogramProcessor`/
   `SpectrogramDifferenceProcessor` composition
   `RNNDownBeatProcessor.__init__` builds, minus the NN ensemble/DBN decode
   steps this Phase-3a frontend explicitly excludes -- reconstructed here
   without a network-downloaded model so this stays a plain,
   network-free test). At `float32`, numpy's hardcoded `STFT_DTYPE`/
   `FILTER_DTYPE` ceiling is no ceiling at all (both already float32/
   complex64), so this is a meaningful, tight comparison: BLAS/FFT
   reduction order differs between scipy's FFT and torch's FFT
   implementation (and across devices), not a port bug -- same root cause
   `tests/test_spectrogram.py`'s module header documents for its own
   `np.dot` BLAS-build comparison. Empirically, max abs diff across all
   cases in this file is ~2.3e-6. This file uses `np.testing.assert_allclose`
   (atol-dominated) rather than `assert_array_max_ulp`, DELIBERATELY
   DEPARTING from `test_spectrogram.py`'s ULP convention: the temporal-
   difference stage (`positive_diffs=True`) clamps many outputs to exactly
   or near-exactly 0, where a relative/ULP metric explodes for a tiny
   absolute difference (verified: a 1.8e-7 absolute diff near a
   clamped-near-zero value showed as ~89000x its own ULP) -- a well-known
   caveat of ULP/relative metrics on sparse, clamped data, not evidence of
   a bigger error. `atol` carries the real bound here; `rtol` is a loose
   secondary check.
2. **float64**: numpy's own classes CANNOT produce a genuine float64 output
   (`STFT_DTYPE=np.complex64`, `FILTER_DTYPE=np.float32` are hardcoded
   regardless of input dtype -- `madmom_infer/audio/stft.py`,
   `madmom_infer/audio/filters.py`), so comparing a float64 torch instance
   against them would just be measuring numpy's float32 rounding, not
   algorithmic correctness. `_numpy_float64_single_branch_reference` below
   is a bespoke, test-only harness that recomputes ONE branch's chain at
   float64 throughout, reusing the same numpy-computed window/filterbank-
   matrix/`diff_frames` values (just widened to float64) rather than
   reimplementing them. This is what the ~1e-12 tolerance is checked
   against.

Reads: madmom_infer.torch (frontend, guarded import), madmom_infer.audio.*
(the numpy reference chain), tests/fixtures/wavs (reused only for its
sample-rate-realistic durations -- signals here are synthetic, not loaded
from the fixture wavs, since this file needs signals whose exact float
sample values are known in Python for the bespoke float64 harness).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from madmom_infer.audio.filters import (  # noqa: E402
    A4,
    FMAX,
    FMIN,
    NORM_FILTERS,
    UNIQUE_FILTERS,
    LogarithmicFilterbank,
)
from madmom_infer.audio.signal import (  # noqa: E402
    FramedSignal,
    FramedSignalProcessor,
    Signal,
    signal_frame,
)
from madmom_infer.audio.spectrogram import (  # noqa: E402
    ADD,
    DIFF_RATIO,
    MUL,
    FilteredSpectrogramProcessor,
    LogarithmicSpectrogramProcessor,
    SpectrogramDifferenceProcessor,
)
from madmom_infer.audio.spectrogram import _diff_frames as _np_diff_frames  # noqa: E402
from madmom_infer.audio.stft import ShortTimeFourierTransformProcessor, fft_frequencies  # noqa: E402
from madmom_infer.processors import ParallelProcessor, SequentialProcessor  # noqa: E402
from madmom_infer.torch.audio.frontend import (  # noqa: E402
    SpectrogramFrontend,
    frame_signal,
    rnn_downbeat_frontend,
)

SAMPLE_RATE = 44100
FPS = 100
HOP_SIZE = SAMPLE_RATE / FPS  # 441.0

# frame_size/num_bands (bands-per-octave) pairs RNNDownBeatProcessor uses,
# see madmom_infer/features/downbeats.py:96-97
RNN_DOWNBEAT_BRANCHES = ((1024, 3), (2048, 6), (4096, 12))

# float32 cross-library (scipy.fft vs torch.fft) tolerance: atol-dominated,
# ~5x the worst observed (2.3e-6) -- see module docstring point 1 for why
# ULP/relative metrics are inappropriate here (diff-stage zero-clamping).
FLOAT32_ATOL = 1e-5
FLOAT32_RTOL = 1e-3

# float64-vs-float64 algorithmic tolerance (see module docstring, point 2)
FLOAT64_ATOL = 1e-10
FLOAT64_RTOL = 1e-10

# batching non-associativity tolerance (matmul/FFT batch-shape-dependent
# kernels, same root cause as the float32 cross-library tolerance above)
BATCH_ATOL = 1e-5
BATCH_RTOL = 1e-3


# ---------------------------------------------------------------------------
# synthetic signals: sine mix, noise, clicks, silence; multiple lengths
# including non-multiples of hop_size (441)
# ---------------------------------------------------------------------------
def _sine_mix(num_samples, sample_rate=SAMPLE_RATE, seed=0):
    t = np.arange(num_samples) / sample_rate
    rng = np.random.default_rng(seed)
    sig = np.zeros(num_samples, dtype=np.float64)
    for freq, amp in ((220.0, 0.5), (880.0, 0.25), (3300.0, 0.1)):
        phase = rng.uniform(0, 2 * np.pi)
        sig += amp * np.sin(2 * np.pi * freq * t + phase)
    return sig.astype(np.float32)


def _noise(num_samples, seed=1):
    rng = np.random.default_rng(seed)
    return (0.3 * rng.standard_normal(num_samples)).astype(np.float32)


def _clicks(num_samples, seed=2):
    sig = np.zeros(num_samples, dtype=np.float32)
    rng = np.random.default_rng(seed)
    num_clicks = max(1, num_samples // 4000)
    positions = rng.choice(num_samples, size=num_clicks, replace=False)
    sig[positions] = 1.0
    return sig


def _silence(num_samples):
    return np.zeros(num_samples, dtype=np.float32)


LENGTHS = (44100, 44100 + 200, 5000, 2 * 44100 - 37)

SIGNAL_BUILDERS = {
    "sine_mix": _sine_mix,
    "noise": _noise,
    "clicks": _clicks,
    "silence": _silence,
}


def _cases():
    for length in LENGTHS:
        for name, builder in SIGNAL_BUILDERS.items():
            yield f"{name}_{length}", builder, length


CASES = list(_cases())
CASE_IDS = [c[0] for c in CASES]


# ---------------------------------------------------------------------------
# numpy reference #1: the REAL shipped processor chain, network-free
# (RNNDownBeatProcessor's DSP cascade, minus the NN ensemble/DBN decode it
# needs a downloaded model for -- reconstructed here from the same
# processors classes, not a reimplementation; see
# madmom_infer/features/downbeats.py:89-107 for the object this mirrors)
# ---------------------------------------------------------------------------
def _numpy_rnn_downbeat_preprocessor():
    multi = ParallelProcessor([])
    for frame_size, bands in RNN_DOWNBEAT_BRANCHES:
        frames = FramedSignalProcessor(frame_size=frame_size, fps=FPS)
        stft_proc = ShortTimeFourierTransformProcessor()
        filt = FilteredSpectrogramProcessor(
            num_bands=bands, fmin=FMIN, fmax=FMAX, norm_filters=NORM_FILTERS
        )
        spec = LogarithmicSpectrogramProcessor(mul=MUL, add=ADD)
        diff = SpectrogramDifferenceProcessor(
            diff_ratio=DIFF_RATIO, positive_diffs=True, stack_diffs=np.hstack
        )
        multi.append(SequentialProcessor((frames, stft_proc, filt, spec, diff)))
    return SequentialProcessor((multi, np.hstack))


# ---------------------------------------------------------------------------
# numpy reference #2: bespoke float64 single-branch harness (see module
# docstring, point 2) -- reuses the real window/filterbank/diff_frames
# values, computed via the actual numpy code, just widened to float64
# instead of the hardcoded complex64/float32 the shipped classes force.
# ---------------------------------------------------------------------------
def _numpy_float64_single_branch_reference(
    waveform_f64, frame_size=2048, fps=FPS, num_bands=12, sample_rate=SAMPLE_RATE,
):
    hop_size = float(sample_rate) / float(fps)
    sig = Signal(waveform_f64, sample_rate=sample_rate)
    framed = FramedSignal(sig, frame_size=frame_size, hop_size=hop_size, fps=None)
    num_frames = framed.num_frames

    # frames, reused via signal_frame (float64 preserved, no forced cast)
    frames = np.stack(
        [signal_frame(sig.data, i, frame_size, hop_size, origin=framed.origin)
         for i in range(num_frames)]
    ).astype(np.float64)

    window = np.hanning(frame_size).astype(np.float64)  # reused: np.hanning
    windowed = frames * window
    num_fft_bins = frame_size >> 1
    spectrum = np.fft.fft(windowed, n=frame_size, axis=-1)[:, :num_fft_bins]
    magnitude = np.abs(spectrum)

    bin_frequencies = fft_frequencies(num_fft_bins, sample_rate)  # reused
    filterbank = LogarithmicFilterbank(  # reused: real filter construction
        bin_frequencies, num_bands=num_bands, fmin=FMIN, fmax=FMAX, fref=A4,
        norm_filters=NORM_FILTERS, unique_filters=UNIQUE_FILTERS,
    )
    fb64 = np.asarray(filterbank, dtype=np.float64)
    filtered = magnitude @ fb64
    logspec = np.log10(MUL * filtered + ADD)

    diff_frames = _np_diff_frames(  # reused: real diff_frames formula
        DIFF_RATIO, hop_size=hop_size, frame_size=frame_size, window=np.hanning
    )
    diff = np.zeros_like(logspec)
    if diff_frames < num_frames:
        diff[diff_frames:] = logspec[diff_frames:] - logspec[:-diff_frames]
    diff = np.maximum(diff, 0)
    return np.hstack([logspec, diff])


# ---------------------------------------------------------------------------
# framing correctness: torch frame_signal vs FramedSignal.__getitem__,
# frame by frame (see frontend.py's module docstring)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("origin", [0, "left", "right"])
def test_frame_signal_matches_framed_signal_getitem(length, origin):
    sig = _sine_mix(length).astype(np.float64)
    numpy_framed = FramedSignal(sig, frame_size=2048, hop_size=HOP_SIZE, origin=origin)
    expected = np.stack([numpy_framed[i] for i in range(numpy_framed.num_frames)])

    frames = frame_signal(
        torch.from_numpy(sig), frame_size=2048, hop_size=HOP_SIZE, origin=origin
    )
    assert frames.shape == expected.shape
    np.testing.assert_array_equal(frames.numpy(), expected)


# ---------------------------------------------------------------------------
# parity #1: torch float32 vs the real shipped numpy processor chain
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case_name,builder,length", CASES, ids=CASE_IDS)
def test_float32_parity_against_numpy_processor_chain(case_name, builder, length):
    waveform = builder(length)  # float32

    numpy_chain = _numpy_rnn_downbeat_preprocessor()
    numpy_out = np.asarray(numpy_chain(Signal(waveform, sample_rate=SAMPLE_RATE)))

    frontend = rnn_downbeat_frontend(dtype=torch.float32)
    with torch.no_grad():
        torch_out = frontend(torch.from_numpy(waveform)[None, :])[0].numpy()

    assert torch_out.shape == numpy_out.shape
    np.testing.assert_allclose(
        torch_out, numpy_out, atol=FLOAT32_ATOL, rtol=FLOAT32_RTOL
    )


# ---------------------------------------------------------------------------
# parity #2: torch float64 vs the bespoke float64 numpy harness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case_name,builder,length", CASES, ids=CASE_IDS)
def test_float64_parity_against_bespoke_numpy_reference(case_name, builder, length):
    waveform_f32 = builder(length)
    waveform_f64 = waveform_f32.astype(np.float64)

    numpy_out = _numpy_float64_single_branch_reference(waveform_f64)

    frontend = SpectrogramFrontend(
        sample_rate=SAMPLE_RATE, frame_size=2048, fps=FPS, num_bands=12,
        fmin=FMIN, fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
        unique_filters=UNIQUE_FILTERS, log_mul=MUL, log_add=ADD,
        diff_ratio=DIFF_RATIO, positive_diffs=True, include_diff=True,
        dtype=torch.float64,
    )
    with torch.no_grad():
        torch_out = frontend(torch.from_numpy(waveform_f64)[None, :])[0].numpy()

    assert torch_out.shape == numpy_out.shape
    np.testing.assert_allclose(
        torch_out, numpy_out, atol=FLOAT64_ATOL, rtol=FLOAT64_RTOL
    )


# ---------------------------------------------------------------------------
# differentiability: gradcheck on a small float64 instance
# ---------------------------------------------------------------------------
def test_gradcheck_small_float64_frontend():
    torch.manual_seed(0)
    frontend = SpectrogramFrontend(
        sample_rate=800, frame_size=32, fps=50, num_bands=2, fmin=50, fmax=300,
        include_diff=True, dtype=torch.float64,
    )
    # a small, non-silent signal: avoids the exact-zero-magnitude gradient
    # singularity inherent to |z| (not specific to this implementation --
    # torch.abs on a complex tensor has the same non-differentiable point at
    # z=0; picking a signal with no exact zero-crossing frame sidesteps it).
    waveform = 0.1 * torch.randn(2, 220, dtype=torch.float64) + 0.3
    waveform.requires_grad_(True)

    assert torch.autograd.gradcheck(frontend, (waveform,), eps=1e-6, atol=1e-4)


# ---------------------------------------------------------------------------
# batching: batched output equals per-item outputs stacked
# ---------------------------------------------------------------------------
def test_batching_matches_per_item_processing():
    torch.manual_seed(1)
    frontend = rnn_downbeat_frontend(dtype=torch.float32)
    waveform = torch.randn(3, 44100 + 137, dtype=torch.float32)

    with torch.no_grad():
        batched = frontend(waveform)
        per_item = torch.stack([frontend(waveform[i:i + 1])[0] for i in range(3)])

    # NOT bit-exact: batched vs single-item matmul/FFT calls can take
    # different vectorized code paths at float32 (empirically, max abs diff
    # ~4.8e-7 here) -- the same batch-shape-dependent non-associativity
    # documented for the float32 numpy-parity tolerance above, not a bug.
    torch.testing.assert_close(batched, per_item, atol=BATCH_ATOL, rtol=BATCH_RTOL)


# ---------------------------------------------------------------------------
# device: cpu + cuda-if-available
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="no CUDA device available"
        ),
    )],
)
def test_runs_on_device(device):
    torch.manual_seed(2)
    frontend = rnn_downbeat_frontend(dtype=torch.float32).to(device)
    waveform = torch.randn(2, 44100, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = frontend(waveform)
    assert out.device.type == device
    assert out.shape == (2, 100, 314)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# error handling: dtype/shape mismatches raise clearly, not silently
# ---------------------------------------------------------------------------
def test_wrong_ndim_raises():
    frontend = SpectrogramFrontend(dtype=torch.float32)
    with pytest.raises(ValueError, match="batch, num_samples"):
        frontend(torch.randn(44100))


def test_dtype_mismatch_raises():
    frontend = SpectrogramFrontend(dtype=torch.float32)
    with pytest.raises(TypeError, match="dtype"):
        frontend(torch.randn(2, 44100, dtype=torch.float64))
