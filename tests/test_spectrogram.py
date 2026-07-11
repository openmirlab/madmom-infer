"""Golden-fixture tests for madmom_infer.audio.spectrogram against real madmom.

**Read this before "fixing" any near-miss below.** Every stage up to and
including the STFT and the filterbank MATRIX construction is bit-identical
in this environment (see test_stft.py, test_filters.py) -- no tolerance
needed there. The stages THIS file tests (`FilteredSpectrogram`,
`LogarithmicSpectrogram`, and the full `SequentialProcessor` chain) all
depend on `np.dot(spectrogram, filterbank)`, a BLAS-backed (`sgemm`) matrix
multiply, and THIS is where bit-identity in the strict sense breaks down --
not because of a port bug, but because of a proven, root-caused, upstream
floating-point non-associativity issue:

**Root cause, empirically proven (not assumed).** `float32` matrix-multiply
accumulation order is implementation-defined and differs between BLAS
library builds. The golden fixtures were generated against numpy 1.23.5
linked to `openblas64_`; this project's own environment resolves to a
different OpenBLAS build via numpy's bundled `scipy-openblas` wheels. Proof:
exporting this port's own computed magnitude-spectrogram and filterbank-
matrix arrays (bit-identical to madmom's own, independently verified) and
running `np.dot` on them through the ORIGINAL reference venv's numpy/BLAS
(`all-in-one-fix/.venv`, numpy 1.23.5) reproduces the golden fixture with
ZERO differing elements -- see `test_filtered_spectrogram_algorithm_is_exact_
under_original_blas` below, which performs exactly this cross-environment
check as an executable test, not just a one-off investigation. Under THIS
environment's BLAS build, the same inputs differ from the fixture by at
most ~12 float32 ULPs (`np.testing.assert_array_max_ulp`, verified
empirically across all cases/frames) -- numerically negligible for any
downstream feature-extraction/neural-net consumer, but not bit-for-bit.

Per this project's golden-fixture philosophy (CLAUDE.md): shipping an
"approximately right" comparison silently, mislabeled as bit-identical,
would be worse than admitting the limit plainly. So: `test_filters.py`'s
filterbank-matrix and `test_stft.py`'s STFT assertions stay strict
`np.array_equal`. This file's post-matmul assertions use
`np.testing.assert_array_max_ulp(..., maxulp=64)` (4x the worst case
observed, documented margin) -- correctness-preserving, not tolerance-
hiding, given the proof above that the ALGORITHM is exact.

One additional, deliberate, DOCUMENTED departure from a literal port,
unrelated to BLAS: `Spectrogram`'s magnitude computation does not use plain
`np.abs(stft)` -- see `madmom_infer/audio/spectrogram.py`'s
`_stft_magnitude()` docstring for why (numpy's own `np.abs` on `complex64`
is not correctly rounded on every numpy build, verified via `mpmath`
cross-check; this port's explicit float64-then-cast computation IS,
matching what real madmom effectively produced).

A THIRD gotcha, found empirically while writing these tests (not called out
in the original task brief): `filterbank.npz`/`logspec.npz` were both
recorded by reusing ONE `FilteredSpectrogramProcessor` instance across the
44.1kHz and 48kHz cases, which hits the filterbank-caching bug documented in
`FilteredSpectrogramProcessor`'s docstring -- `filterbank_matrix_48000` is
actually a stale copy of the 44.1kHz filterbank, not a real 48kHz one. The
tests below replicate that exact call order to compare against the right
(bug-included) numbers; see test_filters.py for a dedicated pinned-behavior
test of the bug itself.

Reads: madmom_infer/audio/{signal,stft,filters,spectrogram}.py,
tests/fixtures/{filterbank,logspec,full_chain}.npz
"""

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest

from madmom_infer.audio.signal import FramedSignalProcessor, Signal
from madmom_infer.audio.spectrogram import (
    FilteredSpectrogramProcessor,
    LogarithmicSpectrogramProcessor,
)
from madmom_infer.audio.stft import ShortTimeFourierTransformProcessor
from madmom_infer.processors import SequentialProcessor

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
WAVS_DIR = FIXTURES_DIR / "wavs"

FRAME_SIZE = 2048
FPS = 100
NUM_BANDS = 12
FMIN = 30
FMAX = 17000
NORM_FILTERS = True
LOG_MUL = 1
LOG_ADD = 1

FILTERBANK_CHAIN_CASES = ("mono_44100", "stereo_48000_mono")

ALL_CASES = {
    "mono_44100": ("mono_44100.wav", None),
    "stereo_44100_mono": ("stereo_44100.wav", 1),
    "stereo_48000_mono": ("stereo_48000.wav", 1),
    "float32_44100": ("float32_44100.wav", None),
}

# generous (4x the worst observed) but not unlimited -- see module header
MAX_ULP = 64

REFERENCE_PYTHON = Path(
    "/home/worzpro/Desktop/dev/openmirlab/all-in-one-fix/.venv/bin/python"
)


def _sha256_of_array(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _mono_signal(wav_name, num_channels):
    return Signal(str(WAVS_DIR / wav_name), num_channels=num_channels)


def _framed(sig):
    return FramedSignalProcessor(frame_size=FRAME_SIZE, fps=FPS)(sig)


def build_spec_processor():
    frames = FramedSignalProcessor(frame_size=FRAME_SIZE, fps=FPS)
    stft = ShortTimeFourierTransformProcessor()
    filt = FilteredSpectrogramProcessor(
        num_bands=NUM_BANDS, fmin=FMIN, fmax=FMAX, norm_filters=NORM_FILTERS
    )
    spec = LogarithmicSpectrogramProcessor(mul=LOG_MUL, add=LOG_ADD)
    return SequentialProcessor([frames, stft, filt, spec])


@pytest.fixture(scope="module")
def filterbank_fixture():
    return np.load(FIXTURES_DIR / "filterbank.npz")


@pytest.fixture(scope="module")
def logspec_fixture():
    return np.load(FIXTURES_DIR / "logspec.npz")


@pytest.fixture(scope="module")
def full_chain_fixture():
    return np.load(FIXTURES_DIR / "full_chain.npz")


# ---------------------------------------------------------------------------
# FilteredSpectrogram (filterbank.npz) and LogarithmicSpectrogram
# (logspec.npz). NOTE: both fixtures were recorded by reusing ONE
# FilteredSpectrogramProcessor instance across BOTH FILTERBANK_CHAIN_CASES,
# in order (mono_44100 first, stereo_48000_mono second) -- see
# spectrogram.py's FilteredSpectrogramProcessor docstring for the resulting
# filterbank-caching gotcha (the stereo_48000_mono case's "filtered"/
# "logspec" fixture values were computed with the STALE 44.1kHz filterbank,
# not a correctly-built 48kHz one). These tests must replicate that exact
# call order/instance-reuse to compare against the right numbers -- a
# fresh-processor-per-case approach (as test_filters.py's dedicated pinned-
# behavior test demonstrates) would silently compare against the wrong
# (differently-shaped) filterbank.
# ---------------------------------------------------------------------------
def test_filtered_spectrogram_matches_fixture_within_blas_ulp(filterbank_fixture):
    frames_proc = FramedSignalProcessor(frame_size=FRAME_SIZE, fps=FPS)
    filt_proc = FilteredSpectrogramProcessor(
        num_bands=NUM_BANDS, fmin=FMIN, fmax=FMAX, norm_filters=NORM_FILTERS
    )
    for case in FILTERBANK_CHAIN_CASES:
        wav_name, num_channels = ALL_CASES[case]
        sig = _mono_signal(wav_name, num_channels)
        stft_out = ShortTimeFourierTransformProcessor()(frames_proc(sig))
        filtered = filt_proc(stft_out)
        all_filtered = np.asarray(filtered)

        assert (
            all_filtered.dtype == filterbank_fixture[f"{case}_filtered_frame0"].dtype
        )
        np.testing.assert_array_max_ulp(
            all_filtered[0], filterbank_fixture[f"{case}_filtered_frame0"],
            maxulp=MAX_ULP,
        )
        np.testing.assert_array_max_ulp(
            all_filtered[1], filterbank_fixture[f"{case}_filtered_frame1"],
            maxulp=MAX_ULP,
        )
        np.testing.assert_array_max_ulp(
            all_filtered[-1], filterbank_fixture[f"{case}_filtered_frame_last"],
            maxulp=MAX_ULP,
        )


def test_logarithmic_spectrogram_matches_fixture_within_blas_ulp(logspec_fixture):
    frames_proc = FramedSignalProcessor(frame_size=FRAME_SIZE, fps=FPS)
    filt_proc = FilteredSpectrogramProcessor(
        num_bands=NUM_BANDS, fmin=FMIN, fmax=FMAX, norm_filters=NORM_FILTERS
    )
    log_proc = LogarithmicSpectrogramProcessor(mul=LOG_MUL, add=LOG_ADD)
    for case in FILTERBANK_CHAIN_CASES:
        wav_name, num_channels = ALL_CASES[case]
        sig = _mono_signal(wav_name, num_channels)
        stft_out = ShortTimeFourierTransformProcessor()(frames_proc(sig))
        filtered = filt_proc(stft_out)
        logspec = log_proc(filtered)
        all_log = np.asarray(logspec)

        assert all_log.dtype == logspec_fixture[f"{case}_logspec_frame0"].dtype
        np.testing.assert_array_max_ulp(
            all_log[0], logspec_fixture[f"{case}_logspec_frame0"], maxulp=MAX_ULP
        )
        np.testing.assert_array_max_ulp(
            all_log[1], logspec_fixture[f"{case}_logspec_frame1"], maxulp=MAX_ULP
        )
        np.testing.assert_array_max_ulp(
            all_log[-1], logspec_fixture[f"{case}_logspec_frame_last"],
            maxulp=MAX_ULP,
        )


# ---------------------------------------------------------------------------
# Full chain: SequentialProcessor([frames, stft, filt, spec]) (full_chain.npz)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", sorted(ALL_CASES))
def test_full_chain_matches_fixture_within_blas_ulp(case, full_chain_fixture):
    wav_name, num_channels = ALL_CASES[case]
    sig = _mono_signal(wav_name, num_channels)
    chain = build_spec_processor()
    result = np.asarray(chain(sig))

    assert result.shape[0] == int(full_chain_fixture[f"{case}_num_frames"])
    np.testing.assert_array_max_ulp(
        result[0], full_chain_fixture[f"{case}_frame0"], maxulp=MAX_ULP
    )
    np.testing.assert_array_max_ulp(
        result[1], full_chain_fixture[f"{case}_frame1"], maxulp=MAX_ULP
    )
    np.testing.assert_array_max_ulp(
        result[-1], full_chain_fixture[f"{case}_frame_last"], maxulp=MAX_ULP
    )


def test_full_chain_stereo_raw_signal_raises_valueerror():
    """Feeding a raw (un-downmixed) stereo Signal through the standard chain
    must raise -- matches tests/fixtures/manifest.json's
    known_error_cases.stereo_full_chain (STFT requires mono input)."""
    chain = build_spec_processor()
    sig = Signal(str(WAVS_DIR / "stereo_44100.wav"))
    with pytest.raises(ValueError, match="frames must be a 2D array or iterable"):
        chain(sig)


# ---------------------------------------------------------------------------
# The proof: this port's algorithm (magnitude computation + filterbank
# construction) IS exact -- the only source of the above ULP-level slack is
# which BLAS library computes np.dot. Cross-checks by exporting this port's
# own intermediate arrays and running the matrix multiply through the
# ORIGINAL reference venv's numpy/BLAS build.
# ---------------------------------------------------------------------------
def _reference_python_available():
    return REFERENCE_PYTHON.exists()


@pytest.mark.skipif(
    not _reference_python_available(),
    reason="reference madmom install (all-in-one-fix/.venv) not found on "
           "this machine; the cross-BLAS proof requires it",
)
def test_filtered_spectrogram_algorithm_is_exact_under_original_blas(tmp_path):
    from madmom_infer.audio.filters import LogarithmicFilterbank
    from madmom_infer.audio.spectrogram import Spectrogram

    sig = _mono_signal("mono_44100.wav", None)
    framed = _framed(sig)
    stft_out = ShortTimeFourierTransformProcessor()(framed)
    spectrogram = Spectrogram(stft_out)
    fbank = LogarithmicFilterbank(
        spectrogram.bin_frequencies, num_bands=NUM_BANDS, fmin=FMIN, fmax=FMAX,
        norm_filters=NORM_FILTERS,
    )

    mag_path = tmp_path / "mag.npy"
    fb_path = tmp_path / "fb.npy"
    np.save(mag_path, np.asarray(spectrogram))
    np.save(fb_path, np.asarray(fbank))

    fixture_path = FIXTURES_DIR / "filterbank.npz"
    script = f"""
import numpy as np
mag = np.load({str(mag_path)!r})
fb = np.load({str(fb_path)!r})
expected = np.load({str(fixture_path)!r})["mono_44100_filtered_frame0"]
result = np.dot(mag, fb)[0]
assert np.array_equal(result, expected), (result, expected)
print("EXACT_MATCH")
"""
    proc = subprocess.run(
        [str(REFERENCE_PYTHON), "-c", script],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "EXACT_MATCH" in proc.stdout
