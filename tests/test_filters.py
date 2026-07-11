"""Golden-fixture tests for madmom_infer.audio.filters against real madmom.

Bit-identical assertions (`np.array_equal` + exact dtype) against
`tests/fixtures/filterbank.npz`'s `filterbank_matrix_{44100,48000}` --
the filterbank MATRIX construction is independent of any BLAS-backed
matrix multiply (it's built via `np.zeros`/`np.linspace`/`np.maximum`
placement only), so this stage is bit-identical in every environment tested
so far, unlike the filtered-spectrogram fixtures in test_spectrogram.py
(see that file's module header for why those need a different comparison).

Also pins the "num_bands means bands per octave" surprise documented in
tests/fixtures/README.md: `num_bands=12` does NOT mean a 12-band filterbank.

Reads: madmom_infer/audio/{signal,stft,filters}.py, tests/fixtures/filterbank.npz
"""

from pathlib import Path

import numpy as np
import pytest

from madmom_infer.audio.filters import LogarithmicFilterbank, log_frequencies
from madmom_infer.audio.signal import FramedSignalProcessor, Signal
from madmom_infer.audio.spectrogram import FilteredSpectrogramProcessor
from madmom_infer.audio.stft import ShortTimeFourierTransformProcessor

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
WAVS_DIR = FIXTURES_DIR / "wavs"

FRAME_SIZE = 2048
FPS = 100
NUM_BANDS = 12
FMIN = 30
FMAX = 17000
NORM_FILTERS = True


def _assert_exact(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected)


@pytest.fixture(scope="module")
def filterbank_fixture():
    return np.load(FIXTURES_DIR / "filterbank.npz")


def _stft_for(wav_name, num_channels=None):
    sig = Signal(str(WAVS_DIR / wav_name), num_channels=num_channels)
    framed = FramedSignalProcessor(frame_size=FRAME_SIZE, fps=FPS)(sig)
    return ShortTimeFourierTransformProcessor()(framed)


def test_filterbank_matrix_matches_fixture_at_44100(filterbank_fixture):
    """The 44.1kHz filterbank matrix is a plain, fresh `LogarithmicFilterbank`
    build -- bit-identical, independent of any BLAS matmul (this stage is
    pure `np.zeros`/`np.linspace`/`np.maximum` placement, see filters.py)."""
    stft_out = _stft_for("mono_44100.wav")
    fbank = LogarithmicFilterbank(
        stft_out.bin_frequencies, num_bands=NUM_BANDS, fmin=FMIN, fmax=FMAX,
        norm_filters=NORM_FILTERS,
    )
    _assert_exact(np.asarray(fbank), filterbank_fixture["filterbank_matrix_44100"])
    assert fbank.shape == (1024, 81)


def test_filtered_spectrogram_processor_caches_stale_filterbank_across_sample_rates(
    filterbank_fixture,
):
    """Pinned-behavior test for the filterbank-caching gotcha documented
    loudly in madmom_infer/audio/spectrogram.py's `FilteredSpectrogramProcessor`
    docstring (found while porting, not called out in the original task
    brief): `tests/fixtures/filterbank.npz`'s `filterbank_matrix_48000` is
    NOT actually a 48kHz filterbank -- it's the STALE 44.1kHz filterbank,
    because `tools/generate_fixtures.py`'s `generate_filterbank_fixtures()`
    reuses one `FilteredSpectrogramProcessor` instance across both the
    44.1kHz and 48kHz cases (in that order), and the processor only rebuilds
    its filterbank when given a CLASS, not an already-built instance.
    Reproduced here with the exact same call order/instance-reuse pattern.
    """
    frames_proc = FramedSignalProcessor(frame_size=FRAME_SIZE, fps=FPS)
    filt_proc = FilteredSpectrogramProcessor(
        num_bands=NUM_BANDS, fmin=FMIN, fmax=FMAX, norm_filters=NORM_FILTERS
    )

    sig_44100 = Signal(str(WAVS_DIR / "mono_44100.wav"))
    stft_44100 = ShortTimeFourierTransformProcessor()(frames_proc(sig_44100))
    filtered_44100 = filt_proc(stft_44100)

    sig_48000 = Signal(str(WAVS_DIR / "stereo_48000.wav"), num_channels=1)
    stft_48000 = ShortTimeFourierTransformProcessor()(frames_proc(sig_48000))
    filtered_48000 = filt_proc(stft_48000)

    # the bug: the SAME filterbank instance/matrix is (wrongly) reused
    assert filtered_48000.filterbank is filtered_44100.filterbank
    _assert_exact(
        np.asarray(filtered_48000.filterbank),
        filterbank_fixture["filterbank_matrix_48000"],
    )
    _assert_exact(
        filterbank_fixture["filterbank_matrix_48000"],
        filterbank_fixture["filterbank_matrix_44100"],
    )

    # sanity: a FRESH, correctly-built 48kHz filterbank has a DIFFERENT
    # shape (80 bands, not 81 -- log-spaced bin mapping differs at 48kHz)
    correct_48000 = LogarithmicFilterbank(
        stft_48000.bin_frequencies, num_bands=NUM_BANDS, fmin=FMIN, fmax=FMAX,
        norm_filters=NORM_FILTERS,
    )
    assert correct_48000.shape == (1024, 80)
    assert correct_48000.shape != filtered_48000.filterbank.shape


def test_num_bands_means_bands_per_octave_not_total_bands():
    """tests/fixtures/README.md's "Surprises": num_bands=12, fmin=30,
    fmax=17000 produces an 81-band filterbank at 44.1kHz, not a 12-band
    one."""
    stft_out = _stft_for("mono_44100.wav")
    fbank = LogarithmicFilterbank(
        stft_out.bin_frequencies, num_bands=NUM_BANDS, fmin=FMIN, fmax=FMAX,
        norm_filters=NORM_FILTERS,
    )
    assert fbank.num_bands == 81
    assert fbank.num_bands != NUM_BANDS
    assert fbank.shape == (stft_out.num_bins, 81)


def test_filterbank_bands_normalized_to_area_one_in_float32():
    """norm_filters=True normalizes each band's column to sum to 1 -- in
    float32 precision (see filters.py's module header bit-identity trap:
    normalization happens BEFORE any float64 widening, not after)."""
    stft_out = _stft_for("mono_44100.wav")
    fbank = LogarithmicFilterbank(
        stft_out.bin_frequencies, num_bands=NUM_BANDS, fmin=FMIN, fmax=FMAX,
        norm_filters=True,
    )
    matrix = np.asarray(fbank)
    sums = matrix.sum(axis=0)
    # every non-empty band should sum to (very close to) 1
    assert np.allclose(sums, 1.0, atol=1e-6)


def test_filterbank_without_norm_filters_not_area_one():
    stft_out = _stft_for("mono_44100.wav")
    fbank = LogarithmicFilterbank(
        stft_out.bin_frequencies, num_bands=NUM_BANDS, fmin=FMIN, fmax=FMAX,
        norm_filters=False,
    )
    matrix = np.asarray(fbank)
    sums = matrix.sum(axis=0)
    # un-normalized triangular filters have height 1 at the peak, so most
    # bands sum to > 1
    assert np.any(sums > 1.0001)


def test_log_frequencies_semitone_spacing_is_midi_like():
    # 12 bands per octave from A4 (440 Hz) should hit octaves of 440 exactly
    freqs = log_frequencies(12, 55, 3520, fref=440.0)
    assert np.any(np.isclose(freqs, 440.0))
    assert np.any(np.isclose(freqs, 880.0))
