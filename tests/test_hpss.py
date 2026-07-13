"""Golden-fixture tests for `madmom_infer.audio.hpss` -- Wave 4g's port of
`madmom.audio.hpss.HarmonicPercussiveSourceSeparation`. Fixtures recorded
by `tools/generate_leftovers_fixtures.py` from real (compiled) madmom on a
real spectrogram of `mono_44100.wav`.

**`slices()`/`masks()` are proven EXACTLY equal (`np.array_equal`), both
in-process AND cross-BLAS** -- `scipy.ndimage.median_filter` and the
elementwise mask arithmetic touch no BLAS at all. `process()` is covered as
the usable composition of those helpers for both `Spectrogram` and array
inputs, including exact reconstruction under binary masking.

Reads: madmom_infer/audio/hpss.py, tests/fixtures/hpss.npz.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from madmom_infer.audio.hpss import HPSS, HarmonicPercussiveSourceSeparation
from madmom_infer.audio.spectrogram import SpectrogramProcessor
from madmom_infer.audio.signal import FramedSignalProcessor, SignalProcessor
from madmom_infer.audio.stft import ShortTimeFourierTransformProcessor

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parent.parent
WAV_PATH = FIXTURES_DIR / "wavs" / "mono_44100.wav"
REFERENCE_PYTHON = Path(
    "/home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python"
)


@pytest.fixture(scope="module")
def fixture():
    return np.load(FIXTURES_DIR / "hpss.npz")


@pytest.fixture(scope="module")
def spectrogram():
    sig = SignalProcessor(num_channels=1, sample_rate=44100)
    frames = FramedSignalProcessor(frame_size=2048, fps=100)
    stft = ShortTimeFourierTransformProcessor()
    spec = SpectrogramProcessor()
    return spec(stft(frames(sig(str(WAV_PATH)))))


def test_hpss_is_hpss_alias():
    assert HPSS is HarmonicPercussiveSourceSeparation


def test_masking_rejects_neither_binary_nor_float_gracefully():
    # 'binary' and None both mean binary masking; anything else is coerced
    # via float() -- a non-numeric string raises ValueError, matching
    # Python's own float() behavior (this class does no extra validation).
    h = HarmonicPercussiveSourceSeparation(masking="not-a-number")
    with pytest.raises(ValueError):
        h.masks(np.ones((2, 2)), np.ones((2, 2)))


def test_process_accepts_spectrogram(fixture, spectrogram):
    h = HarmonicPercussiveSourceSeparation()
    harmonic, percussive = h.process(spectrogram)
    np.testing.assert_array_equal(harmonic + percussive, np.asarray(spectrogram))


def test_process_accepts_array():
    h = HarmonicPercussiveSourceSeparation()
    data = np.ones((10, 10), dtype=np.float32)
    harmonic, percussive = h.process(data)
    np.testing.assert_array_equal(harmonic + percussive, data)


# ---------------------------------------------------------------------------
# real-madmom-fixture exactness (slices()/masks(), the actually-working
# surface)
# ---------------------------------------------------------------------------
def test_spectrogram_matches_fixture_exactly(fixture, spectrogram):
    np.testing.assert_array_equal(
        np.asarray(spectrogram), fixture["spectrogram_input"])


def test_slices_binary_masks_match_fixture_exactly(fixture, spectrogram):
    spec_arr = np.asarray(spectrogram)
    h = HarmonicPercussiveSourceSeparation(masking="binary")
    h_slice, p_slice = h.slices(spec_arr)
    np.testing.assert_array_equal(h_slice, fixture["harmonic_slice"])
    np.testing.assert_array_equal(p_slice, fixture["percussive_slice"])
    h_mask, p_mask = h.masks(h_slice, p_slice)
    np.testing.assert_array_equal(h_mask, fixture["harmonic_mask_binary"])
    np.testing.assert_array_equal(p_mask, fixture["percussive_mask_binary"])


def test_soft_mask_matches_fixture_exactly(fixture, spectrogram):
    spec_arr = np.asarray(spectrogram)
    binary = HarmonicPercussiveSourceSeparation(masking="binary")
    h_slice, p_slice = binary.slices(spec_arr)
    soft = HarmonicPercussiveSourceSeparation(masking=2.0)
    h_mask, p_mask = soft.masks(h_slice, p_slice)
    np.testing.assert_array_equal(h_mask, fixture["harmonic_mask_soft"])
    np.testing.assert_array_equal(p_mask, fixture["percussive_mask_soft"])


def test_custom_filter_sizes_match_fixture_exactly(fixture, spectrogram):
    spec_arr = np.asarray(spectrogram)
    h = HarmonicPercussiveSourceSeparation(
        harmonic_filter=(9, 1), percussive_filter=(1, 9))
    h_slice, p_slice = h.slices(spec_arr)
    np.testing.assert_array_equal(h_slice, fixture["harmonic_slice_custom"])
    np.testing.assert_array_equal(p_slice, fixture["percussive_slice_custom"])


# ---------------------------------------------------------------------------
# cross-BLAS exactness (completeness -- see module header)
# ---------------------------------------------------------------------------
def _reference_python_available():
    return REFERENCE_PYTHON.exists()


@pytest.mark.skipif(
    not _reference_python_available(),
    reason="reference madmom install (madmom-reference/.venv) not found on "
           "this machine",
)
def test_hpss_is_exact_under_original_blas():
    """This port's own `slices()`/`masks()`, run under the reference
    venv's numpy/scipy build, reproduce real madmom's fixture values with
    ZERO differing elements."""
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import numpy as np
from madmom_infer.audio.hpss import HarmonicPercussiveSourceSeparation
from madmom_infer.audio.spectrogram import SpectrogramProcessor
from madmom_infer.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom_infer.audio.stft import ShortTimeFourierTransformProcessor

fixture = np.load({str(FIXTURES_DIR / "hpss.npz")!r})
sig = SignalProcessor(num_channels=1, sample_rate=44100)
frames = FramedSignalProcessor(frame_size=2048, fps=100)
stft = ShortTimeFourierTransformProcessor()
spec_proc = SpectrogramProcessor()
spec = spec_proc(stft(frames(sig({str(WAV_PATH)!r}))))
spec_arr = np.asarray(spec)
assert np.array_equal(spec_arr, fixture["spectrogram_input"])

binary = HarmonicPercussiveSourceSeparation(masking="binary")
h_slice, p_slice = binary.slices(spec_arr)
assert np.array_equal(h_slice, fixture["harmonic_slice"])
assert np.array_equal(p_slice, fixture["percussive_slice"])
h_mask, p_mask = binary.masks(h_slice, p_slice)
assert np.array_equal(h_mask, fixture["harmonic_mask_binary"])
assert np.array_equal(p_mask, fixture["percussive_mask_binary"])

soft = HarmonicPercussiveSourceSeparation(masking=2.0)
h_mask_s, p_mask_s = soft.masks(h_slice, p_slice)
assert np.array_equal(h_mask_s, fixture["harmonic_mask_soft"])
assert np.array_equal(p_mask_s, fixture["percussive_mask_soft"])

custom = HarmonicPercussiveSourceSeparation(
    harmonic_filter=(9, 1), percussive_filter=(1, 9))
h_slice_c, p_slice_c = custom.slices(spec_arr)
assert np.array_equal(h_slice_c, fixture["harmonic_slice_custom"])
assert np.array_equal(p_slice_c, fixture["percussive_slice_custom"])
print("EXACT_MATCH")
"""
    proc = subprocess.run(
        [str(REFERENCE_PYTHON), "-c", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "EXACT_MATCH" in proc.stdout
