"""Golden-fixture tests for madmom_infer.audio.stft against real madmom.

Accuracy is the overriding rule for this port (CLAUDE.md, docs/DESIGN.md):
these tests assert BIT-IDENTICAL output (`np.array_equal` + exact dtype),
not float tolerance, against `tests/fixtures/stft.npz` -- recorded from a
real, compiled madmom 0.17.dev0 install (see `tests/fixtures/README.md`).

Covers: complex STFT of frame 0/1/last + a whole-array SHA-256 fingerprint
for every mono-compatible test wav (`stft.npz`'s 4 cases), the int16-window-
scaling convention flowing correctly through `frames.signal.dtype`, the
window-caching gotcha's exact reproduced-bug output (`ShortTimeFourierTransformProcessor.
fft_window`'s caching, module header of `madmom_infer/audio/stft.py`), and
the `ValueError` raised for a raw multi-channel `FramedSignal` (`stft`
requires 2D input, matching `tests/fixtures/manifest.json`'s
`known_error_cases.stereo_full_chain`).

Reads: madmom_infer/audio/{signal,stft}.py, tests/fixtures/stft.npz
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from madmom_infer.audio.signal import FramedSignalProcessor, Signal
from madmom_infer.audio.stft import ShortTimeFourierTransformProcessor

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
WAVS_DIR = FIXTURES_DIR / "wavs"

FRAME_SIZE = 2048
FPS = 100

CASES = {
    "mono_44100": ("mono_44100.wav", None),
    "stereo_44100_mono": ("stereo_44100.wav", 1),
    "stereo_48000_mono": ("stereo_48000.wav", 1),
    "float32_44100": ("float32_44100.wav", None),
}


def _assert_exact(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.dtype == expected.dtype, (
        "dtype mismatch: got %s, expected %s" % (actual.dtype, expected.dtype)
    )
    assert np.array_equal(actual, expected)


def _sha256_of_array(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@pytest.fixture(scope="module")
def stft_fixture():
    return np.load(FIXTURES_DIR / "stft.npz")


@pytest.fixture(scope="module")
def mono_frames():
    frames_proc = FramedSignalProcessor(frame_size=FRAME_SIZE, fps=FPS)
    out = {}
    for case, (wav_name, num_channels) in CASES.items():
        sig = Signal(str(WAVS_DIR / wav_name), num_channels=num_channels)
        out[case] = frames_proc(sig)
    return out


@pytest.mark.parametrize("case", sorted(CASES))
def test_stft_matches_fixture(case, mono_frames, stft_fixture):
    stft_proc = ShortTimeFourierTransformProcessor()
    stft_out = stft_proc(mono_frames[case])
    all_stft = np.asarray(stft_out)

    _assert_exact(all_stft[0], stft_fixture[f"{case}_stft_frame0"])
    _assert_exact(all_stft[1], stft_fixture[f"{case}_stft_frame1"])
    _assert_exact(all_stft[-1], stft_fixture[f"{case}_stft_frame_last"])
    assert _sha256_of_array(all_stft) == str(stft_fixture[f"{case}_stft_all_sha256"])


def test_stft_dtype_is_complex64(mono_frames):
    stft_out = ShortTimeFourierTransformProcessor()(mono_frames["mono_44100"])
    assert np.asarray(stft_out).dtype == np.complex64


def test_stft_bin_frequencies_shape_matches_num_bins(mono_frames):
    stft_out = ShortTimeFourierTransformProcessor()(mono_frames["mono_44100"])
    assert stft_out.bin_frequencies.shape == (stft_out.num_bins,)
    assert stft_out.num_bins == FRAME_SIZE // 2


def test_window_caching_gotcha_reproduces_exact_bug(mono_frames, stft_fixture):
    """Pinned-behavior test for the window-caching trap documented loudly in
    madmom_infer/audio/stft.py's module header: a REUSED
    ShortTimeFourierTransformProcessor instance silently keeps the first
    call's dtype-scaled window on a later, differently-dtyped call.
    """
    shared_stft = ShortTimeFourierTransformProcessor()
    # first call: int16 signal -- caches an int16-scaled window
    _ = shared_stft(mono_frames["mono_44100"])
    # second call, same instance, float32 signal -- BUG: stale int16-scaled
    # window is reused verbatim, not recomputed
    reused_output = np.asarray(shared_stft(mono_frames["float32_44100"]))

    fresh_stft = ShortTimeFourierTransformProcessor()
    fresh_output = np.asarray(fresh_stft(mono_frames["float32_44100"]))

    _assert_exact(reused_output, stft_fixture["window_caching_reused_output"])
    _assert_exact(fresh_output, stft_fixture["window_caching_fresh_output"])

    # sanity: confirm this is a REAL bug reproduction (a big divergence),
    # not an accidental/negligible float rounding difference
    max_abs_diff = np.abs(reused_output - fresh_output).max()
    expected_max_abs_diff = float(stft_fixture["window_caching_max_abs_diff"])
    assert np.isclose(max_abs_diff, expected_max_abs_diff, rtol=1e-5)
    assert max_abs_diff > 1.0, (
        "expected the window-caching bug to produce a large numeric "
        "divergence, not a rounding-level difference"
    )
    assert not np.array_equal(reused_output, fresh_output)


def test_stft_requires_mono_raises_valueerror_on_raw_stereo():
    """STFT requires a 2D (mono) FramedSignal; a raw (un-downmixed) stereo
    FramedSignal is 3D and must raise, matching
    tests/fixtures/manifest.json's known_error_cases.stereo_full_chain."""
    frames_proc = FramedSignalProcessor(frame_size=FRAME_SIZE, fps=FPS)
    sig = Signal(str(WAVS_DIR / "stereo_44100.wav"))
    framed = frames_proc(sig)
    assert framed.ndim == 3

    with pytest.raises(ValueError, match="frames must be a 2D array or iterable"):
        ShortTimeFourierTransformProcessor()(framed)
