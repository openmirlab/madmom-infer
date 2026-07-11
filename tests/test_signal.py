"""A/B tests for madmom_infer.audio.signal against real madmom.

Accuracy is the overriding rule for this port (docs/DESIGN.md, CLAUDE.md):
these tests assert BIT-IDENTICAL output (`np.array_equal` + exact dtype
match), not float tolerance, against a real madmom 0.17.dev0 install
(`all-in-one-fix/.venv`, Python 3.10.18, numpy 1.23.5, scipy 1.15.3).

Self-sufficient by design (per the task brief, this workstream does not
depend on any other workstream's fixture harness): this file writes its own
deterministic synthetic .wav fixtures into a pytest `tmp_path`, then runs
`reference_gen.py` (a standalone script with zero `madmom_infer` imports) as
a subprocess of the *other* venv's interpreter to compute real madmom's
outputs for those exact same files, and finally compares
`madmom_infer.audio.signal`'s in-process outputs against that reference.

Covers: frame 0 / frame 1 / last frame (origin + padding), frame count,
mono-downmix truncation on an odd-sum (x.5 mean) construction, 48kHz hop
computation, wav-file dtype flow-through (int16 stays int16, float32 stays
float32, no rescale), and the literal `origin` string translations.

Reads: madmom_infer/audio/signal.py, tests/reference_gen.py
"""

import subprocess

import numpy as np
import pytest
from scipy.io import wavfile

from madmom_infer.audio.signal import (
    FramedSignal,
    FramedSignalProcessor,
    Signal,
    remix,
)

REFERENCE_PYTHON = (
    "/home/worzpro/Desktop/dev/openmirlab/all-in-one-fix/.venv/bin/python"
)
REFERENCE_GEN_SCRIPT = str(
    __import__("pathlib").Path(__file__).parent / "reference_gen.py"
)


def _reference_python_available():
    return __import__("pathlib").Path(REFERENCE_PYTHON).exists()


pytestmark = pytest.mark.skipif(
    not _reference_python_available(),
    reason="reference madmom install (all-in-one-fix/.venv) not found on "
           "this machine; A/B comparison tests require it",
)


@pytest.fixture(scope="module")
def wav_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("signal_wavs")

    rng = np.random.default_rng(42)
    mono16 = rng.integers(-30000, 30000, size=5000, dtype=np.int16)
    wavfile.write(str(d / "mono16_44100.wav"), 44100, mono16)

    # crafted so every channel-pair sum is odd -> mean is always x.5,
    # covering both positive and negative truncate-vs-round divergence
    ch0 = np.arange(-1000, 1000, dtype=np.int16)
    ch1 = (ch0 + 1).astype(np.int16)
    stereo16 = np.stack([ch0, ch1], axis=-1)
    wavfile.write(str(d / "stereo16_44100.wav"), 44100, stereo16)

    mono16_48k = rng.integers(-30000, 30000, size=5000, dtype=np.int16)
    wavfile.write(str(d / "mono16_48000.wav"), 48000, mono16_48k)

    t = np.arange(2000) / 44100.0
    float32 = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wavfile.write(str(d / "mono_float32.wav"), 44100, float32)

    return d


@pytest.fixture(scope="module")
def reference(wav_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("signal_ref") / "ref.npz"
    subprocess.run(
        [REFERENCE_PYTHON, REFERENCE_GEN_SCRIPT, str(wav_dir), str(out)],
        check=True,
        capture_output=True,
        text=True,
    )
    return np.load(out)


def _assert_exact(actual, expected):
    """np.array_equal + exact dtype match -- bit-identity, not allclose."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.dtype == expected.dtype, (
        "dtype mismatch: got %s, expected %s" % (actual.dtype, expected.dtype)
    )
    assert np.array_equal(actual, expected)


# ---------------------------------------------------------------------------
# A. array-based cases (Signal-from-ndarray, FramedSignal origin/padding math)
# ---------------------------------------------------------------------------
def test_framed_signal_default_origin_frames(reference):
    sig = Signal(np.arange(1, 21, dtype=np.int16), sample_rate=10)
    frames = FramedSignal(sig, frame_size=6, hop_size=2, origin=0)
    assert frames.num_frames == int(reference["arr_center__num_frames"])
    assert frames.hop_size == float(reference["arr_center__hop_size"])
    _assert_exact(frames[0], reference["arr_center__frame0"])
    _assert_exact(frames[1], reference["arr_center__frame1"])
    _assert_exact(frames[frames.num_frames - 1], reference["arr_center__frame_last"])
    assert str(frames[0].dtype) == str(reference["arr_center__dtype"])


def test_framed_signal_left_origin_translation(reference):
    sig = Signal(np.arange(1, 21, dtype=np.int16), sample_rate=10)
    frames = FramedSignal(sig, frame_size=6, hop_size=2, origin="left")
    assert frames.origin == int(reference["arr_left__origin"])
    _assert_exact(frames[0], reference["arr_left__frame0"])


def test_framed_signal_right_origin_translation(reference):
    sig = Signal(np.arange(1, 21, dtype=np.int16), sample_rate=10)
    frames = FramedSignal(sig, frame_size=6, hop_size=2, origin="right")
    assert frames.origin == int(reference["arr_right__origin"])
    _assert_exact(frames[0], reference["arr_right__frame0"])


def test_num_frames_formula_normal_vs_extend_diverge(reference):
    sig = Signal(np.arange(1, 10, dtype=np.int16), sample_rate=10)  # len 9
    fn = FramedSignal(sig, frame_size=4, hop_size=3, end="normal")
    fe = FramedSignal(sig, frame_size=4, hop_size=3, end="extend")
    assert fn.num_frames == int(reference["numframes_normal"])
    assert fe.num_frames == int(reference["numframes_extend"])
    # sanity: this case is specifically chosen so the two formulas diverge
    assert fn.num_frames != fe.num_frames


def test_fps_derived_hop_size_at_48khz(reference):
    sig = Signal(np.arange(1, 21, dtype=np.int16), sample_rate=48000)
    frames = FramedSignalProcessor(frame_size=2048, fps=100)(sig)
    assert frames.hop_size == float(reference["fps48k__hop_size"])
    assert frames.hop_size == 480.0


def test_remix_mono_downmix_truncates_not_rounds(reference):
    stereo_small = np.array(
        [[3, 4], [1, 2], [-3, -4], [-1, -2], [5, 6]], dtype=np.int16
    )
    result = remix(stereo_small, 1)
    _assert_exact(result, reference["remix_x5__mono"])
    # spot check the exact truncation-not-rounding semantics by hand too:
    # (3+4)/2=3.5 -> 3 (not 4); (-3-4)/2=-3.5 -> -3 (not -4, truncate to zero)
    assert result.tolist() == [3, 1, -3, -1, 5]


# ---------------------------------------------------------------------------
# B. wav-file-based cases (scipy.io.wavfile loading path)
# ---------------------------------------------------------------------------
def test_signal_from_wav_int16_dtype_preserved(wav_dir, reference):
    sig = Signal(str(wav_dir / "mono16_44100.wav"))
    assert str(sig.dtype) == str(reference["mono16_44100__dtype"])
    assert sig.dtype == np.int16
    assert sig.sample_rate == int(reference["mono16_44100__sample_rate"])
    _assert_exact(np.asarray(sig), reference["mono16_44100__data"])


def test_framed_signal_from_wav_frames_match(wav_dir, reference):
    sig = Signal(str(wav_dir / "mono16_44100.wav"))
    frames = FramedSignalProcessor(frame_size=2048, fps=100)(sig)
    assert frames.num_frames == int(reference["mono16_44100__num_frames"])
    assert frames.hop_size == float(reference["mono16_44100__hop_size"])
    _assert_exact(frames[0], reference["mono16_44100__frame0"])
    _assert_exact(frames[1], reference["mono16_44100__frame1"])
    _assert_exact(
        frames[frames.num_frames - 1], reference["mono16_44100__frame_last"]
    )


def test_signal_from_wav_stereo_downmix_to_mono_matches(wav_dir, reference):
    sig = Signal(str(wav_dir / "stereo16_44100.wav"), num_channels=1)
    assert str(sig.dtype) == str(reference["stereo16_mono__dtype"])
    _assert_exact(np.asarray(sig), reference["stereo16_mono__data"])


def test_signal_from_wav_stereo_raw_matches(wav_dir, reference):
    sig = Signal(str(wav_dir / "stereo16_44100.wav"))
    assert sig.num_channels == int(reference["stereo16_raw__num_channels"])
    _assert_exact(np.asarray(sig), reference["stereo16_raw__data"])


def test_framed_signal_from_wav_48khz_matches(wav_dir, reference):
    sig = Signal(str(wav_dir / "mono16_48000.wav"))
    assert sig.sample_rate == int(reference["mono16_48000__sample_rate"])
    frames = FramedSignalProcessor(frame_size=2048, fps=100)(sig)
    assert frames.hop_size == float(reference["mono16_48000__hop_size"])
    assert frames.num_frames == int(reference["mono16_48000__num_frames"])
    _assert_exact(frames[0], reference["mono16_48000__frame0"])
    _assert_exact(frames[1], reference["mono16_48000__frame1"])
    _assert_exact(
        frames[frames.num_frames - 1], reference["mono16_48000__frame_last"]
    )


def test_signal_from_wav_float32_no_rescale(wav_dir, reference):
    sig = Signal(str(wav_dir / "mono_float32.wav"))
    assert str(sig.dtype) == str(reference["mono_float32__dtype"])
    assert sig.dtype == np.float32
    _assert_exact(np.asarray(sig), reference["mono_float32__data"])


# ---------------------------------------------------------------------------
# additional unit-level checks (composition-specific behavior, no madmom
# comparison needed -- these assert properties of our own design decisions
# documented in the module header, e.g. zero-copy from-array construction)
# ---------------------------------------------------------------------------
def test_signal_from_array_is_zero_copy():
    arr = np.arange(20, dtype=np.int16)
    sig = Signal(arr, sample_rate=44100)
    assert np.shares_memory(np.asarray(sig), arr)


def test_signal_array_interop():
    arr = np.arange(20, dtype=np.int16)
    sig = Signal(arr, sample_rate=44100)
    # np.asarray(sig) must work transparently via __array__
    assert np.array_equal(np.asarray(sig), arr)
    # dot/other numpy ops should also work transparently
    assert np.dot(np.asarray(sig), np.ones(20)) == arr.sum()


def test_framed_signal_processor_is_a_processor():
    from madmom_infer.processors import Processor

    proc = FramedSignalProcessor(frame_size=2048, fps=100)
    assert isinstance(proc, Processor)


def test_framed_signal_negative_index_and_out_of_range():
    sig = Signal(np.arange(1, 21, dtype=np.int16), sample_rate=10)
    frames = FramedSignal(sig, frame_size=6, hop_size=2)
    assert np.array_equal(frames[-1], frames[frames.num_frames - 1])
    with pytest.raises(IndexError):
        frames[frames.num_frames]


def test_framed_signal_slice_returns_framed_signal_with_correct_frames():
    sig = Signal(np.arange(1, 21, dtype=np.int16), sample_rate=10)
    frames = FramedSignal(sig, frame_size=6, hop_size=2)
    sub = frames[2:5]
    assert isinstance(sub, FramedSignal)
    assert len(sub) == 3
    for i in range(3):
        assert np.array_equal(sub[i], frames[2 + i])


def test_framed_signal_shape_includes_channels_for_multichannel():
    stereo = np.stack(
        [np.arange(1, 21, dtype=np.int16), np.arange(21, 41, dtype=np.int16)],
        axis=-1,
    )
    sig = Signal(stereo, sample_rate=10)
    frames = FramedSignal(sig, frame_size=6, hop_size=2)
    assert frames.shape == (frames.num_frames, 6, 2)
    assert frames.ndim == 3
    assert frames[0].shape == (6, 2)
