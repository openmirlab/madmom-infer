"""Golden-fixture tests for the 6 `madmom_infer.audio.signal` functions Wave
4g ported to resolve the 4b audit-table TO-VERIFY flag: `attenuate`,
`rescale`, `trim`, `energy`, `root_mean_square`, `sound_pressure_level`.
Fixtures recorded by `tools/generate_leftovers_fixtures.py` from real
(compiled) madmom on `mono_44100.wav`.

**All 6 functions are pure numpy (no BLAS, no FFT) -- proven EXACTLY equal
(`np.array_equal`), not just within a tolerance, both in-process AND
cross-BLAS.** Same precedent as `tests/test_comb_filters.py`.

Reads: madmom_infer/audio/signal.py, tests/fixtures/signal_leftovers.npz.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from madmom_infer.audio.signal import (
    FramedSignalProcessor, Signal, attenuate, energy, rescale,
    root_mean_square, sound_pressure_level, trim,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parent.parent
WAV_PATH = FIXTURES_DIR / "wavs" / "mono_44100.wav"
REFERENCE_PYTHON = Path(
    "/home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python"
)


@pytest.fixture(scope="module")
def fixture():
    return np.load(FIXTURES_DIR / "signal_leftovers.npz")


@pytest.fixture(scope="module")
def sig():
    return Signal(str(WAV_PATH), num_channels=1)


# ---------------------------------------------------------------------------
# small hand-written sanity checks (not fixture-dependent)
# ---------------------------------------------------------------------------
def test_attenuate_zero_is_noop():
    x = np.array([1, 2, 3], dtype=np.int16)
    assert attenuate(x, 0) is x


def test_attenuate_positive_int_dtype_raises():
    x = np.array([1, 2, 3], dtype=np.int16)
    with pytest.raises(ValueError):
        attenuate(x, -6.0)  # negative attenuation = positive gain


def test_rescale_rejects_non_float_dtype():
    x = np.array([1, 2, 3], dtype=np.int16)
    with pytest.raises(ValueError):
        rescale(x, dtype=np.int16)


def test_trim_all_zero_signal():
    x = np.zeros(10)
    assert len(trim(x)) == 0


def test_energy_rejects_non_ndarray():
    with pytest.raises(TypeError):
        energy([1, 2, 3])


def test_energy_takes_abs_of_complex_signal():
    x = np.array([1j, 2j, -3j])
    # |1j|^2 + |2j|^2 + |3j|^2 = 1 + 4 + 9 = 14
    assert energy(x) == pytest.approx(14.0)


# ---------------------------------------------------------------------------
# real-madmom-fixture exactness
# ---------------------------------------------------------------------------
def test_attenuate_matches_fixture_exactly(fixture, sig):
    np.testing.assert_array_equal(
        np.asarray(attenuate(sig, 6.0)), fixture["attenuate_6db"])
    np.testing.assert_array_equal(
        np.asarray(attenuate(sig, 0.0)), fixture["attenuate_0db"])


def test_rescale_matches_fixture_exactly(fixture, sig):
    np.testing.assert_array_equal(
        np.asarray(rescale(sig, dtype=np.float32)),
        fixture["rescale_float32"])
    np.testing.assert_array_equal(
        np.asarray(rescale(sig, dtype=np.float64)),
        fixture["rescale_float64"])


def test_trim_matches_fixture_exactly(fixture):
    padded = fixture["trim_input"]
    np.testing.assert_array_equal(trim(padded, where="fb"), fixture["trim_fb"])
    np.testing.assert_array_equal(trim(padded, where="f"), fixture["trim_f"])
    np.testing.assert_array_equal(trim(padded, where="b"), fixture["trim_b"])


def test_energy_rms_spl_matches_fixture_exactly(fixture):
    sig_data = fixture["signal_input"]
    np.testing.assert_array_equal(energy(sig_data), fixture["energy_1d"])
    np.testing.assert_array_equal(
        root_mean_square(sig_data), fixture["root_mean_square_1d"])
    np.testing.assert_array_equal(
        sound_pressure_level(sig_data), fixture["sound_pressure_level_1d"])


def test_energy_rms_spl_framed_matches_fixture_exactly(fixture, sig):
    frames = FramedSignalProcessor(frame_size=2048, fps=100)(sig)
    np.testing.assert_array_equal(energy(frames), fixture["energy_framed"])
    np.testing.assert_array_equal(
        root_mean_square(frames), fixture["root_mean_square_framed"])
    np.testing.assert_array_equal(
        sound_pressure_level(frames), fixture["sound_pressure_level_framed"])


def test_energy_spl_float_dtype_matches_fixture_exactly(fixture):
    sig_data = fixture["signal_input"]
    float_sig = sig_data.astype(np.float32) / 32768.0
    np.testing.assert_array_equal(energy(float_sig), fixture["energy_float"])
    np.testing.assert_array_equal(
        sound_pressure_level(float_sig), fixture["sound_pressure_level_float"])


# ---------------------------------------------------------------------------
# cross-BLAS exactness (completeness -- see module header, not load-bearing
# the way NN-forward-pass cross-BLAS tests are, since no BLAS call is
# involved at all here).
# ---------------------------------------------------------------------------
def _reference_python_available():
    return REFERENCE_PYTHON.exists()


@pytest.mark.skipif(
    not _reference_python_available(),
    reason="reference madmom install (madmom-reference/.venv) not found on "
           "this machine",
)
def test_signal_leftovers_are_exact_under_original_blas():
    """This port's own leftover signal functions, run under the reference
    venv's numpy/scipy build, reproduce real madmom's fixture values with
    ZERO differing elements."""
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import numpy as np
from madmom_infer.audio.signal import (
    Signal, FramedSignalProcessor, attenuate, rescale, trim, energy,
    root_mean_square, sound_pressure_level,
)
fixture = np.load({str(FIXTURES_DIR / "signal_leftovers.npz")!r})
sig = Signal({str(WAV_PATH)!r}, num_channels=1)
assert np.array_equal(np.asarray(attenuate(sig, 6.0)), fixture["attenuate_6db"])
assert np.array_equal(np.asarray(attenuate(sig, 0.0)), fixture["attenuate_0db"])
assert np.array_equal(np.asarray(rescale(sig, dtype=np.float32)), fixture["rescale_float32"])
assert np.array_equal(np.asarray(rescale(sig, dtype=np.float64)), fixture["rescale_float64"])
padded = fixture["trim_input"]
assert np.array_equal(trim(padded, where="fb"), fixture["trim_fb"])
assert np.array_equal(trim(padded, where="f"), fixture["trim_f"])
assert np.array_equal(trim(padded, where="b"), fixture["trim_b"])
sig_data = fixture["signal_input"]
assert np.array_equal(energy(sig_data), fixture["energy_1d"])
assert np.array_equal(root_mean_square(sig_data), fixture["root_mean_square_1d"])
assert np.array_equal(sound_pressure_level(sig_data), fixture["sound_pressure_level_1d"])
frames = FramedSignalProcessor(frame_size=2048, fps=100)(sig)
assert np.array_equal(energy(frames), fixture["energy_framed"])
assert np.array_equal(root_mean_square(frames), fixture["root_mean_square_framed"])
assert np.array_equal(sound_pressure_level(frames), fixture["sound_pressure_level_framed"])
float_sig = sig_data.astype(np.float32) / 32768.0
assert np.array_equal(energy(float_sig), fixture["energy_float"])
assert np.array_equal(sound_pressure_level(float_sig), fixture["sound_pressure_level_float"])
print("EXACT_MATCH")
"""
    proc = subprocess.run(
        [str(REFERENCE_PYTHON), "-c", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "EXACT_MATCH" in proc.stdout
