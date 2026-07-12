"""Golden-fixture tests for Wave 4c: `madmom_infer.features.tempo` -- the
tempo histogram processor family (`ACFTempoHistogramProcessor`,
`CombFilterTempoHistogramProcessor`, `DBNTempoHistogramProcessor`) and
`TempoEstimationProcessor`, all recorded by `tools/
generate_beat_tempo_fixtures.py` from real (compiled) madmom.

**Every test in this file is fully OFFLINE, including the cross-BLAS
proof** -- unlike `test_beats.py`/`test_onsets.py`/`test_downbeats_rnn.py`,
nothing in `features/tempo.py` unpickles a `.pkl` model file or downloads
any weights: `TempoEstimationProcessor`/the histogram processors operate
purely on a given beat-activation NUMPY ARRAY (recorded once in the fixture
itself, real `RNNBeatProcessor(online=False)` output on `mono_44100.wav`)
-- `DBNTempoHistogramProcessor`'s `dbn` mode reuses `features/beats.py`'s
`DBNBeatTrackingProcessor`, which is HMM/Viterbi decoding, not an NN
forward pass either. So this file needs neither `-m network` nor the
reference-venv subprocess dance for its correctness claims, though the
cross-BLAS test is still included (comb-filter-mode tempo estimation does
touch `madmom_infer/audio/comb_filters.py`, which IS proven bit-identical
there -- this test extends that same claim one level up, through the full
histogram + peak-detection pipeline).

Reads: madmom_infer/features/tempo.py, tests/fixtures/tempo_histograms.npz.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from madmom_infer.features.tempo import (
    ACFTempoHistogramProcessor, CombFilterTempoHistogramProcessor,
    DBNTempoHistogramProcessor, TempoEstimationProcessor, detect_tempo,
    dominant_interval, interval_histogram_acf, interval_histogram_comb,
    smooth_histogram,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PYTHON = Path(
    "/home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python"
)


@pytest.fixture(scope="module")
def fixture():
    return np.load(FIXTURES_DIR / "tempo_histograms.npz")


@pytest.fixture(scope="module")
def act(fixture):
    return fixture["tempo_input_activations"]


# ---------------------------------------------------------------------------
# helper-function sanity (small, hand-built, not fixture-based -- pure
# arithmetic, doesn't depend on real madmom's trained weights)
# ---------------------------------------------------------------------------
def test_smooth_histogram_only_smooths_bins():
    bins = np.array([0., 1., 5., 1., 0.])
    delays = np.array([1, 2, 3, 4, 5])
    smoothed_bins, smoothed_delays = smooth_histogram((bins, delays), 3)
    np.testing.assert_array_equal(smoothed_delays, delays)
    assert not np.array_equal(smoothed_bins, bins)


def test_dominant_interval_picks_the_max_bin():
    bins = np.array([0., 1., 5., 1., 0.])
    delays = np.array([10, 20, 30, 40, 50])
    assert dominant_interval((bins, delays)) == 30


def test_detect_tempo_single_peak():
    bins = np.array([0., 1., 0.])
    tempi = np.array([60., 120., 240.])
    out = detect_tempo((bins, tempi))
    np.testing.assert_array_equal(out, np.array([[120., 1.]]))


def test_detect_tempo_no_peaks_returns_no_tempo():
    bins = np.zeros(5)
    tempi = np.array([60., 90., 120., 150., 180.])
    out = detect_tempo((bins, tempi))
    assert np.isnan(out[0, 0])
    assert out[0, 1] == 0.


def test_interval_histogram_acf_rejects_multidimensional():
    with pytest.raises(NotImplementedError):
        interval_histogram_acf(np.zeros((10, 2)))


def test_interval_histogram_comb_rejects_multidimensional():
    with pytest.raises(NotImplementedError):
        interval_histogram_comb(np.zeros((10, 2, 2)), alpha=0.79)


# ---------------------------------------------------------------------------
# real-madmom-fixture exactness, per histogram mode -- fully offline (see
# module header)
# ---------------------------------------------------------------------------
def test_acf_histogram_matches_fixture(fixture, act):
    proc = ACFTempoHistogramProcessor(fps=100)
    bins, delays = proc(act.astype(float))
    np.testing.assert_array_equal(np.asarray(bins), fixture["acf_histogram_bins"])
    np.testing.assert_array_equal(np.asarray(delays), fixture["acf_histogram_delays"])


def test_comb_histogram_matches_fixture(fixture, act):
    proc = CombFilterTempoHistogramProcessor(fps=100)
    bins, delays = proc(act.astype(float))
    np.testing.assert_array_equal(np.asarray(bins), fixture["comb_histogram_bins"])
    np.testing.assert_array_equal(np.asarray(delays), fixture["comb_histogram_delays"])


def test_dbn_histogram_matches_fixture(fixture, act):
    proc = DBNTempoHistogramProcessor(fps=100)
    bins, delays = proc(act)
    np.testing.assert_array_equal(np.asarray(bins), fixture["dbn_histogram_bins"])
    np.testing.assert_array_equal(np.asarray(delays), fixture["dbn_histogram_delays"])


@pytest.mark.parametrize("method", ["acf", "comb", "dbn"])
def test_tempo_estimation_processor_matches_fixture(fixture, act, method):
    proc = TempoEstimationProcessor(method=method, fps=100)
    out = proc(act)
    np.testing.assert_array_equal(out, fixture[f"{method}_tempi"])


def test_tempo_estimation_processor_rejects_unknown_method():
    with pytest.raises(ValueError):
        TempoEstimationProcessor(method="sideways", fps=100)


def test_tempo_estimation_processor_min_max_bpm_properties():
    proc = TempoEstimationProcessor(method="acf", min_bpm=40., max_bpm=250.,
                                     fps=100)
    assert proc.min_bpm == 40.
    assert proc.max_bpm == 250.
    assert proc.min_interval == int(np.floor(60. * 100 / 250.))
    assert proc.max_interval == int(np.ceil(60. * 100 / 40.))
    np.testing.assert_array_equal(
        proc.intervals, np.arange(proc.min_interval, proc.max_interval + 1))


# ---------------------------------------------------------------------------
# cross-BLAS exactness -- fully offline (no models/network needed, see
# module header)
# ---------------------------------------------------------------------------
def _reference_python_available():
    return REFERENCE_PYTHON.exists()


@pytest.mark.skipif(
    not _reference_python_available(),
    reason="reference madmom install (madmom-reference/.venv) not found on "
           "this machine; the cross-BLAS proof requires it",
)
def test_tempo_estimation_is_exact_under_original_blas():
    """This port's own `TempoEstimationProcessor` (all 3 modes), run under
    the ORIGINAL reference venv's numpy/scipy build, reproduces real
    madmom's fixture tempi with ZERO differing elements."""
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import numpy as np
from madmom_infer.features.tempo import TempoEstimationProcessor

fixture = np.load({str(FIXTURES_DIR / "tempo_histograms.npz")!r})
act = fixture["tempo_input_activations"]
for method in ("acf", "comb", "dbn"):
    proc = TempoEstimationProcessor(method=method, fps=100)
    out = proc(act)
    assert np.array_equal(out, fixture[f"{{method}}_tempi"]), \\
        f"{{method}}: tempi differ"
print("EXACT_MATCH")
"""
    proc = subprocess.run(
        [str(REFERENCE_PYTHON), "-c", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "EXACT_MATCH" in proc.stdout
