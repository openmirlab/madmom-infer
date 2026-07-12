"""Golden-fixture tests for `madmom_infer.features.beats_crf` -- Wave 4f's
numpy port of madmom's Cython CRF beat-tracking Viterbi decoder
(`features/beats_crf.pyx`), recorded by `tools/generate_crf_pattern_fixtures.py`
from real (compiled) madmom.

Several independent things are verified here:

1. **The 4 individual functions** (`initial_distribution`,
   `transition_distribution`, `normalisation_factors`, `best_sequence`),
   each fed a REAL beat activation function (`RNNBeatProcessor(online=False)`
   on `mono_44100.wav`) at 3 representative intervals -- offline, fully
   self-contained fixture (`tests/fixtures/beats_crf_functions.npz`).
   `best_sequence` is checked BOTH for its decoded path (`np.array_equal`,
   an integer sequence, no tolerance possible) AND its scalar `log_prob`
   (compared as `float32`, matching the actual computed precision -- see
   `madmom_infer/features/beats_crf.py`'s module header for why float32,
   not float64, is the correct comparison dtype).
2. **A randomized fuzz test against real madmom directly** (network-free,
   in-process -- no fixture needed): confirms bit-identical decode/log_prob
   across many random `(activations, interval, interval_sigma)` combinations,
   not just the one fixture case. Skipped if the reference venv isn't
   available (same skip precedent as the cross-BLAS tests below).
3. **Cross-BLAS exactness**: this port's own `viterbi()`/`best_sequence()`,
   run under the ORIGINAL reference venv's numpy/scipy build, reproduce real
   madmom's decoded path AND `log_prob` with ZERO differing elements.
   `viterbi()` never touches BLAS at all (pure elementwise/reduction numpy
   ops, no matrix multiply) -- bit-identical, not merely ULP-close, is the
   expected and verified claim here, same precedent as `test_comb_filters.py`/
   `test_crf.py`.

Reads: madmom_infer/features/beats_crf.py,
tests/fixtures/beats_crf_functions.npz.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from madmom_infer.features.beats_crf import (
    best_sequence, initial_distribution, normalisation_factors,
    transition_distribution, viterbi,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_PYTHON = Path(
    "/home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python"
)


@pytest.fixture(scope="module")
def crf_fn_fixture():
    return np.load(FIXTURES_DIR / "beats_crf_functions.npz")


# ---------------------------------------------------------------------------
# 1. Individual functions, offline, self-contained fixture
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("interval", [20, 35, 55])
def test_initial_distribution_matches_fixture(crf_fn_fixture, interval):
    act = crf_fn_fixture["crf_input_activations"]
    out = initial_distribution(act.shape[0], interval)
    expected = crf_fn_fixture[f"init_dist_interval{interval}"]
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == expected.dtype == np.float32


@pytest.mark.parametrize("interval", [20, 35, 55])
def test_transition_distribution_matches_fixture(crf_fn_fixture, interval):
    out = transition_distribution(interval, 0.18)
    expected = crf_fn_fixture[f"trans_dist_interval{interval}"]
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("interval", [20, 35, 55])
def test_normalisation_factors_matches_fixture(crf_fn_fixture, interval):
    act = crf_fn_fixture["crf_input_activations"]
    trans = crf_fn_fixture[f"trans_dist_interval{interval}"]
    out = normalisation_factors(act, trans)
    expected = crf_fn_fixture[f"norm_factors_interval{interval}"]
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("interval", [20, 35, 55])
def test_best_sequence_matches_fixture_exact(crf_fn_fixture, interval):
    act = crf_fn_fixture["crf_input_activations"]
    contiguous_act = np.ascontiguousarray(act, dtype=np.float32)
    path, log_prob = best_sequence(contiguous_act, interval, 0.18)
    expected_path = crf_fn_fixture[f"best_sequence_path_interval{interval}"]
    expected_log_prob = crf_fn_fixture[
        f"best_sequence_log_prob_interval{interval}"]
    np.testing.assert_array_equal(path, expected_path)
    # compare as float32 -- viterbi()'s log_prob is computed and stored
    # entirely in float32 precision, see that module's header.
    assert np.float32(log_prob) == np.float32(expected_log_prob[()])


# ---------------------------------------------------------------------------
# 2. Randomized fuzz test against real madmom directly (in-process, no
# fixture -- needs the reference venv's madmom installed in THIS process,
# so it only runs when this test file itself is executed under that venv;
# skipped otherwise, matching the cross-BLAS tests' own skip precedent)
# ---------------------------------------------------------------------------
def _real_madmom_available():
    try:
        import madmom.features.beats_crf  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _real_madmom_available(),
    reason="real (compiled) madmom not importable in this interpreter -- "
           "this fuzz test only runs directly under the reference venv",
)
def test_viterbi_fuzz_matches_real_madmom():
    import madmom.features.beats_crf as real_crf

    rng = np.random.RandomState(20260713)
    n_checked = 0
    for _ in range(80):
        n = rng.randint(50, 400)
        act = (rng.rand(n).astype(np.float32) * 0.95 + 0.02)
        interval = rng.randint(5, 60)
        if n // interval < 2:
            continue
        sigma = float(rng.uniform(0.05, 0.4))
        real_path, real_prob = real_crf.best_sequence(act, interval, sigma)
        ported_path, ported_prob = best_sequence(act, interval, sigma)
        np.testing.assert_array_equal(real_path, ported_path)
        assert np.float32(real_prob) == np.float32(ported_prob)
        n_checked += 1
    assert n_checked > 50


# ---------------------------------------------------------------------------
# 3. Cross-BLAS exactness (the strongest claim)
# ---------------------------------------------------------------------------
def _reference_python_available():
    return REFERENCE_PYTHON.exists()


@pytest.mark.skipif(
    not _reference_python_available(),
    reason="reference madmom install (madmom-reference/.venv) not found on "
           "this machine; the cross-BLAS proof requires it",
)
def test_viterbi_is_exact_under_original_blas():
    """This port's own `best_sequence()`/`viterbi()`, run under the
    ORIGINAL reference venv's numpy/scipy build, reproduce real madmom's
    decoded path AND log_prob with ZERO differing elements for several
    representative intervals against the recorded fixture activations, plus
    a randomized fuzz sweep -- `viterbi()` never touches BLAS at all (pure
    numpy elementwise/reduction ops), so bit-identity (not just ULP-
    closeness) is the expected, verified claim.
    """
    fixture_path = str(FIXTURES_DIR / "beats_crf_functions.npz")
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import numpy as np
from madmom_infer.features.beats_crf import best_sequence

fixture = np.load({fixture_path!r})
act = fixture["crf_input_activations"]
contiguous_act = np.ascontiguousarray(act, dtype=np.float32)
for interval in (20, 35, 55):
    path, log_prob = best_sequence(contiguous_act, interval, 0.18)
    expected_path = fixture[f"best_sequence_path_interval{{interval}}"]
    expected_log_prob = fixture[f"best_sequence_log_prob_interval{{interval}}"]
    assert np.array_equal(path, expected_path), f"interval={{interval}}: path differs"
    assert np.float32(log_prob) == np.float32(expected_log_prob[()]), \\
        f"interval={{interval}}: log_prob differs"

# also fuzz against real madmom directly, under this reference venv
import madmom.features.beats_crf as real_crf
rng = np.random.RandomState(20260713)
n_checked = 0
for _ in range(150):
    n = rng.randint(50, 400)
    a = (rng.rand(n).astype(np.float32) * 0.95 + 0.02)
    interval = rng.randint(5, 60)
    if n // interval < 2:
        continue
    sigma = float(rng.uniform(0.05, 0.4))
    real_path, real_prob = real_crf.best_sequence(a, interval, sigma)
    ported_path, ported_prob = best_sequence(a, interval, sigma)
    assert np.array_equal(real_path, ported_path), "fuzz: path differs"
    assert np.float32(real_prob) == np.float32(ported_prob), "fuzz: log_prob differs"
    n_checked += 1
assert n_checked > 50
print("EXACT_MATCH")
"""
    proc = subprocess.run(
        [str(REFERENCE_PYTHON), "-c", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "EXACT_MATCH" in proc.stdout


def test_viterbi_negative_beat_count_raises():
    """A too-short activation array for the given `tau` yields `num_x < 1`
    -- `bps`'s first dimension goes negative, matching real madmom's own
    failure mode (`np.empty` with a negative shape raises `ValueError`
    there too, confirmed empirically against the reference venv)."""
    act = np.full(10, 0.5, dtype=np.float32)
    with pytest.raises(ValueError):
        viterbi(np.zeros(10, dtype=np.float32),
               np.zeros(4, dtype=np.float32),
               np.zeros(10, dtype=np.float32),
               np.log(act), tau=20)
