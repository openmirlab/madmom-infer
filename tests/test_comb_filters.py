"""Golden-fixture tests for `madmom_infer.audio.comb_filters` -- Wave 4c's
numpy port of `madmom.audio.comb_filters` (`comb_filters.pyx`). Fixtures
recorded by `tools/generate_beat_tempo_fixtures.py` from real (compiled)
madmom, fed a REAL beat activation function (`RNNBeatProcessor(online=
False)`'s output on `mono_44100.wav`), not synthetic noise.

**Comb filters are proven EXACTLY equal (`np.array_equal`), not just within
a ULP tolerance -- both in-process AND cross-BLAS.** Unlike the NN forward
passes elsewhere in this project (BLAS-non-associativity-sensitive matmuls),
`feed_forward_comb_filter` is one vectorized elementwise multiply-add (no
reduction, no summation-order ambiguity) and `feed_backward_comb_filter` is
a scalar Python `for` loop (same float64 operation order as real madmom's
own Cython `for` loop, see `madmom_infer/audio/comb_filters.py`'s module
header) -- neither touches BLAS at all, so there is no cross-build
non-associativity to average away. This test file asserts bit-identity
directly, in-process, WITHOUT needing the reference-venv subprocess dance
`test_downbeats_rnn.py`/`test_onsets.py` need for their NN paths (a
cross-BLAS test is still included for completeness/consistency with this
repo's established pattern, and because it is the more defensible claim if
this project's numpy/scipy pin ever changes).

Reads: madmom_infer/audio/comb_filters.py, tests/fixtures/
beats_comb_filters.npz.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from madmom_infer.audio.comb_filters import (
    CombFilterbankProcessor, comb_filter, feed_backward_comb_filter,
    feed_forward_comb_filter,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PYTHON = Path(
    "/home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python"
)


@pytest.fixture(scope="module")
def fixture():
    return np.load(FIXTURES_DIR / "beats_comb_filters.npz")


# ---------------------------------------------------------------------------
# docstring examples (hand-verified against real madmom's own docstrings)
# ---------------------------------------------------------------------------
def test_feed_forward_docstring_example():
    x = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1])
    out = feed_forward_comb_filter(x, tau=3, alpha=0.5)
    np.testing.assert_array_equal(
        out, np.array([0., 0., 1., 0., 0., 1.5, 0., 0., 1.5]))


def test_feed_backward_docstring_example():
    x = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1])
    out = feed_backward_comb_filter(x, tau=3, alpha=0.5)
    np.testing.assert_array_equal(
        out, np.array([0., 0., 1., 0., 0., 1.5, 0., 0., 1.75]))


def test_comb_filter_bank_docstring_examples():
    x = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1])
    fwd = comb_filter(x, feed_forward_comb_filter, [2, 3], [0.5, 0.5])
    np.testing.assert_array_equal(fwd, np.array([
        [0., 0.], [0., 0.], [1., 1.], [0., 0.], [0.5, 0.],
        [1., 1.5], [0., 0.], [0.5, 0.], [1., 1.5],
    ]))
    bwd = comb_filter(x, feed_backward_comb_filter, [2, 3], [0.5, 0.5])
    np.testing.assert_array_equal(bwd, np.array([
        [0., 0.], [0., 0.], [1., 1.], [0., 0.], [0.5, 0.],
        [1., 1.5], [0.25, 0.], [0.5, 0.], [1.125, 1.75],
    ]))


def test_combfilterbankprocessor_forward_and_backward():
    x = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1])
    fwd = CombFilterbankProcessor("forward", [2, 3], [0.5, 0.5])(x)
    bwd = CombFilterbankProcessor("backward", [2, 3], [0.5, 0.5])(x)
    np.testing.assert_array_equal(
        fwd, comb_filter(x, feed_forward_comb_filter, [2, 3], [0.5, 0.5]))
    np.testing.assert_array_equal(
        bwd, comb_filter(x, feed_backward_comb_filter, [2, 3], [0.5, 0.5]))


def test_combfilterbankprocessor_rejects_unknown_filter_function():
    with pytest.raises(ValueError):
        CombFilterbankProcessor("sideways", [2], [0.5])


def test_feed_forward_rejects_non_positive_tau():
    with pytest.raises(ValueError):
        feed_forward_comb_filter(np.zeros(10), tau=0, alpha=0.5)


def test_feed_backward_rejects_non_positive_tau():
    with pytest.raises(ValueError):
        feed_backward_comb_filter(np.zeros(10), tau=0, alpha=0.5)


def test_comb_filter_rejects_mismatched_tau_alpha_length():
    # a single alpha auto-expands to match `tau`'s length (not an error --
    # see `comb_filter`'s `len(alpha) == 1` branch); 2 alphas for 3 taus is
    # a genuine length mismatch.
    with pytest.raises(ValueError):
        comb_filter(np.zeros(10), feed_forward_comb_filter, [2, 3, 5],
                    [0.5, 0.6])


# ---------------------------------------------------------------------------
# real-madmom-fixture exactness -- a REAL beat activation function, alpha
# (0.79) not exactly representable in float32 (deliberately exercises the
# feed_backward float32-alpha-rounding quirk).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tau", [5, 17, 43])
def test_feed_forward_matches_fixture_exactly(fixture, tau):
    act = fixture["comb_filter_input_1d"]
    out = feed_forward_comb_filter(act, tau, 0.79)
    np.testing.assert_array_equal(out, fixture[f"feed_forward_tau{tau}"])


@pytest.mark.parametrize("tau", [5, 17, 43])
def test_feed_backward_matches_fixture_exactly(fixture, tau):
    act = fixture["comb_filter_input_1d"]
    out = feed_backward_comb_filter(act, tau, 0.79)
    np.testing.assert_array_equal(out, fixture[f"feed_backward_tau{tau}"])


def test_feed_backward_2d_matches_fixture_exactly(fixture):
    act_2d = fixture["comb_filter_input_2d"]
    out = feed_backward_comb_filter(act_2d, 17, 0.79)
    np.testing.assert_array_equal(out, fixture["feed_backward_2d_tau17"])


def test_comb_filter_bank_matches_fixture_exactly(fixture):
    act = fixture["comb_filter_input_1d"]
    fwd = comb_filter(act, feed_forward_comb_filter, [5, 17, 43],
                       [0.79, 0.79, 0.79])
    bwd = comb_filter(act, feed_backward_comb_filter, [5, 17, 43],
                       [0.79, 0.79, 0.79])
    np.testing.assert_array_equal(fwd, fixture["comb_filter_bank_forward"])
    np.testing.assert_array_equal(bwd, fixture["comb_filter_bank_backward"])


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
def test_comb_filters_are_exact_under_original_blas():
    """This port's own comb-filter functions, run under the reference
    venv's numpy/scipy build, reproduce real madmom's fixture values with
    ZERO differing elements -- confirms the in-process exactness above
    isn't an artifact of this particular numpy build."""
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import numpy as np
from madmom_infer.audio.comb_filters import (
    feed_forward_comb_filter, feed_backward_comb_filter, comb_filter,
)
fixture = np.load({str(FIXTURES_DIR / "beats_comb_filters.npz")!r})
act = fixture["comb_filter_input_1d"]
act_2d = fixture["comb_filter_input_2d"]
for tau in (5, 17, 43):
    assert np.array_equal(
        feed_forward_comb_filter(act, tau, 0.79),
        fixture[f"feed_forward_tau{{tau}}"]), f"feed_forward tau={{tau}}"
    assert np.array_equal(
        feed_backward_comb_filter(act, tau, 0.79),
        fixture[f"feed_backward_tau{{tau}}"]), f"feed_backward tau={{tau}}"
assert np.array_equal(
    feed_backward_comb_filter(act_2d, 17, 0.79),
    fixture["feed_backward_2d_tau17"])
fwd = comb_filter(act, feed_forward_comb_filter, [5, 17, 43],
                   [0.79, 0.79, 0.79])
bwd = comb_filter(act, feed_backward_comb_filter, [5, 17, 43],
                   [0.79, 0.79, 0.79])
assert np.array_equal(fwd, fixture["comb_filter_bank_forward"])
assert np.array_equal(bwd, fixture["comb_filter_bank_backward"])
print("EXACT_MATCH")
"""
    proc = subprocess.run(
        [str(REFERENCE_PYTHON), "-c", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "EXACT_MATCH" in proc.stdout
