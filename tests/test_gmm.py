"""Golden-fixture tests for `madmom_infer.ml.gmm` -- Wave 4f's numpy port of
madmom's forward-inference-only Gaussian Mixture Model, backing
`features/beats_hmm.py`'s `GMMPatternTrackingObservationModel`. Fixtures
recorded by `tools/generate_crf_pattern_fixtures.py` from real (compiled)
madmom, against the actual shipped `PATTERNS_BALLROOM` GMMs.

Three independent things are verified here:

1. **`score`/`score_samples` against real `PATTERNS_BALLROOM` GMM
   parameters** (`tests/fixtures/gmm_scores.npz`) -- fully self-contained
   (records each sampled GMM's means/covars/weights/covariance_type plus a
   fixed random query array and real madmom's output), no model download or
   unpickling needed. Both target pattern files use `covariance_type='full'`
   (confirmed empirically, see `madmom_infer/ml/gmm.py`'s module header),
   so this exercises `_log_multivariate_normal_density_full` specifically.
   Floating goldens require the separately verified reference environment;
   portable semantic checks below remain active on every build.
2. **Unpickling correctness** (network): both `PATTERNS_BALLROOM` `.pkl`
   files, loaded via this project's own restricted `SafeUnpickler`, must
   structurally match real madmom's own bare `pickle.load` -- same
   `num_beats`/per-GMM means/covars/weights digest comparison shape as
   other waves' NN structural-digest tests.
3. **Cross-BLAS exactness**: this port's own `GMM.score`/`score_samples`,
   run under the ORIGINAL reference venv's numpy/scipy build, reproduce
   real madmom's output with ZERO differing elements -- `score_samples`
   does call `scipy.linalg.cholesky`/`solve_triangular` (BLAS/LAPACK-backed),
   so bit-identity here is a genuine (not free) claim, verified rather than
   assumed.

Reads: madmom_infer/ml/gmm.py, madmom_infer/ml/nn/unpickle.py,
tests/fixtures/gmm_scores.npz, tests/fixtures/patterns_structural_digest.json.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests._legacy_fixture_reference import assert_gmm_exact, gmm_cases, reference_mismatch

from madmom_infer.ml.gmm import (
    GMM, log_multivariate_normal_density, logsumexp,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_PYTHON = Path(
    "/home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python"
)
UPSTREAM_PATTERNS_DIR = (
    REPO_ROOT.parent / "madmom-upstream" / "madmom" / "models" / "patterns" / "2013"
)


@pytest.fixture(scope="module")
def gmm_fixture():
    return np.load(FIXTURES_DIR / "gmm_scores.npz")


# ---------------------------------------------------------------------------
# 1. score/score_samples against real GMM parameters, offline
# ---------------------------------------------------------------------------
def test_gmm_score_matches_fixture_exact(gmm_fixture):
    reason = reference_mismatch()
    if reason:
        pytest.skip(reason)
    assert_gmm_exact(gmm_fixture)


def test_gmm_fixture_component_decisions_are_portable(gmm_fixture):
    checked = 0
    for key, gmm in gmm_cases(gmm_fixture):
        log_prob, responsibilities = gmm.score_samples(gmm_fixture[key + "_x"])
        assert np.isfinite(log_prob).all() and np.isfinite(responsibilities).all()
        assert ((responsibilities >= 0) & (responsibilities <= 1)).all()
        np.testing.assert_array_equal(
            responsibilities.argmax(axis=1),
            gmm_fixture[key + "_responsibilities"].argmax(axis=1))
        checked += 1
    assert checked == 12


def test_gmm_single_standard_normal_has_analytic_density():
    gmm = GMM(n_components=1, covariance_type="full")
    gmm.means = np.zeros((1, 1))
    gmm.covars = np.ones((1, 1, 1))
    gmm.weights = np.ones(1)
    x = np.array([[0.], [1.], [-1.], [2.]])
    log_prob, responsibilities = gmm.score_samples(x)
    expected = -0.5 * (np.log(2 * np.pi) + np.array([0., 1., 1., 4.]))
    np.testing.assert_array_equal(log_prob, expected)
    np.testing.assert_array_equal(responsibilities, np.ones((4, 1)))


def test_gmm_equal_gaussians_choose_nearest_mean_and_tie_at_midpoint():
    gmm = GMM(n_components=2, covariance_type="full")
    gmm.means = np.array([[-1.], [1.]])
    gmm.covars = np.ones((2, 1, 1))
    gmm.weights = np.array([0.5, 0.5])
    _, responsibilities = gmm.score_samples(np.array([[-4.], [0.], [4.]]))
    np.testing.assert_array_equal(responsibilities.argmax(axis=1), [0, 0, 1])
    assert responsibilities[1, 0] == responsibilities[1, 1]


def test_gmm_covariance_type_is_full_for_both_patterns(gmm_fixture):
    """Confirms this project's own module-header claim: both
    `PATTERNS_BALLROOM` pattern files' GMMs use `covariance_type='full'`."""
    for p_idx in (0, 1):
        assert str(gmm_fixture[f"pattern{p_idx}_covariance_type"]) == "full"


def test_gmm_score_samples_empty_input():
    gmm = GMM(n_components=2, covariance_type="diag")
    gmm.means = np.zeros((2, 3))
    gmm.covars = np.ones((2, 3))
    log_prob, responsibilities = gmm.score_samples(np.empty((0, 3)))
    assert log_prob.shape == (0,)
    assert responsibilities.shape == (0, 2)


def test_gmm_score_samples_shape_mismatch_raises():
    gmm = GMM(n_components=2, covariance_type="diag")
    gmm.means = np.zeros((2, 3))
    gmm.covars = np.ones((2, 3))
    with pytest.raises(ValueError):
        gmm.score_samples(np.zeros((5, 4)))


def test_gmm_invalid_covariance_type_raises():
    with pytest.raises(ValueError):
        GMM(covariance_type="bogus")


def test_logsumexp_matches_naive():
    rng = np.random.RandomState(0)
    arr = rng.randn(5, 7) * 10
    out = logsumexp(arr, axis=0)
    naive = np.log(np.sum(np.exp(arr), axis=0))
    np.testing.assert_allclose(out, naive, rtol=1e-10)


def test_log_multivariate_normal_density_diag_matches_full_for_diag_covars():
    """Sanity check: for a diagonal covariance matrix, the 'diag' and 'full'
    code paths should agree (not a golden-fixture claim, a mathematical
    consistency check between the 2 ported code paths)."""
    rng = np.random.RandomState(1)
    n_dim = 3
    means = rng.randn(2, n_dim)
    diag_covars = np.abs(rng.randn(2, n_dim)) + 0.5
    x = rng.randn(4, n_dim)

    diag_result = log_multivariate_normal_density(x, means, diag_covars, "diag")
    full_covars = np.array([np.diag(c) for c in diag_covars])
    full_result = log_multivariate_normal_density(x, means, full_covars, "full")
    np.testing.assert_allclose(diag_result, full_result, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Unpickling correctness (network -- needs the real .pkl bytes)
# ---------------------------------------------------------------------------
def _arr_digest(arr):
    arr = np.ascontiguousarray(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


@pytest.fixture(scope="module")
def patterns_structural_digest_fixture():
    with open(FIXTURES_DIR / "patterns_structural_digest.json") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def _patterns_ready():
    """Downloads (or reuses the local cache for) PATTERNS_BALLROOM.
    Deliberately NOT module-level eager code -- see test_ml_nn.py's
    identical fixture for why."""
    from madmom_infer.models import patterns_ballroom

    try:
        return patterns_ballroom()
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"could not download PATTERNS_BALLROOM: {exc}")


@pytest.mark.network
def test_unpickled_patterns_structurally_match_real_madmom(
    patterns_structural_digest_fixture, _patterns_ready
):
    from madmom_infer.ml.nn.unpickle import load_model

    for p_idx, pattern_path in enumerate(_patterns_ready):
        pattern = load_model(pattern_path)
        expected = patterns_structural_digest_fixture[f"pattern{p_idx}"]
        assert pattern["num_beats"] == expected["num_beats"]
        assert len(pattern["gmms"]) == expected["num_gmms"]
        for gmm, exp_gmm in zip(pattern["gmms"], expected["gmms"]):
            assert gmm.n_components == exp_gmm["n_components"]
            assert gmm.covariance_type == exp_gmm["covariance_type"]
            assert _arr_digest(gmm.means) == exp_gmm["means"]
            assert _arr_digest(gmm.covars) == exp_gmm["covars"]
            assert _arr_digest(gmm.weights) == exp_gmm["weights"]


# ---------------------------------------------------------------------------
# 3. Cross-BLAS exactness (the strongest claim)
# ---------------------------------------------------------------------------
def _reference_python_available():
    return REFERENCE_PYTHON.exists()


def _upstream_patterns_available():
    return (
        UPSTREAM_PATTERNS_DIR.exists()
        and (UPSTREAM_PATTERNS_DIR / "ballroom_pattern_3_4.pkl").exists()
        and (UPSTREAM_PATTERNS_DIR / "ballroom_pattern_4_4.pkl").exists()
    )


@pytest.mark.skipif(
    not _reference_python_available(),
    reason="reference madmom install (madmom-reference/.venv) not found on "
           "this machine; the cross-BLAS proof requires it",
)
@pytest.mark.skipif(
    not _upstream_patterns_available(),
    reason="local ../madmom-upstream/madmom/models/patterns checkout not "
           "found; the cross-BLAS proof needs it (no network required this "
           "way, direct .pkl paths)",
)
def test_gmm_score_is_exact_under_original_blas():
    """This port's own `GMM.score`/`score_samples`, run under the ORIGINAL
    reference venv's numpy/scipy build (loading the real `PATTERNS_BALLROOM`
    `.pkl` files via this project's own `SafeUnpickler`), reproduce real
    madmom's `GMM.score`/`score_samples` output with ZERO differing
    elements, for every GMM in both pattern files, on a fixed random query
    array.
    """
    pattern_paths = [
        str(UPSTREAM_PATTERNS_DIR / "ballroom_pattern_3_4.pkl"),
        str(UPSTREAM_PATTERNS_DIR / "ballroom_pattern_4_4.pkl"),
    ]
    script = f"""
import sys, pickle, warnings
sys.path.insert(0, {str(REPO_ROOT)!r})
warnings.filterwarnings("ignore")
import numpy as np
from madmom_infer.ml.nn.unpickle import load_model

pattern_paths = {pattern_paths!r}
rng = np.random.RandomState(20260713)

for path in pattern_paths:
    with open(path, "rb") as fh:
        real_pattern = pickle.load(fh, encoding="latin1")
    ported_pattern = load_model(path)
    assert ported_pattern["num_beats"] == real_pattern["num_beats"]
    for real_gmm, ported_gmm in zip(real_pattern["gmms"], ported_pattern["gmms"]):
        x = rng.randn(10, real_gmm.means.shape[1]).astype(np.float64)
        real_score = real_gmm.score(x)
        ported_score = ported_gmm.score(x)
        assert np.array_equal(real_score, ported_score), f"{{path}}: score differs"
        real_lp, real_resp = real_gmm.score_samples(x)
        ported_lp, ported_resp = ported_gmm.score_samples(x)
        assert np.array_equal(real_lp, ported_lp), f"{{path}}: log_prob differs"
        assert np.array_equal(real_resp, ported_resp), f"{{path}}: responsibilities differ"
print("EXACT_MATCH")
"""
    proc = subprocess.run(
        [str(REFERENCE_PYTHON), "-c", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "EXACT_MATCH" in proc.stdout
