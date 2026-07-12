"""Golden-fixture tests for `madmom_infer.ml.crf.ConditionalRandomField` --
Wave 4d's numpy-Viterbi CRF decoder, the chord-decoding backend for
`features/chords.py`'s `DeepChromaChordRecognitionProcessor`/
`CRFChordRecognitionProcessor`.

Three independent things are verified here:

1. **The docstring example** (hardcoded expected values, no fixture/network
   dependency at all) -- same discipline as `test_hmm.py`'s docstring test.
2. **Direct decode on REAL chord-feature data** (`tests/fixtures/
   crf_decode.npz`, recorded by `tools/generate_chroma_chord_fixtures.py`):
   this port's own `ConditionalRandomField.load()` + `.process()`, fed real
   madmom's own `chords_dccrf.pkl`/`chords_cnncrf.pkl` weights AND real
   madmom's own `DeepChromaProcessor`/`CNNChordFeatureProcessor` observation
   sequences, must decode the EXACT same state-id sequence (`y_star`) AND
   the exact same merged chord-label segments as real madmom -- `y_star` is
   an integer argmax-over-classes decode, not a float, so this is a
   no-tolerance, `np.array_equal` claim, not ULP-bounded.
3. **Cross-BLAS exactness**: this port's own CRF, run under the reference
   venv, reproduces the same decode -- since `ConditionalRandomField.process`
   never calls BLAS-backed matrix ops beyond one small `np.dot` per frame
   (12 or 128 x 25), this is expected (and verified) to be bit-identical,
   not merely close, matching `test_comb_filters.py`'s "no
   summation-order non-associativity to average away" precedent.

Reads: madmom_infer/ml/crf.py, madmom_infer/ml/nn/unpickle.py,
tests/fixtures/crf_decode.npz
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from madmom_infer.ml.crf import ConditionalRandomField

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_PYTHON = Path(
    "/home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python"
)
UPSTREAM_CHORDS_DIR = (
    REPO_ROOT.parent / "madmom-upstream" / "madmom" / "models" / "chords" / "2016"
)


def test_docstring_example_exact():
    """madmom's own ConditionalRandomField doctest example
    (`madmom-upstream/madmom/ml/crf.py:47-72`) -- a hand-built 3-state CRF
    emulating a discrete HMM, hardcoded expected decode, no fixture/network
    dependency."""
    eta = np.spacing(1)
    initial = np.log(np.array([0.7, 0.2, 0.1]) + eta)
    final = np.ones(3)
    bias = np.ones(3)
    transition = np.log(np.array([[0.6, 0.2, 0.2],
                                  [0.1, 0.7, 0.2],
                                  [0.1, 0.1, 0.8]]) + eta)
    observation = np.log(np.array([[0.9, 0.5, 0.1],
                                   [0.1, 0.5, 0.1]]) + eta)
    crf = ConditionalRandomField(initial, final, bias, transition, observation)

    obs = np.array([[1, 0], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1]])
    y_star = crf.process(obs)
    np.testing.assert_array_equal(y_star, [0, 0, 1, 1, 1, 1])
    assert y_star.dtype == np.uint32


@pytest.fixture(scope="module")
def crf_decode_fixture():
    return np.load(FIXTURES_DIR / "crf_decode.npz")


@pytest.fixture(scope="module")
def _chords_models_ready():
    """Downloads (or reuses the local cache for) CHORDS_DCCRF/CHORDS_CFCRF.
    Deliberately NOT module-level eager code -- see test_ml_nn.py's
    identical fixture for why."""
    from madmom_infer.models import chords_cfcrf, chords_dccrf

    try:
        return {"dccrf": chords_dccrf(), "cfcrf": chords_cfcrf()}
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"could not download CHORDS_DCCRF/CHORDS_CFCRF weights: {exc}")


@pytest.mark.network
def test_dccrf_decode_matches_fixture_exact(crf_decode_fixture, _chords_models_ready):
    from madmom_infer.features.chords import majmin_targets_to_chord_labels

    crf = ConditionalRandomField.load(_chords_models_ready["dccrf"][0])
    observations = crf_decode_fixture["dccrf_observations"]
    y_star = crf.process(observations)
    np.testing.assert_array_equal(y_star, crf_decode_fixture["dccrf_y_star"])

    labels = majmin_targets_to_chord_labels(y_star, fps=10)
    np.testing.assert_array_equal(
        np.asarray(labels["start"]), crf_decode_fixture["dccrf_labels_start"])
    np.testing.assert_array_equal(
        np.asarray(labels["end"]), crf_decode_fixture["dccrf_labels_end"])
    np.testing.assert_array_equal(
        np.asarray([str(x) for x in labels["label"]]),
        crf_decode_fixture["dccrf_labels_label"])


@pytest.mark.network
def test_cfcrf_decode_matches_fixture_exact(crf_decode_fixture, _chords_models_ready):
    from madmom_infer.features.chords import majmin_targets_to_chord_labels

    crf = ConditionalRandomField.load(_chords_models_ready["cfcrf"][0])
    observations = crf_decode_fixture["cfcrf_observations"]
    y_star = crf.process(observations)
    np.testing.assert_array_equal(y_star, crf_decode_fixture["cfcrf_y_star"])

    labels = majmin_targets_to_chord_labels(y_star, fps=10)
    np.testing.assert_array_equal(
        np.asarray(labels["start"]), crf_decode_fixture["cfcrf_labels_start"])
    np.testing.assert_array_equal(
        np.asarray(labels["end"]), crf_decode_fixture["cfcrf_labels_end"])
    np.testing.assert_array_equal(
        np.asarray([str(x) for x in labels["label"]]),
        crf_decode_fixture["cfcrf_labels_label"])


def _reference_python_available():
    return REFERENCE_PYTHON.exists()


def _upstream_chords_available():
    return (
        UPSTREAM_CHORDS_DIR.exists()
        and (UPSTREAM_CHORDS_DIR / "chords_dccrf.pkl").exists()
        and (UPSTREAM_CHORDS_DIR / "chords_cnncrf.pkl").exists()
    )


@pytest.mark.skipif(
    not _reference_python_available(),
    reason="reference madmom install (madmom-reference/.venv) not found on "
           "this machine; the cross-BLAS proof requires it",
)
@pytest.mark.skipif(
    not _upstream_chords_available(),
    reason="local ../madmom-upstream/madmom/models/chords checkout not "
           "found; the cross-BLAS proof needs it (no network required this "
           "way, direct .pkl paths)",
)
def test_crf_decode_is_exact_under_original_blas():
    """This port's own `ConditionalRandomField`, run under the ORIGINAL
    reference venv's numpy/scipy build, reproduces real madmom's decoded
    state sequence for BOTH CRF models with ZERO differing elements. Since
    `process()` only ever does one small `(num_frames,)x(12 or 128, 25)`
    matmul per frame (not a large BLAS-backed batch operation), this is
    expected to be bit-identical, not merely ULP-close -- verified here.
    """
    dccrf_path = str(UPSTREAM_CHORDS_DIR / "chords_dccrf.pkl")
    cfcrf_path = str(UPSTREAM_CHORDS_DIR / "chords_cnncrf.pkl")
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import numpy as np
from madmom_infer.ml.crf import ConditionalRandomField
from madmom_infer.features.chords import majmin_targets_to_chord_labels

fixture = np.load({str(FIXTURES_DIR / "crf_decode.npz")!r})

dccrf = ConditionalRandomField.load({dccrf_path!r})
dccrf_y = dccrf.process(fixture["dccrf_observations"])
assert np.array_equal(dccrf_y, fixture["dccrf_y_star"]), "dccrf y_star differs"

cfcrf = ConditionalRandomField.load({cfcrf_path!r})
cfcrf_y = cfcrf.process(fixture["cfcrf_observations"])
assert np.array_equal(cfcrf_y, fixture["cfcrf_y_star"]), "cfcrf y_star differs"
print("EXACT_MATCH")
"""
    proc = subprocess.run(
        [str(REFERENCE_PYTHON), "-c", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "EXACT_MATCH" in proc.stdout
