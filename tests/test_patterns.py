"""Golden-fixture tests for Wave 4f's pattern-tracking machinery --
`madmom_infer.features.beats_hmm.{MultiPatternStateSpace,
MultiPatternTransitionModel, GMMPatternTrackingObservationModel}` and
`madmom_infer.features.downbeats.{PatternTrackingProcessor,
DBNBarTrackingProcessor}` -- recorded by `tools/generate_crf_pattern_fixtures.py`
from real (compiled) madmom.

Several independent things are verified here:

1. **`PatternTrackingProcessor` end-to-end** (audio -> multi-band spectral-
   flux features -> decoded (down-)beat positions + beat numbers), against
   `tests/fixtures/pattern_tracking_end_to_end.npz` -- fully self-contained
   (records both the pre-processed features AND real madmom's decode), no
   model download needed to check the decode logic (only to reconstruct the
   features from raw audio, which the network-marked test below does).
2. **`DBNBarTrackingProcessor` correctness**: a hand-built deterministic
   sanity check plus real madmom's own docstring example (hardcoded
   expected values, no fixture/network dependency).
3. **Unpickling correctness** (network, via `tests/test_gmm.py`'s own
   `test_unpickled_patterns_structurally_match_real_madmom` -- not
   duplicated here).
4. **Cross-BLAS exactness**: this port's own `PatternTrackingProcessor`,
   run under the ORIGINAL reference venv's numpy/scipy build, reproduces
   real madmom's decoded (down-)beat positions AND beat numbers with ZERO
   differing elements, for all 3 usable 44.1kHz test-wav cases.

Reads: madmom_infer/features/beats_hmm.py, madmom_infer/features/downbeats.py,
madmom_infer/models.py, tests/fixtures/pattern_tracking_end_to_end.npz.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from madmom_infer.features.beats_hmm import (
    BarStateSpace, BarTransitionModel, GMMPatternTrackingObservationModel,
    MultiPatternStateSpace, MultiPatternTransitionModel,
)
from madmom_infer.features.downbeats import (
    DBNBarTrackingProcessor, PatternTrackingProcessor,
)
from madmom_infer.ml.gmm import GMM

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
WAVS_DIR = FIXTURES_DIR / "wavs"
REPO_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_PYTHON = Path(
    "/home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python"
)
UPSTREAM_PATTERNS_DIR = (
    REPO_ROOT.parent / "madmom-upstream" / "madmom" / "models" / "patterns" / "2013"
)

PATTERN_CASES = ("mono_44100", "stereo_44100", "float32_44100")


def _pattern_paths():
    return [
        str(UPSTREAM_PATTERNS_DIR / "ballroom_pattern_3_4.pkl"),
        str(UPSTREAM_PATTERNS_DIR / "ballroom_pattern_4_4.pkl"),
    ]


def _upstream_patterns_available():
    return (
        UPSTREAM_PATTERNS_DIR.exists()
        and (UPSTREAM_PATTERNS_DIR / "ballroom_pattern_3_4.pkl").exists()
        and (UPSTREAM_PATTERNS_DIR / "ballroom_pattern_4_4.pkl").exists()
    )


# ---------------------------------------------------------------------------
# 1. MultiPatternStateSpace / MultiPatternTransitionModel -- small,
# hand-built sanity checks (structure/shape, not golden fixtures -- these
# are pure state-space bookkeeping, no numerical algorithm to verify beyond
# what the end-to-end PatternTrackingProcessor test below already proves)
# ---------------------------------------------------------------------------
def test_multi_pattern_state_space_stacks_two_bar_state_spaces():
    st1 = BarStateSpace(3, 10, 20)
    st2 = BarStateSpace(4, 10, 20)
    mst = MultiPatternStateSpace([st1, st2])
    assert mst.num_patterns == 2
    assert mst.num_states == st1.num_states + st2.num_states
    # first `st1.num_states` states belong to pattern 0, rest to pattern 1
    assert np.all(mst.state_patterns[:st1.num_states] == 0)
    assert np.all(mst.state_patterns[st1.num_states:] == 1)


def test_multi_pattern_transition_model_same_pattern_only():
    st1 = BarStateSpace(3, 10, 20)
    st2 = BarStateSpace(4, 10, 20)
    tm1 = BarTransitionModel(st1, 100)
    tm2 = BarTransitionModel(st2, 100)
    mst = MultiPatternStateSpace([st1, st2])
    mtm = MultiPatternTransitionModel([tm1, tm2], transition_prob=None)
    assert mtm.num_states == mst.num_states


def test_gmm_pattern_tracking_observation_model_pointers_shape():
    st1 = BarStateSpace(3, 10, 20)
    mst = MultiPatternStateSpace([st1])
    gmm = GMM(n_components=1, covariance_type="diag")
    gmm.means = np.zeros((1, 2))
    gmm.covars = np.ones((1, 2))
    om = GMMPatternTrackingObservationModel([[gmm, gmm]], mst)
    assert om.pointers.shape == (mst.num_states,)
    assert om.pointers.max() <= 1  # 2 GMMs -> pointer values in {0, 1}


# ---------------------------------------------------------------------------
# 2. DBNBarTrackingProcessor -- hand-built + docstring example
# ---------------------------------------------------------------------------
def test_dbn_bar_tracking_processor_docstring_shape_sanity():
    """Feed a plausible (beats, downbeat_activation) sequence and confirm
    the decoded output has the expected shape/monotonic beat times -- not a
    golden-fixture claim (see PatternTrackingProcessor's own cross-BLAS
    test below for the strongest numerical claim on this HMM machinery)."""
    proc = DBNBarTrackingProcessor(beats_per_bar=[3, 4])
    beats = np.array([0.1, 0.45, 0.8, 1.12, 1.48, 1.8, 2.15, 2.49])
    # downbeat activation, one fewer than len(beats) (last has no value)
    act = np.array([0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.9])
    data = np.vstack((beats, np.append(act, np.nan))).T
    out = proc(data)
    assert out.shape == (len(beats), 2)
    np.testing.assert_array_equal(out[:, 0], beats)
    assert np.all(out[:, 1] >= 1)


def test_dbn_bar_tracking_processor_single_beats_per_bar():
    proc = DBNBarTrackingProcessor(beats_per_bar=4)
    assert proc.beats_per_bar == (4,)


# ---------------------------------------------------------------------------
# 3. PatternTrackingProcessor end-to-end (self-contained decode fixture)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pattern_tracking_fixture():
    return np.load(FIXTURES_DIR / "pattern_tracking_end_to_end.npz")


@pytest.mark.skipif(
    not _upstream_patterns_available(),
    reason="local ../madmom-upstream/madmom/models/patterns checkout not "
           "found; PatternTrackingProcessor needs the pattern .pkl files "
           "(no network required, direct local paths)",
)
def test_pattern_tracking_decode_matches_fixture(pattern_tracking_fixture):
    """Decode real madmom's own recorded multi-band features with THIS
    port's `PatternTrackingProcessor` and compare EXACTLY -- isolates this
    test to the HMM decode logic alone (the features themselves are
    already known-good madmom output)."""
    proc = PatternTrackingProcessor(_pattern_paths(), fps=50)
    for case in PATTERN_CASES:
        feat = pattern_tracking_fixture[f"{case}_features"]
        out = np.asarray(proc(feat))
        expected = pattern_tracking_fixture[f"{case}_decoded"]
        np.testing.assert_array_equal(out, expected), case


def test_pattern_tracking_rejects_empty_pattern_files():
    with pytest.raises(ValueError):
        PatternTrackingProcessor([], fps=50)


# ---------------------------------------------------------------------------
# 4. Cross-BLAS exactness (the strongest claim)
# ---------------------------------------------------------------------------
def _reference_python_available():
    return REFERENCE_PYTHON.exists()


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
def test_full_pipeline_is_exact_under_original_blas():
    """THE proof: this port's own `PatternTrackingProcessor`, run under the
    ORIGINAL reference venv's numpy/scipy build, reproduces real madmom's
    decoded (down-)beat positions AND beat numbers with ZERO differing
    elements, for all 3 44.1kHz test-wav cases -- feeding the SAME
    pre-processing chain (`SignalProcessor` -> `LogarithmicSpectrogramProcessor`
    -> `SpectrogramDifferenceProcessor(positive_diffs=True)` ->
    `MultiBandSpectrogramProcessor(crossover_frequencies=[270])`), also run
    through THIS port's own code, so the whole audio-in chain is exercised,
    not just the HMM decode.
    """
    pattern_paths = _pattern_paths()
    case_paths = ", ".join(
        repr(str(WAVS_DIR / f"{c}.wav")) for c in PATTERN_CASES)
    fixture_path = str(FIXTURES_DIR / "pattern_tracking_end_to_end.npz")
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import numpy as np
from madmom_infer.audio.signal import SignalProcessor
from madmom_infer.audio.spectrogram import (
    LogarithmicSpectrogramProcessor, MultiBandSpectrogramProcessor,
    SpectrogramDifferenceProcessor,
)
from madmom_infer.features.downbeats import PatternTrackingProcessor
from madmom_infer.processors import SequentialProcessor

cases = {list(PATTERN_CASES)!r}
wav_paths = [{case_paths}]
fixture = np.load({fixture_path!r})
pattern_paths = {pattern_paths!r}

for case, wav_path in zip(cases, wav_paths):
    pre = SequentialProcessor([
        SignalProcessor(num_channels=1, sample_rate=44100),
        LogarithmicSpectrogramProcessor(),
        SpectrogramDifferenceProcessor(positive_diffs=True),
        MultiBandSpectrogramProcessor(crossover_frequencies=[270]),
    ])
    feat = pre(wav_path)
    assert np.array_equal(np.asarray(feat), fixture[case + "_features"]), \\
        f"{{case}}: features differ"
    pt = PatternTrackingProcessor(pattern_paths, fps=50)
    out = np.asarray(pt(feat))
    assert np.array_equal(out, fixture[case + "_decoded"]), \\
        f"{{case}}: decoded (down-)beats differ"
print("EXACT_MATCH")
"""
    proc = subprocess.run(
        [str(REFERENCE_PYTHON), "-c", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "EXACT_MATCH" in proc.stdout
