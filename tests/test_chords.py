"""Golden-fixture tests for Wave 4d's `madmom_infer.features.chords` module
-- `DeepChromaChordRecognitionProcessor` (`DeepChromaProcessor` -> CRF) and
`CNNChordFeatureProcessor` + `CRFChordRecognitionProcessor` (CNN features ->
CRF), both chained end-to-end: real audio in, chord segments (start, end,
label) out. `ConditionalRandomField` itself is covered independently by
`tests/test_crf.py`; this file focuses on the surrounding chord-specific
plumbing (`majmin_targets_to_chord_labels`, `CNNChordFeatureProcessor`'s CNN
feature extraction, and the two full end-to-end chains).

**Chord segment boundaries and labels are EXACT, not ULP-bounded** -- the
task's own stated acceptance bar for this wave, and empirically true here:
`ConditionalRandomField.process` decodes an integer state-id sequence
(argmax over 25 classes), which absorbs the ULP-level noise in its
`DeepChromaProcessor`/`CNNChordFeatureProcessor` input, same "decode is
integer-domain, should absorb noise" pattern already established by
`test_key.py`'s decoded label / `test_downbeats_rnn.py`'s decoded beat
times -- confirmed here for all 3 usable 44.1kHz test-wav cases, both
recognition paths.

`CNNChordFeatureProcessor`'s own raw feature output (before CRF decoding)
is a real CNN forward pass -- ULP-bounded, not exact, same shape of claim
as `test_key.py`'s CNN layers.

Reads: madmom_infer/features/chords.py, madmom_infer/audio/chroma.py,
tests/fixtures/chords_end_to_end.npz
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from madmom_infer.audio.chroma import DeepChromaProcessor
from madmom_infer.features.chords import (
    CNNChordFeatureProcessor, CRFChordRecognitionProcessor,
    DeepChromaChordRecognitionProcessor, majmin_targets_to_chord_labels,
)
from madmom_infer.processors import SequentialProcessor

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
WAVS_DIR = FIXTURES_DIR / "wavs"
REPO_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_PYTHON = Path(
    "/home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python"
)

CASES = ("mono_44100", "stereo_44100", "float32_44100")


# ---------------------------------------------------------------------------
# majmin_targets_to_chord_labels -- pure function, no network/model needed
# ---------------------------------------------------------------------------
def test_majmin_targets_to_chord_labels_docstring_example():
    """Verbatim port of `madmom.features.chords.majmin_targets_to_chord_
    labels`'s own docstring hand-worked example -- hardcoded, no fixture."""
    # 0..11 = A..G# major, 12..23 = A..G# minor, 24 = no-chord
    targets = [24, 24, 0, 0, 0, 12]  # N, N, A:maj, A:maj, A:maj, A:min
    labels = majmin_targets_to_chord_labels(targets, fps=2)
    assert list(labels["label"]) == ["N", "A:maj", "A:min"]
    np.testing.assert_allclose(labels["start"], [0.0, 1.0, 2.5])
    np.testing.assert_allclose(labels["end"], [1.0, 2.5, 3.0])


def test_majmin_targets_to_chord_labels_no_chord_id():
    labels = majmin_targets_to_chord_labels([24], fps=10)
    assert list(labels["label"]) == ["N"]


# ---------------------------------------------------------------------------
# CNNChordFeatureProcessor -- network (needs real chords_cnnfeat.pkl bytes)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def chords_end_to_end_fixture():
    return np.load(FIXTURES_DIR / "chords_end_to_end.npz")


@pytest.fixture(scope="module")
def _chords_models_ready():
    from madmom_infer.models import chords_cfcrf, chords_cnn_feat, chords_dccrf

    try:
        return {
            "dccrf": chords_dccrf(),
            "cnn_feat": chords_cnn_feat(),
            "cfcrf": chords_cfcrf(),
        }
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"could not download CHORDS_* weights: {exc}")


@pytest.mark.network
@pytest.mark.parametrize("case", CASES)
def test_deep_chroma_chord_recognition_end_to_end_exact(
    chords_end_to_end_fixture, _chords_models_ready, case
):
    dcp = DeepChromaProcessor()
    decode = DeepChromaChordRecognitionProcessor()
    chain = SequentialProcessor([dcp, decode])
    out = chain(str(WAVS_DIR / f"{case}.wav"))

    expected_start = chords_end_to_end_fixture[f"{case}_dccrf_start"]
    expected_end = chords_end_to_end_fixture[f"{case}_dccrf_end"]
    expected_label = chords_end_to_end_fixture[f"{case}_dccrf_label"]
    np.testing.assert_array_equal(np.asarray(out["start"]), expected_start)
    np.testing.assert_array_equal(np.asarray(out["end"]), expected_end)
    np.testing.assert_array_equal(
        np.asarray([str(x) for x in out["label"]]), expected_label)


@pytest.mark.network
@pytest.mark.parametrize("case", CASES)
def test_cnn_feature_crf_chord_recognition_end_to_end_exact(
    chords_end_to_end_fixture, _chords_models_ready, case
):
    featproc = CNNChordFeatureProcessor()
    decode = CRFChordRecognitionProcessor()
    chain = SequentialProcessor([featproc, decode])
    out = chain(str(WAVS_DIR / f"{case}.wav"))

    expected_start = chords_end_to_end_fixture[f"{case}_cfcrf_start"]
    expected_end = chords_end_to_end_fixture[f"{case}_cfcrf_end"]
    expected_label = chords_end_to_end_fixture[f"{case}_cfcrf_label"]
    np.testing.assert_array_equal(np.asarray(out["start"]), expected_start)
    np.testing.assert_array_equal(np.asarray(out["end"]), expected_end)
    np.testing.assert_array_equal(
        np.asarray([str(x) for x in out["label"]]), expected_label)


def _reference_python_available():
    return REFERENCE_PYTHON.exists()


@pytest.mark.skipif(
    not _reference_python_available(),
    reason="reference madmom install (madmom-reference/.venv) not found on "
           "this machine; the cross-BLAS proof requires it",
)
def test_chord_recognition_is_exact_under_original_blas():
    """This port's own full audio-in-to-chord-segments pipelines (BOTH
    `DeepChromaChordRecognitionProcessor` and `CNNChordFeatureProcessor` +
    `CRFChordRecognitionProcessor`), run under the ORIGINAL reference
    venv's numpy/scipy build, reproduce real madmom's decoded chord
    segments (start, end, label) with ZERO differing elements, for all 3
    cases. Uses local `../madmom-upstream` `.pkl` copies directly, no
    network needed.
    """
    upstream_chroma = (
        REPO_ROOT.parent / "madmom-upstream" / "madmom" / "models" / "chroma"
        / "2016" / "chroma_dnn.pkl"
    )
    upstream_chords = (
        REPO_ROOT.parent / "madmom-upstream" / "madmom" / "models" / "chords"
        / "2016"
    )
    dccrf_path = upstream_chords / "chords_dccrf.pkl"
    cnnfeat_path = upstream_chords / "chords_cnnfeat.pkl"
    cfcrf_path = upstream_chords / "chords_cnncrf.pkl"
    for p in (upstream_chroma, dccrf_path, cnnfeat_path, cfcrf_path):
        if not p.exists():
            pytest.skip(f"local model file not found at {p}")

    case_paths = ", ".join(repr(str(WAVS_DIR / f"{c}.wav")) for c in CASES)
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import numpy as np
from madmom_infer.audio.chroma import DeepChromaProcessor
from madmom_infer.features.chords import (
    CNNChordFeatureProcessor, CRFChordRecognitionProcessor,
    DeepChromaChordRecognitionProcessor,
)
from madmom_infer.processors import SequentialProcessor

cases = {list(CASES)!r}
wav_paths = [{case_paths}]
fixture = np.load({str(FIXTURES_DIR / "chords_end_to_end.npz")!r})

for case, wav_path in zip(cases, wav_paths):
    dcp = DeepChromaProcessor(models=[{str(upstream_chroma)!r}])
    decode = DeepChromaChordRecognitionProcessor(model={str(dccrf_path)!r})
    out = SequentialProcessor([dcp, decode])(wav_path)
    assert np.array_equal(np.asarray(out["start"]), fixture[case + "_dccrf_start"])
    assert np.array_equal(np.asarray(out["end"]), fixture[case + "_dccrf_end"])
    assert np.array_equal([str(x) for x in out["label"]],
                          list(fixture[case + "_dccrf_label"]))

    featproc = CNNChordFeatureProcessor(nn_file={str(cnnfeat_path)!r})
    decode2 = CRFChordRecognitionProcessor(model={str(cfcrf_path)!r})
    out2 = SequentialProcessor([featproc, decode2])(wav_path)
    assert np.array_equal(np.asarray(out2["start"]), fixture[case + "_cfcrf_start"])
    assert np.array_equal(np.asarray(out2["end"]), fixture[case + "_cfcrf_end"])
    assert np.array_equal([str(x) for x in out2["label"]],
                          list(fixture[case + "_cfcrf_label"]))
print("EXACT_MATCH")
"""
    proc = subprocess.run(
        [str(REFERENCE_PYTHON), "-c", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "EXACT_MATCH" in proc.stdout
