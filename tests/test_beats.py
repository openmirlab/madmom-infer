"""Golden-fixture tests for Wave 4c: `madmom_infer.features.beats` --
`RNNBeatProcessor`, `DBNBeatTrackingProcessor` (beat-only), and
`MultiModelSelectionProcessor` -- all recorded by `tools/
generate_beat_tempo_fixtures.py` from real (compiled) madmom.

Several independent things are verified here:

1. **`MultiModelSelectionProcessor` correctness, fully OFFLINE**: a
   self-contained fixture (the 8 per-network `BEATS_BLSTM` predictions for
   `mono_44100.wav` + real madmom's own selection) -- no model download,
   no unpickling.
2. **`DBNBeatTrackingProcessor` correctness**: a hand-built deterministic
   sanity check (both `correct=True` and `correct=False` branches) plus
   fixture-based decode of real `RNNBeatProcessor` activations.
3. **Unpickling correctness** (network): `beats_lstm_1.pkl`/
   `beats_blstm_1.pkl` structural digests match real madmom's own
   unpickling exactly.
4. **RNN end-to-end activation correctness + decoded-beat-time exactness**
   (network): `RNNBeatProcessor`(online=False/True) activations match
   within a documented ULP bound; `DBNBeatTrackingProcessor` decoded beat
   TIMES are bit-exact (same shape of claim as `test_downbeats_rnn.py`'s/
   `test_onsets.py`'s decoded discrete outputs).
5. **Cross-BLAS exactness** (the strongest claim): this port's own
   `RNNBeatProcessor` + `DBNBeatTrackingProcessor`, run under the ORIGINAL
   reference venv's numpy/scipy build, reproduce real madmom's activations
   AND decoded beat times with ZERO differing elements.

**Same "shared-instance-in-order" caching-gotcha discipline as
`test_downbeats_rnn.py`/`test_onsets.py`**: `RNNBeatProcessor` builds ONE
`ShortTimeFourierTransformProcessor`/`FilteredSpectrogramProcessor` PER
FRAME-SIZE BRANCH inside `__init__` and reuses them across calls --
`tools/generate_beat_tempo_fixtures.py` processes all 3 cases through ONE
shared `RNNBeatProcessor(online=False)`/`RNNBeatProcessor(online=True)`
instance, in order (`mono_44100` -> `stereo_44100` -> `float32_44100`);
every network test below replicates that exact call order/instance-reuse.

Reads: madmom_infer/features/beats.py, madmom_infer/models.py,
tests/fixtures/beats_structural_digest.json,
tests/fixtures/beats_activations.npz,
tests/fixtures/beats_multimodel_selection.npz.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from madmom_infer.features.beats import (
    DBNBeatTrackingProcessor, MultiModelSelectionProcessor,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
WAVS_DIR = FIXTURES_DIR / "wavs"
REPO_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_PYTHON = Path(
    "/home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python"
)

# 44.1kHz-native cases only -- see module header (no resampling support).
BEAT_CASES = ("mono_44100", "stereo_44100", "float32_44100")

UPSTREAM_BEATS_LSTM_DIR = (
    REPO_ROOT.parent / "madmom-upstream" / "madmom" / "models" / "beats" / "2016"
)
UPSTREAM_BEATS_BLSTM_DIR = (
    REPO_ROOT.parent / "madmom-upstream" / "madmom" / "models" / "beats" / "2015"
)

# Measured worst case for the BLSTM/LSTM activations is well within the
# same order of magnitude as test_downbeats_rnn.py's 190/512 for its own
# (bigger, 8-network) ensemble -- 512 is a generous, consistent margin.
MAX_ULP_NN = 512


# ---------------------------------------------------------------------------
# 1. MultiModelSelectionProcessor (offline, self-contained fixture)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mm_fixture():
    return np.load(FIXTURES_DIR / "beats_multimodel_selection.npz")


def test_multi_model_selection_matches_fixture(mm_fixture):
    predictions = [mm_fixture[f"prediction_{i}"] for i in range(8)]
    mm = MultiModelSelectionProcessor(num_ref_predictions=None)
    out = mm(predictions)
    np.testing.assert_array_equal(out, mm_fixture["selected"])


def test_multi_model_selection_with_explicit_reference():
    """Small, hand-constructed sanity check of the `num_ref_predictions > 0`
    branch (not fixture-based -- the selection LOGIC's correctness doesn't
    depend on real madmom's trained weights)."""
    reference = np.array([1.0, 1.0, 1.0, 1.0])
    close = np.array([1.0, 1.0, 1.0, 0.9])
    far = np.array([0.0, 0.0, 0.0, 0.0])
    mm = MultiModelSelectionProcessor(num_ref_predictions=1)
    out = mm([reference, close, far])
    np.testing.assert_array_equal(out, close)


def test_multi_model_selection_rejects_negative_num_ref_predictions():
    mm = MultiModelSelectionProcessor(num_ref_predictions=-1)
    with pytest.raises(ValueError):
        mm([np.zeros(4), np.zeros(4)])


# ---------------------------------------------------------------------------
# 2. DBNBeatTrackingProcessor correctness
# ---------------------------------------------------------------------------
def _synthetic_beat_activations(n_frames=500, fps=100, bpm=120, seed=0):
    """A plausible (never exactly 0/1) periodic gaussian-pulse beat
    activation array -- `RNNBeatTrackingObservationModel.log_densities`
    takes `log(observations)`/`log(1 - observations)`, both -inf/nan at the
    0/1 boundary, so this avoids that edge case (same technique as
    `tools/generate_fixtures.py`'s `generate_dbn_fixtures`)."""
    rng = np.random.default_rng(seed)
    act = np.full(n_frames, 0.02, dtype=np.float64)
    beat_period = int(fps * 60 / bpm)
    width = 3
    for i in range(0, n_frames, beat_period):
        for w in range(-width, width + 1):
            idx = i + w
            if 0 <= idx < n_frames:
                act[idx] = max(act[idx], 0.02 + 0.6 * np.exp(-0.5 * (w / 1.2) ** 2))
    act += rng.normal(0, 0.005, size=act.shape)
    act = np.clip(act, 0.001, 0.97).astype(np.float32)
    return act


def test_dbn_beat_tracking_processor_correct_true():
    proc = DBNBeatTrackingProcessor(fps=100)
    act = _synthetic_beat_activations(bpm=120)
    out = proc(act)
    assert out.ndim == 1
    assert len(out) > 0
    # beats should be roughly evenly spaced near the 0.5s period (120bpm)
    np.testing.assert_allclose(np.diff(out), 0.5, atol=0.05)


def test_dbn_beat_tracking_processor_correct_false():
    proc = DBNBeatTrackingProcessor(fps=100, correct=False)
    act = _synthetic_beat_activations(bpm=120)
    out = proc(act)
    assert out.ndim == 1
    assert len(out) > 0


def test_dbn_beat_tracking_processor_empty_activations_returns_no_beats():
    proc = DBNBeatTrackingProcessor(fps=100)
    out = proc(np.zeros(100, dtype=np.float32))
    assert len(out) == 0


@pytest.fixture(scope="module")
def beats_activations_fixture():
    return np.load(FIXTURES_DIR / "beats_activations.npz")


def test_dbn_decode_of_real_activations_matches_fixture(beats_activations_fixture):
    """Decode real madmom's own recorded BLSTM/LSTM activations with THIS
    port's DBNBeatTrackingProcessor and compare beat times exactly (the
    activations themselves are already known-good madmom output, isolating
    this test to the decoder alone)."""
    dbn = DBNBeatTrackingProcessor(fps=100)
    for case in BEAT_CASES:
        for model in ("blstm", "lstm"):
            act = beats_activations_fixture[f"{case}_{model}_activations"]
            out = np.asarray(dbn(act))
            expected = beats_activations_fixture[f"{case}_{model}_beat_times"]
            np.testing.assert_array_equal(out, expected), f"{case}/{model}"


# ---------------------------------------------------------------------------
# 3. Unpickling correctness (network -- needs the real .pkl bytes)
# ---------------------------------------------------------------------------
def _arr_digest(arr):
    arr = np.ascontiguousarray(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def digest_layer(layer):
    """Independent reimplementation of
    tools/generate_beat_tempo_fixtures.py's digest_layer -- deliberately not
    imported from tools/, same discipline as test_key.py's/test_onsets.py's
    own copies."""
    t = type(layer).__name__
    d = {"type": t}
    if hasattr(layer, "weights"):
        d["weights"] = _arr_digest(layer.weights)
    if hasattr(layer, "bias"):
        d["bias"] = _arr_digest(layer.bias)
    if t in ("Gate", "Cell", "GRUCell", "RecurrentLayer"):
        if hasattr(layer, "recurrent_weights"):
            d["recurrent_weights"] = _arr_digest(layer.recurrent_weights)
    if t == "Gate" and getattr(layer, "peephole_weights", None) is not None:
        d["peephole_weights"] = _arr_digest(layer.peephole_weights)
    if getattr(layer, "activation_fn", None) is not None:
        d["activation_fn"] = layer.activation_fn.__name__
    if t == "BidirectionalLayer":
        d["fwd_layer"] = digest_layer(layer.fwd_layer)
        d["bwd_layer"] = digest_layer(layer.bwd_layer)
    elif t == "LSTMLayer":
        d["input_gate"] = digest_layer(layer.input_gate)
        d["forget_gate"] = digest_layer(layer.forget_gate)
        d["cell"] = digest_layer(layer.cell)
        d["output_gate"] = digest_layer(layer.output_gate)
    return d


@pytest.fixture(scope="module")
def structural_digest_fixture():
    with open(FIXTURES_DIR / "beats_structural_digest.json") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def _beats_models_ready():
    """Downloads (or reuses the local cache for) both beat model families.
    Deliberately NOT module-level eager code -- see test_ml_nn.py's
    identical fixture for why."""
    from madmom_infer.models import beats_blstm, beats_lstm

    try:
        return {"lstm": beats_lstm(), "blstm": beats_blstm()}
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"could not download beat model weights: {exc}")


@pytest.mark.network
def test_unpickled_beats_lstm_structurally_matches_real_madmom(
    structural_digest_fixture, _beats_models_ready
):
    from madmom_infer.ml.nn.unpickle import load_model

    nn = load_model(_beats_models_ready["lstm"][0])
    ours = [digest_layer(l) for l in nn.layers]
    assert ours == structural_digest_fixture["beats_lstm_1"]


@pytest.mark.network
def test_unpickled_beats_blstm_structurally_matches_real_madmom(
    structural_digest_fixture, _beats_models_ready
):
    from madmom_infer.ml.nn.unpickle import load_model

    nn = load_model(_beats_models_ready["blstm"][0])
    ours = [digest_layer(l) for l in nn.layers]
    assert ours == structural_digest_fixture["beats_blstm_1"]


# ---------------------------------------------------------------------------
# 4. End-to-end activation correctness + decoded-beat-time exactness
# (network) -- shared-instance-in-order, see module header
# ---------------------------------------------------------------------------
@pytest.mark.network
def test_rnn_beat_activations_and_dbn_decode_match_fixture(
    beats_activations_fixture, _beats_models_ready
):
    from madmom_infer.features.beats import RNNBeatProcessor

    blstm = RNNBeatProcessor(online=False)
    lstm = RNNBeatProcessor(online=True)
    dbn = DBNBeatTrackingProcessor(fps=100)
    for case in BEAT_CASES:
        wav_path = str(WAVS_DIR / f"{case}.wav")
        act_blstm = blstm(wav_path)
        act_lstm = lstm(wav_path)
        expected_blstm = beats_activations_fixture[f"{case}_blstm_activations"]
        expected_lstm = beats_activations_fixture[f"{case}_lstm_activations"]
        assert act_blstm.shape == expected_blstm.shape, case
        assert act_blstm.dtype == expected_blstm.dtype, case
        np.testing.assert_array_max_ulp(act_blstm, expected_blstm,
                                         maxulp=MAX_ULP_NN)
        np.testing.assert_array_max_ulp(act_lstm, expected_lstm,
                                         maxulp=MAX_ULP_NN)

        beats_blstm = np.asarray(dbn(act_blstm))
        beats_lstm = np.asarray(dbn(act_lstm))
        assert np.array_equal(
            beats_blstm, beats_activations_fixture[f"{case}_blstm_beat_times"]
        ), case
        assert np.array_equal(
            beats_lstm, beats_activations_fixture[f"{case}_lstm_beat_times"]
        ), case


# ---------------------------------------------------------------------------
# 5. Cross-BLAS exactness (the strongest claim)
# ---------------------------------------------------------------------------
def _reference_python_available():
    return REFERENCE_PYTHON.exists()


def _upstream_beats_models_available():
    return (
        UPSTREAM_BEATS_LSTM_DIR.exists()
        and (UPSTREAM_BEATS_LSTM_DIR / "beats_lstm_1.pkl").exists()
        and UPSTREAM_BEATS_BLSTM_DIR.exists()
        and (UPSTREAM_BEATS_BLSTM_DIR / "beats_blstm_1.pkl").exists()
    )


@pytest.mark.skipif(
    not _reference_python_available(),
    reason="reference madmom install (madmom-reference/.venv) not found on "
           "this machine; the cross-BLAS proof requires it",
)
@pytest.mark.skipif(
    not _upstream_beats_models_available(),
    reason="local ../madmom-upstream/madmom/models/beats checkout not "
           "found; the cross-BLAS proof needs it (no network required this "
           "way, see nn_files= override)",
)
def test_full_pipeline_is_exact_under_original_blas():
    """THE proof: this port's own `RNNBeatProcessor`(online=False/True) +
    `DBNBeatTrackingProcessor`, run under the ORIGINAL reference venv's
    numpy/scipy build, reproduce real madmom's activations AND decoded beat
    times with ZERO differing elements, for all 3 cases and both model
    families. Uses the local `../madmom-upstream` `.pkl` copies directly
    (`nn_files=` override) so this test needs neither network nor a prior
    `-m network` run.
    """
    lstm_paths = [str(UPSTREAM_BEATS_LSTM_DIR / f"beats_lstm_{i}.pkl")
                  for i in range(1, 9)]
    blstm_paths = [str(UPSTREAM_BEATS_BLSTM_DIR / f"beats_blstm_{i}.pkl")
                   for i in range(1, 9)]

    case_paths = ", ".join(repr(str(WAVS_DIR / f"{c}.wav")) for c in BEAT_CASES)
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import numpy as np
from madmom_infer.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor

cases = {list(BEAT_CASES)!r}
wav_paths = [{case_paths}]
blstm = RNNBeatProcessor(online=False, nn_files={blstm_paths!r})
lstm = RNNBeatProcessor(online=True, nn_files={lstm_paths!r})
dbn = DBNBeatTrackingProcessor(fps=100)
fixture = np.load({str(FIXTURES_DIR / "beats_activations.npz")!r})

for case, wav_path in zip(cases, wav_paths):
    act_blstm = blstm(wav_path)
    act_lstm = lstm(wav_path)
    beats_blstm = np.asarray(dbn(act_blstm))
    beats_lstm = np.asarray(dbn(act_lstm))
    assert np.array_equal(act_blstm, fixture[case + "_blstm_activations"]), \\
        f"{{case}}: blstm activations differ"
    assert np.array_equal(act_lstm, fixture[case + "_lstm_activations"]), \\
        f"{{case}}: lstm activations differ"
    assert np.array_equal(beats_blstm, fixture[case + "_blstm_beat_times"]), \\
        f"{{case}}: blstm beat times differ"
    assert np.array_equal(beats_lstm, fixture[case + "_lstm_beat_times"]), \\
        f"{{case}}: lstm beat times differ"
print("EXACT_MATCH")
"""
    proc = subprocess.run(
        [str(REFERENCE_PYTHON), "-c", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "EXACT_MATCH" in proc.stdout
