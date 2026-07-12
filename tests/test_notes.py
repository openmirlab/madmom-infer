"""Golden-fixture tests for Wave 4e's `madmom_infer.features.notes`/
`madmom_infer.features.notes_hmm` modules -- `RNNPianoNoteProcessor`/
`CNNPianoNoteProcessor` (audio -> note activations), `NoteOnsetPeakPicking
Processor`/`NotePeakPickingProcessor` (activations -> onset events), and
`ADSRNoteTrackingProcessor` (CNN activations -> note segments, via the new
`notes_hmm.py` ADSR HMM state space on `ml/hmm.py`'s existing Viterbi
decoder). Also covers `ml/nn/layers.py`'s 2 new layer classes this wave
ports, `ReshapeLayer`/`TransposeLayer`.

Several independent things are verified here:

1. **Per-layer-type forward-pass correctness, fully OFFLINE**
   (`tests/fixtures/notes_layers.npz` + `notes_layer_params.json`,
   `ReshapeLayer`/`TransposeLayer` -- no trainable weights, so no need for
   the weights-included self-contained shape `test_key.py` uses).
2. **Synthetic decode-logic correctness, fully OFFLINE**
   (`notes_adsr_synthetic.npz`/`notes_peak_picking_synthetic.npz`) -- the
   real-audio end-to-end fixtures below decode to EMPTY output on every
   available test wav (confirmed empirically, see `tools/
   generate_notes_fixtures.py`'s module header), so these hand-crafted
   activation arrays are what actually exercises `ADSRNoteTrackingProcessor`/
   `NoteOnsetPeakPickingProcessor`'s branch logic (including one
   deliberately INCOMPLETE note that must be discarded).
3. **Unpickling correctness** (network): `notes_brnn.pkl`'s flat layer list
   and `notes_cnn.pkl`'s nested `SequentialProcessor`/`ParallelProcessor`
   graph (the real surprise this wave found, see `ml/nn/unpickle.py`'s
   header) both reconstruct structurally identical to real madmom's own
   unpickling.
4. **End-to-end forward-pass correctness** (network): real audio through
   `RNNPianoNoteProcessor`/`CNNPianoNoteProcessor` reproduces real madmom's
   activations within documented tolerances, and the EMPTY decoded output
   (peak-picking, ADSR) matches exactly.
5. **Cross-BLAS exactness** (the strongest claim): both processors, run
   under the ORIGINAL reference venv, reproduce real madmom's activations
   AND decoded notes with ZERO differing elements.

**Fresh-processor-instance-per-case discipline, same as `tools/
generate_notes_fixtures.py`**: every test/subprocess-script below that
touches `RNNPianoNoteProcessor`/`CNNPianoNoteProcessor` builds a FRESH
instance per wav case, never a shared one reused in a loop -- confirmed
empirically during this wave's own testing that reusing one instance across
`mono_44100`/`stereo_44100`/`float32_44100` (differing dtypes) silently
produces a materially WRONG `float32_44100` activation array from real
madmom itself (a real upstream `FilteredSpectrogramProcessor`/
`ShortTimeFourierTransformProcessor` instance-reuse caching artifact, same
category already documented in those modules' headers and in 4d's
`RNNBarProcessor` fixture, not a fixture-vs-port algorithmic mismatch).

Downloads real weights via `madmom_infer.models.notes_brnn()`/`notes_cnn()`
for tests 3-4 -- network-touching and NON-COMMERCIAL-licensed (CC BY-NC-SA
4.0), see that module's header.

Reads: madmom_infer/features/notes.py, madmom_infer/features/notes_hmm.py,
madmom_infer/ml/nn/layers.py (ReshapeLayer, TransposeLayer),
madmom_infer/models.py, tests/fixtures/notes_layers.npz, tests/fixtures/
notes_layer_params.json, tests/fixtures/notes_structural_digest.json,
tests/fixtures/notes_end_to_end.npz, tests/fixtures/notes_adsr_synthetic.npz,
tests/fixtures/notes_peak_picking_synthetic.npz
"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from madmom_infer.features.notes import (
    ADSRNoteTrackingProcessor, NoteOnsetPeakPickingProcessor,
    NotePeakPickingProcessor, _cnn_pad,
)
from madmom_infer.features.notes_hmm import (
    ADSRObservationModel, ADSRStateSpace, ADSRTransitionModel,
)
from madmom_infer.ml.nn.layers import ReshapeLayer, TransposeLayer

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
WAVS_DIR = FIXTURES_DIR / "wavs"
REPO_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_PYTHON = Path(
    "/home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python"
)

# 44.1kHz-native cases only -- see module header (no resampling support).
CASES = ("mono_44100", "stereo_44100", "float32_44100")

# RNN activations are near-zero-centered (raw linear-layer output, no
# sigmoid squash -- see RNNPianoNoteProcessor's own upstream docstring
# example, values like -0.00014/0.0002); measured in-process (differing-
# BLAS-build) max abs diff is ~7.15e-7 across all 3 cases -- an ULP-view
# metric is unstable that close to zero (small absolute diffs blow up into
# millions of "ULPs" purely from the exponent being small), so this test
# uses an absolute tolerance instead, same precedent as test_chroma.py's
# documented (not ULP) SemitoneBandpassSpectrogram tolerance.
RNN_ATOL = 1e-5  # ~14x the measured 7.15e-7 worst case

# CNN activations are a deep multi-layer conv/dense forward pass, same
# shape of claim as test_key.py's/test_onsets.py's CNN ULP margins; measured
# in-process max is 247 ULP across all 3 cases.
CNN_MAX_ULP = 1024  # ~4x the measured 247 worst case, matching repo convention


# ---------------------------------------------------------------------------
# 0. notes_hmm.py -- pure-logic sanity checks, no fixture needed
# ---------------------------------------------------------------------------
def test_adsr_state_space_layout():
    st = ADSRStateSpace(attack_length=2, decay_length=2, release_length=1)
    assert st.silence == 0
    assert st.attack == 1
    assert st.decay == 3
    assert st.sustain == 5
    assert st.release == 6
    assert st.num_states == 7


def test_adsr_transition_model_is_a_probability_distribution():
    """`ADSRTransitionModel.__init__` builds its transitions as
    `(from_state, to_state, prob)` triples and hands them to
    `TransitionModel.make_sparse`, which itself already asserts (raises
    `ValueError` otherwise) that every state's OUTGOING transition
    probabilities sum to 1 -- so successfully constructing an
    `ADSRTransitionModel` at all is already the interesting assertion here.
    This test additionally checks the resulting CSR shape is sane."""
    st = ADSRStateSpace(attack_length=2, decay_length=2, release_length=1)
    tm = ADSRTransitionModel(st, onset_prob=0.8, note_prob=0.8,
                              offset_prob=0.2)
    assert tm.num_states == st.num_states
    assert len(tm.pointers) == st.num_states + 1
    assert tm.num_transitions == len(tm.probabilities) == len(tm.states)
    # a default end_prob=1.0 produces one legitimate explicit-zero
    # self-loop (release -> release, prob 1 - end_prob = 0), so >= 0 not > 0
    assert np.all(tm.probabilities >= 0)
    assert np.all(tm.probabilities <= 1)


def test_adsr_observation_model_pointers():
    st = ADSRStateSpace(attack_length=2, decay_length=2, release_length=1)
    om = ADSRObservationModel(st)
    # silence -> density col 0, attack -> 1, decay/sustain -> 2, release -> 3
    assert om.pointers[st.silence] == 0
    assert om.pointers[st.attack] == 1
    assert om.pointers[st.decay] == 2
    assert om.pointers[st.sustain] == 2
    assert om.pointers[st.release] == 3


def test_adsr_observation_model_log_densities():
    st = ADSRStateSpace()
    om = ADSRObservationModel(st)
    obs = np.array([[0.7, 0.3, 0.1], [0.2, 0.9, 0.05]])
    densities = om.log_densities(obs)
    assert densities.shape == (2, 4)
    np.testing.assert_allclose(densities[:, 0], np.log(1. - obs[:, 1]))
    np.testing.assert_allclose(densities[:, 1], np.log(obs[:, 1]))
    np.testing.assert_allclose(densities[:, 2], np.log(obs[:, 0]))
    np.testing.assert_allclose(densities[:, 3], np.log(obs[:, 2]))


# ---------------------------------------------------------------------------
# 1. ReshapeLayer / TransposeLayer -- offline, no weights (none needed)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def notes_layers_fixture():
    return np.load(FIXTURES_DIR / "notes_layers.npz")


@pytest.fixture(scope="module")
def notes_layer_params():
    with open(FIXTURES_DIR / "notes_layer_params.json") as fh:
        return json.load(fh)


def test_transpose_layer_matches_fixture(notes_layers_fixture, notes_layer_params):
    params = notes_layer_params["TransposeLayer"]
    axes = tuple(params["axes"]) if params["axes"] is not None else None
    layer = TransposeLayer(axes=axes)
    out = layer.activate(notes_layers_fixture["TransposeLayer_input"])
    expected = notes_layers_fixture["TransposeLayer_output"]
    assert out.shape == expected.shape
    assert out.dtype == expected.dtype
    np.testing.assert_array_equal(out, expected)


def test_reshape_layer_matches_fixture(notes_layers_fixture, notes_layer_params):
    params = notes_layer_params["ReshapeLayer"]
    newshape = params["newshape"]
    if isinstance(newshape, list):
        newshape = tuple(newshape)
    layer = ReshapeLayer(newshape=newshape, order=params["order"])
    out = layer.activate(notes_layers_fixture["ReshapeLayer_input"])
    expected = notes_layers_fixture["ReshapeLayer_output"]
    assert out.shape == expected.shape
    assert out.dtype == expected.dtype
    np.testing.assert_array_equal(out, expected)


def test_cnn_pad_matches_upstream_shape():
    """`_cnn_pad` repeats first/last frame 5 times each -- offline, no
    fixture needed (pure shape/value logic, verbatim port)."""
    data = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    out = _cnn_pad(data)
    assert out.shape == (13, 4)
    np.testing.assert_array_equal(out[:5], np.tile(data[0], (5, 1)))
    np.testing.assert_array_equal(out[-5:], np.tile(data[-1], (5, 1)))
    np.testing.assert_array_equal(out[5:8], data)


# ---------------------------------------------------------------------------
# 2. Synthetic decode-logic fixtures -- offline, no weights/network needed
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def adsr_synthetic_fixture():
    return np.load(FIXTURES_DIR / "notes_adsr_synthetic.npz")


def test_adsr_note_tracking_synthetic_matches_fixture(adsr_synthetic_fixture):
    activations = adsr_synthetic_fixture["activations"]
    expected = adsr_synthetic_fixture["notes"]
    proc = ADSRNoteTrackingProcessor()
    notes = proc(activations)
    # 2 complete notes kept, 1 incomplete note correctly discarded
    assert notes.shape[0] == 2
    np.testing.assert_array_equal(notes, expected)


@pytest.fixture(scope="module")
def peak_picking_synthetic_fixture():
    return np.load(FIXTURES_DIR / "notes_peak_picking_synthetic.npz")


def test_note_onset_peak_picking_synthetic_matches_fixture(
    peak_picking_synthetic_fixture,
):
    activations = peak_picking_synthetic_fixture["activations"]
    expected = peak_picking_synthetic_fixture["onset_notes"]
    proc = NoteOnsetPeakPickingProcessor(fps=100, pitch_offset=21)
    notes = proc(activations)
    assert notes.shape[0] == 3
    np.testing.assert_array_equal(notes, expected)


def test_note_peak_picking_deprecated_alias_synthetic_matches_fixture(
    peak_picking_synthetic_fixture,
):
    activations = peak_picking_synthetic_fixture["activations"]
    expected = peak_picking_synthetic_fixture["deprecated_notes"]
    proc = NotePeakPickingProcessor()
    notes = proc(activations)
    np.testing.assert_array_equal(notes, expected)


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
    """Independent reimplementation of tools/generate_notes_fixtures.py's
    digest_layer -- deliberately not imported from tools/, same discipline
    as test_key.py's own copy."""
    t = type(layer).__name__
    d = {"type": t}
    if hasattr(layer, "weights"):
        d["weights"] = _arr_digest(layer.weights)
    if hasattr(layer, "bias"):
        d["bias"] = _arr_digest(layer.bias)
    if t in ("Gate", "Cell", "RecurrentLayer"):
        if hasattr(layer, "recurrent_weights"):
            d["recurrent_weights"] = _arr_digest(layer.recurrent_weights)
    if getattr(layer, "activation_fn", None) is not None:
        d["activation_fn"] = layer.activation_fn.__name__
    if t == "BidirectionalLayer":
        d["fwd_layer"] = digest_layer(layer.fwd_layer)
        d["bwd_layer"] = digest_layer(layer.bwd_layer)
    elif t == "ConvolutionalLayer":
        d["stride"] = layer.stride
        d["pad"] = layer.pad
    elif t == "BatchNormLayer":
        d["beta"] = _arr_digest(layer.beta)
        d["gamma"] = _arr_digest(layer.gamma)
        d["mean"] = _arr_digest(layer.mean)
        d["inv_std"] = _arr_digest(layer.inv_std)
    elif t == "ReshapeLayer":
        d["newshape"] = list(layer.newshape) if isinstance(
            layer.newshape, (list, tuple)) else layer.newshape
        d["order"] = layer.order
    elif t == "TransposeLayer":
        d["axes"] = list(layer.axes) if layer.axes is not None else None
    return d


def digest_processor(obj):
    """Independent reimplementation of tools/generate_notes_fixtures.py's
    digest_processor."""
    from madmom_infer.processors import ParallelProcessor, SequentialProcessor

    if isinstance(obj, (ParallelProcessor, SequentialProcessor)):
        type_name = ("ParallelProcessor" if isinstance(obj, ParallelProcessor)
                      else "SequentialProcessor")
        return {"type": type_name,
                "processors": [digest_processor(p) for p in obj.processors]}
    if callable(obj) and not hasattr(obj, "activate"):
        return {"type": "function", "name": getattr(obj, "__name__", str(obj))}
    return digest_layer(obj)


@pytest.fixture(scope="module")
def model_paths():
    """Downloads (or reuses the local cache for) notes_brnn.pkl/
    notes_cnn.pkl. Deliberately NOT module-level eager code, same reason as
    test_key.py's identical fixture."""
    from madmom_infer.models import notes_brnn, notes_cnn

    try:
        return {"brnn": notes_brnn(), "cnn": notes_cnn()}
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"could not download NOTES_* weights: {exc}")


@pytest.fixture(scope="module")
def structural_digest_fixture():
    with open(FIXTURES_DIR / "notes_structural_digest.json") as fh:
        return json.load(fh)


@pytest.mark.network
def test_unpickled_notes_brnn_structurally_matches_real_madmom(
    structural_digest_fixture, model_paths
):
    from madmom_infer.ml.nn.unpickle import load_model

    assert len(model_paths["brnn"]) == 1
    nn = load_model(model_paths["brnn"][0])
    ours = [digest_layer(l) for l in nn.layers]
    expected = structural_digest_fixture["notes_brnn"]
    assert ours == expected


@pytest.mark.network
def test_unpickled_notes_cnn_structurally_matches_real_madmom(
    structural_digest_fixture, model_paths
):
    """The interesting one -- confirms this port's own SafeUnpickler
    reconstructs the FULL SequentialProcessor/ParallelProcessor graph
    notes_cnn.pkl actually pickles (see ml/nn/unpickle.py's header), not
    just a bare NeuralNetwork."""
    from madmom_infer.ml.nn.unpickle import load_model

    assert len(model_paths["cnn"]) == 1
    graph = load_model(model_paths["cnn"][0])
    ours = digest_processor(graph)
    expected = structural_digest_fixture["notes_cnn"]
    assert ours == expected


# ---------------------------------------------------------------------------
# 4. End-to-end forward-pass correctness (network)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _notes_models_ready():
    from madmom_infer.models import notes_brnn, notes_cnn

    try:
        return {"brnn": notes_brnn(), "cnn": notes_cnn()}
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"could not download NOTES_* weights: {exc}")


@pytest.fixture(scope="module")
def notes_end_to_end_fixture():
    return np.load(FIXTURES_DIR / "notes_end_to_end.npz")


@pytest.mark.network
@pytest.mark.parametrize("case", CASES)
def test_rnn_piano_note_activations_match_fixture_within_atol(
    notes_end_to_end_fixture, _notes_models_ready, case
):
    from madmom_infer.features.notes import RNNPianoNoteProcessor

    proc = RNNPianoNoteProcessor()
    act = proc(str(WAVS_DIR / f"{case}.wav"))
    expected = notes_end_to_end_fixture[f"{case}_rnn_activations"]
    assert act.shape == expected.shape
    assert act.dtype == expected.dtype
    np.testing.assert_allclose(act, expected, atol=RNN_ATOL, rtol=0)


@pytest.mark.network
@pytest.mark.parametrize("case", CASES)
def test_cnn_piano_note_activations_match_fixture_within_ulp(
    notes_end_to_end_fixture, _notes_models_ready, case
):
    from madmom_infer.features.notes import CNNPianoNoteProcessor

    proc = CNNPianoNoteProcessor()
    act = proc(str(WAVS_DIR / f"{case}.wav"))
    expected = notes_end_to_end_fixture[f"{case}_cnn_activations"]
    assert act.shape == expected.shape
    assert act.dtype == expected.dtype
    np.testing.assert_array_max_ulp(act, expected, maxulp=CNN_MAX_ULP)


@pytest.mark.network
@pytest.mark.parametrize("case", CASES)
def test_note_decode_pipeline_exact(
    notes_end_to_end_fixture, _notes_models_ready, case
):
    """Decoded output (peak-picked onset notes, deprecated-alias notes, ADSR
    notes) must be EXACT even though the underlying activations are only
    ULP/atol-close -- both this port and real madmom decode to EMPTY output
    on every available test wav (see tools/generate_notes_fixtures.py's
    module header), so this specifically pins that empty-but-correct
    result, not a numerically-interesting decode."""
    from madmom_infer.features.notes import (
        ADSRNoteTrackingProcessor, CNNPianoNoteProcessor,
        NoteOnsetPeakPickingProcessor, NotePeakPickingProcessor,
        RNNPianoNoteProcessor,
    )

    wav_path = str(WAVS_DIR / f"{case}.wav")
    rnn_act = RNNPianoNoteProcessor()(wav_path)
    cnn_act = CNNPianoNoteProcessor()(wav_path)

    onset_notes = NoteOnsetPeakPickingProcessor(fps=100, pitch_offset=21)(rnn_act)
    deprecated_notes = NotePeakPickingProcessor()(rnn_act)
    adsr_notes = ADSRNoteTrackingProcessor()(cnn_act)

    np.testing.assert_array_equal(
        onset_notes, notes_end_to_end_fixture[f"{case}_onset_notes"])
    np.testing.assert_array_equal(
        deprecated_notes, notes_end_to_end_fixture[f"{case}_deprecated_notes"])
    np.testing.assert_array_equal(
        adsr_notes, notes_end_to_end_fixture[f"{case}_adsr_notes"])


# ---------------------------------------------------------------------------
# 5. Cross-BLAS exactness (the strongest claim)
# ---------------------------------------------------------------------------
def _reference_python_available():
    return REFERENCE_PYTHON.exists()


@pytest.mark.skipif(
    not _reference_python_available(),
    reason="reference madmom install (madmom-reference/.venv) not found on "
           "this machine; the cross-BLAS proof requires it",
)
def test_full_pipeline_is_exact_under_original_blas():
    """This port's own `RNNPianoNoteProcessor`/`CNNPianoNoteProcessor`, run
    under the ORIGINAL reference venv's numpy/scipy build, reproduce real
    madmom's activations AND decoded notes with ZERO differing elements, for
    all 3 cases -- proving the ULP/atol drift measured above is BLAS
    non-associativity, not an algorithmic difference in this port's new
    ReshapeLayer/TransposeLayer or the notes_cnn.pkl processor-graph
    unpickling. Uses local `../madmom-upstream` `.pkl` copies directly, no
    network needed.
    """
    upstream_brnn = (
        REPO_ROOT.parent / "madmom-upstream" / "madmom" / "models" / "notes"
        / "2013" / "notes_brnn.pkl"
    )
    upstream_cnn = (
        REPO_ROOT.parent / "madmom-upstream" / "madmom" / "models" / "notes"
        / "2019" / "notes_cnn.pkl"
    )
    for p in (upstream_brnn, upstream_cnn):
        if not p.exists():
            pytest.skip(f"local model file not found at {p}")

    case_paths = ", ".join(repr(str(WAVS_DIR / f"{c}.wav")) for c in CASES)
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import numpy as np
from madmom_infer.features.notes import (
    ADSRNoteTrackingProcessor, CNNPianoNoteProcessor,
    NoteOnsetPeakPickingProcessor, NotePeakPickingProcessor,
    RNNPianoNoteProcessor,
)

cases = {list(CASES)!r}
wav_paths = [{case_paths}]
fixture = np.load({str(FIXTURES_DIR / "notes_end_to_end.npz")!r})

# FRESH processor instances per case -- see tools/generate_notes_fixtures.py's
# module header for why (a real, confirmed instance-reuse caching artifact,
# not a fixture-vs-port algorithmic mismatch).
for case, wav_path in zip(cases, wav_paths):
    rnn = RNNPianoNoteProcessor(nn_file={str(upstream_brnn)!r})
    cnn = CNNPianoNoteProcessor(nn_files=[{str(upstream_cnn)!r}])
    onset_pp = NoteOnsetPeakPickingProcessor(fps=100, pitch_offset=21)
    deprecated_pp = NotePeakPickingProcessor()
    adsr = ADSRNoteTrackingProcessor()

    rnn_act = rnn(wav_path)
    cnn_act = cnn(wav_path)
    onset_notes = onset_pp(rnn_act)
    deprecated_notes = deprecated_pp(rnn_act)
    adsr_notes = adsr(cnn_act)

    assert np.array_equal(rnn_act, fixture[case + "_rnn_activations"]), \\
        f"{{case}}: RNN activations differ"
    assert np.array_equal(cnn_act, fixture[case + "_cnn_activations"]), \\
        f"{{case}}: CNN activations differ"
    assert np.array_equal(onset_notes, fixture[case + "_onset_notes"]), \\
        f"{{case}}: onset notes differ"
    assert np.array_equal(deprecated_notes, fixture[case + "_deprecated_notes"]), \\
        f"{{case}}: deprecated-alias notes differ"
    assert np.array_equal(adsr_notes, fixture[case + "_adsr_notes"]), \\
        f"{{case}}: ADSR notes differ"
print("EXACT_MATCH")
"""
    proc = subprocess.run(
        [str(REFERENCE_PYTHON), "-c", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "EXACT_MATCH" in proc.stdout


@pytest.mark.skipif(
    not _reference_python_available(),
    reason="reference madmom install (madmom-reference/.venv) not found on "
           "this machine; the cross-BLAS proof requires it",
)
def test_adsr_synthetic_decode_is_exact_under_original_blas(
    adsr_synthetic_fixture,
):
    """Same synthetic activation array, decoded under the reference venv --
    proves ADSRNoteTrackingProcessor's own segmentation logic (not just the
    CNN forward pass feeding it) is exact, independent of any real model
    weights."""
    activations = adsr_synthetic_fixture["activations"]
    npy_path = FIXTURES_DIR / "_adsr_synthetic_input_scratch.npy"
    np.save(npy_path, activations)
    try:
        script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import numpy as np
from madmom_infer.features.notes import ADSRNoteTrackingProcessor

activations = np.load({str(npy_path)!r})
fixture = np.load({str(FIXTURES_DIR / "notes_adsr_synthetic.npz")!r})
proc = ADSRNoteTrackingProcessor()
notes = proc(activations)
assert np.array_equal(notes, fixture["notes"]), "synthetic ADSR notes differ"
print("EXACT_MATCH")
"""
        proc = subprocess.run(
            [str(REFERENCE_PYTHON), "-c", script],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "EXACT_MATCH" in proc.stdout
    finally:
        npy_path.unlink(missing_ok=True)
