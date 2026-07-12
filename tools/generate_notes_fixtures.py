"""Wave-4e golden-fixture generator: `notes_brnn.pkl`/`notes_cnn.pkl`
structural digests, per-layer-type golden (input, output) fixtures for the 2
new layer classes this wave ports (`ReshapeLayer`, `TransposeLayer`),
`RNNPianoNoteProcessor`/`CNNPianoNoteProcessor` end-to-end activations (all 3
usable 44.1kHz test wavs), `NoteOnsetPeakPickingProcessor`/
`NotePeakPickingProcessor` decoded notes from those RNN activations,
`ADSRNoteTrackingProcessor` decoded notes from those CNN activations, PLUS
two SYNTHETIC (hand-crafted, deterministic, no real audio) fixtures that
actually exercise the peak-picking/ADSR-decode logic non-trivially -- the
4e sibling of `tools/generate_chroma_chord_fixtures.py`, same conventions.

**Why synthetic fixtures, in addition to the real-audio ones**: the shared
44.1kHz test wavs (`mono_44100`/`stereo_44100`/`float32_44100`, short
Phase-1 test clips, not real piano recordings) produce EMPTY decoded output
from both `NoteOnsetPeakPickingProcessor` and `ADSRNoteTrackingProcessor` on
every case (confirmed empirically against the reference venv) -- a
technically-valid but weak golden fixture (an empty array trivially
"matches" almost any decode bug that also happens to produce nothing). The
synthetic fixtures below hand-craft small `[note, onset, offset]`/onset
activation arrays with clear, deliberately-shaped note envelopes (including
one INCOMPLETE note that must be discarded under `complete=True`) so the
decode logic's actual branches are exercised -- confirmed against real
madmom (reference venv) to produce 2 detected notes (the incomplete third
one correctly dropped) for the ADSR case, and 3 combined onset events for
the peak-picking case, before committing this fixture.

HOW TO RUN -- same real-madmom reference venv as prior waves
(`madmom-reference/.venv`, Python 3.10.18, numpy 1.23.5, scipy 1.15.3),
whose already-installed madmom 0.17.dev0 wheel vendors `notes/2013/
notes_brnn.pkl` and `notes/2019/notes_cnn.pkl` as package data (no network
needed to run this script):

    /home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python \\
        tools/generate_notes_fixtures.py

Reuses the same 44.1kHz-native test-wav subset established by
`tools/generate_phase2_fixtures.py` (`mono_44100`, `stereo_44100`,
`float32_44100`) -- `RNNPianoNoteProcessor`/`CNNPianoNoteProcessor` both
hard-code `SignalProcessor(sample_rate=44100)`, and this project has no
ffmpeg-backed resampling for file loading -- `stereo_48000.wav` stays out of
scope for that reason.

Reads: real `madmom` (features.notes.*, features.notes_hmm.*, models.{
NOTES_BRNN,NOTES_CNN}, ml.nn.NeuralNetwork, processors.{SequentialProcessor,
ParallelProcessor}), numpy. Writes: tests/fixtures/notes_structural_digest.
json, tests/fixtures/notes_layers.npz, tests/fixtures/notes_layer_params.
json, tests/fixtures/notes_end_to_end.npz, tests/fixtures/
notes_adsr_synthetic.npz, tests/fixtures/notes_peak_picking_synthetic.npz.
Read by: tests/test_notes.py.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
WAVS_DIR = FIXTURES_DIR / "wavs"

# 44.1kHz-native cases only -- see module header (no resampling support).
CASES = ("mono_44100", "stereo_44100", "float32_44100")


def _arr_digest(arr) -> dict:
    arr = np.ascontiguousarray(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def digest_layer(layer) -> dict:
    """Structural digest of one NN layer -- extends `tools/
    generate_phase2_fixtures.py`'s `digest_layer` (weights/bias/
    recurrent_weights/peephole_weights/BidirectionalLayer/LSTMLayer
    recursion, needed for `notes_brnn.pkl`'s RNN family) with the CNN-era
    attrs `tools/generate_key_fixtures.py`'s version adds (stride/pad,
    beta/gamma/mean/inv_std), PLUS this wave's 2 new layer types
    (`ReshapeLayer`'s `newshape`/`order`, `TransposeLayer`'s `axes`)."""
    t = type(layer).__name__
    d = {"type": t}
    if hasattr(layer, "weights"):
        d["weights"] = _arr_digest(layer.weights)
    if hasattr(layer, "bias"):
        d["bias"] = _arr_digest(layer.bias)
    if t in ("Gate", "Cell", "RecurrentLayer"):
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


def digest_processor(obj) -> dict:
    """Structural digest of a `SequentialProcessor`/`ParallelProcessor`/
    `Layer`/plain-function node -- recurses `notes_cnn.pkl`'s own nested
    multi-task processor graph (see `madmom_infer/ml/nn/unpickle.py`'s
    header for the discovery this digests: the pickle is a whole
    `SequentialProcessor`/`ParallelProcessor` tree, not a bare
    `NeuralNetwork`)."""
    from madmom.processors import ParallelProcessor, SequentialProcessor

    if isinstance(obj, ParallelProcessor):
        return {"type": "ParallelProcessor",
                "processors": [digest_processor(p) for p in obj.processors]}
    if isinstance(obj, SequentialProcessor):
        return {"type": "SequentialProcessor",
                "processors": [digest_processor(p) for p in obj.processors]}
    if callable(obj) and not hasattr(obj, "activate"):
        # plain function (e.g. numpy.dstack) -- record its name only
        return {"type": "function", "name": getattr(obj, "__name__", str(obj))}
    return digest_layer(obj)


# ---------------------------------------------------------------------------
# 1. Structural digests: notes_brnn.pkl (flat), notes_cnn.pkl (nested graph)
# ---------------------------------------------------------------------------
def generate_notes_structural_digest() -> dict:
    import madmom
    from madmom.models import NOTES_BRNN, NOTES_CNN

    assert len(NOTES_BRNN) == 1, (
        f"expected NOTES_BRNN to resolve to exactly 1 file, got "
        f"{len(NOTES_BRNN)}."
    )
    assert len(NOTES_CNN) == 1, (
        f"expected NOTES_CNN to resolve to exactly 1 file, got "
        f"{len(NOTES_CNN)}."
    )
    brnn = madmom.ml.nn.NeuralNetwork.load(NOTES_BRNN[0])
    cnn_graph = madmom.ml.nn.NeuralNetwork.load(NOTES_CNN[0])
    return {
        "notes_brnn": [digest_layer(l) for l in brnn.layers],
        "notes_cnn": digest_processor(cnn_graph),
    }


# ---------------------------------------------------------------------------
# 2. Per-layer-type golden (input, output) fixtures for ReshapeLayer/
#    TransposeLayer -- self-contained (weights not needed, these layers have
#    no trainable parameters), sampled from a real notes_cnn.pkl forward
#    pass on mono_44100.wav's pre-processed spectrogram.
# ---------------------------------------------------------------------------
def generate_notes_layer_fixtures() -> "tuple[dict, dict]":
    from madmom.features.notes import CNNPianoNoteProcessor

    proc = CNNPianoNoteProcessor()
    pre_processor, nn = proc.processors
    wav_path = WAVS_DIR / f"{CASES[0]}.wav"
    spec_input = np.asarray(pre_processor(str(wav_path)))

    # unwrap NeuralNetworkEnsemble((ParallelProcessor([graph]), avg_fn))
    par_processor = nn.processors[0]
    graph = par_processor.processors[0]
    assert len(graph.processors) == 6, (
        f"expected notes_cnn.pkl's top-level graph to have 6 stages, got "
        f"{len(graph.processors)} -- re-inspect notes_cnn.pkl's structure "
        "if this assertion fires."
    )
    batch_norm, conv1, conv2, conv3, parallel, _dstack_fn = graph.processors

    data = batch_norm(spec_input)
    data = conv1(data)
    data = conv2(data)
    data = conv3(data)

    branch0 = parallel.processors[0]
    assert [type(p).__name__ for p in branch0.processors] == [
        "ConvolutionalLayer", "TransposeLayer", "ReshapeLayer",
        "FeedForwardLayer",
    ], (
        f"expected branch 0 to be [Conv, Transpose, Reshape, FeedForward], "
        f"got {[type(p).__name__ for p in branch0.processors]} -- "
        "re-inspect notes_cnn.pkl's branch structure if this fires."
    )
    branch_conv, transpose_layer, reshape_layer, _ff = branch0.processors

    branch_data = branch_conv(data)
    transpose_out = transpose_layer(branch_data)
    reshape_out = reshape_layer(transpose_out)

    npz_payload = {
        "TransposeLayer_input": np.asarray(branch_data),
        "TransposeLayer_output": np.asarray(transpose_out),
        "ReshapeLayer_input": np.asarray(transpose_out),
        "ReshapeLayer_output": np.asarray(reshape_out),
    }
    params_json = {
        "TransposeLayer": {"axes": list(transpose_layer.axes)
                            if transpose_layer.axes is not None else None},
        "ReshapeLayer": {"newshape": list(reshape_layer.newshape)
                          if isinstance(reshape_layer.newshape, (list, tuple))
                          else reshape_layer.newshape,
                          "order": reshape_layer.order},
    }
    return npz_payload, params_json


# ---------------------------------------------------------------------------
# 3. RNNPianoNoteProcessor / CNNPianoNoteProcessor end-to-end activations +
#    NoteOnsetPeakPickingProcessor/NotePeakPickingProcessor/
#    ADSRNoteTrackingProcessor decoded output on those (real-audio) cases.
# ---------------------------------------------------------------------------
def generate_notes_end_to_end_fixtures() -> dict:
    """**Deliberate deviation from `generate_key_fixtures.py`'s/
    `generate_beat_tempo_fixtures.py`'s "shared-instance-in-order"
    discipline, noted explicitly, not silently** -- same rationale as
    `generate_chroma_chord_fixtures.py`'s `RNNBarProcessor` fixture (see
    that tool's module header): FRESH `RNNPianoNoteProcessor`/
    `CNNPianoNoteProcessor` instances are built PER CASE here. Confirmed
    empirically DURING this wave's own testing: reusing one shared instance
    across `mono_44100`/`stereo_44100`/`float32_44100` (differing dtypes)
    silently produced a real, differently-shaped `float32_44100` activation
    array from real madmom itself (max abs diff ~0.097 against the
    fresh-instance recording for the SAME wav+weights) -- a real upstream
    instance-reuse caching artifact (this project's `FilteredSpectrogram
    Processor`/`ShortTimeFourierTransformProcessor` caching gotchas, see
    those modules' headers), not a fixture-vs-port algorithmic mismatch.
    Since this is a NEW fixture set, fresh instances per case sidestep the
    caching-gotcha minefield entirely rather than faithfully reproducing
    it; `tests/test_notes.py` uses this same fresh-per-case discipline, so
    the comparison stays apples-to-apples.
    """
    from madmom.features.notes import (
        ADSRNoteTrackingProcessor, CNNPianoNoteProcessor,
        NoteOnsetPeakPickingProcessor, NotePeakPickingProcessor,
        RNNPianoNoteProcessor,
    )

    out = {}
    for case in CASES:
        wav_path = str(WAVS_DIR / f"{case}.wav")
        rnn = RNNPianoNoteProcessor()
        cnn = CNNPianoNoteProcessor()
        onset_pp = NoteOnsetPeakPickingProcessor(fps=100, pitch_offset=21)
        deprecated_pp = NotePeakPickingProcessor()
        adsr = ADSRNoteTrackingProcessor()

        rnn_act = rnn(wav_path)
        cnn_act = cnn(wav_path)
        onset_notes = onset_pp(rnn_act)
        deprecated_notes = deprecated_pp(rnn_act)
        adsr_notes = adsr(cnn_act)

        out[f"{case}_rnn_activations"] = np.asarray(rnn_act)
        out[f"{case}_cnn_activations"] = np.asarray(cnn_act)
        out[f"{case}_onset_notes"] = np.asarray(onset_notes)
        out[f"{case}_deprecated_notes"] = np.asarray(deprecated_notes)
        out[f"{case}_adsr_notes"] = np.asarray(adsr_notes)
    return out


# ---------------------------------------------------------------------------
# 4. Synthetic ADSR decode fixture -- see module header for why.
# ---------------------------------------------------------------------------
def _make_synthetic_adsr_activations() -> np.ndarray:
    n_frames, n_pitches = 60, 88
    act = np.zeros((n_frames, n_pitches, 3), dtype=np.float32)

    def add_note(pitch, onset_start, onset_len, note_len, offset_len):
        act[onset_start:onset_start + onset_len, pitch, 1] = 0.9
        note_start = onset_start
        note_end = onset_start + onset_len + note_len
        act[note_start:note_end, pitch, 0] = 0.85
        off_start = note_end
        act[off_start:off_start + offset_len, pitch, 2] = 0.8

    add_note(10, 5, 3, 15, 3)   # complete note -- should be kept
    add_note(40, 20, 4, 10, 2)  # complete note -- should be kept
    add_note(55, 0, 3, 5, 2)    # starts at frame 0 -- incomplete, discarded
    return act


def generate_notes_adsr_synthetic_fixture() -> dict:
    from madmom.features.notes import ADSRNoteTrackingProcessor

    activations = _make_synthetic_adsr_activations()
    adsr = ADSRNoteTrackingProcessor()
    notes = adsr(activations)
    return {"activations": activations, "notes": np.asarray(notes)}


# ---------------------------------------------------------------------------
# 5. Synthetic peak-picking fixture -- see module header for why.
# ---------------------------------------------------------------------------
def _make_synthetic_peak_activations() -> np.ndarray:
    n_frames, n_pitches = 50, 88
    act = np.zeros((n_frames, n_pitches), dtype=np.float32)
    act[10, 20] = 0.9
    act[10, 21] = 0.3
    act[30, 20] = 0.8
    act[15, 60] = 0.95
    return act


def generate_notes_peak_picking_synthetic_fixture() -> dict:
    from madmom.features.notes import (
        NoteOnsetPeakPickingProcessor, NotePeakPickingProcessor,
    )

    activations = _make_synthetic_peak_activations()
    onset_pp = NoteOnsetPeakPickingProcessor(fps=100, pitch_offset=21)
    deprecated_pp = NotePeakPickingProcessor()
    return {
        "activations": activations,
        "onset_notes": np.asarray(onset_pp(activations)),
        "deprecated_notes": np.asarray(deprecated_pp(activations)),
    }


def main() -> None:
    try:
        import madmom  # noqa: F401
    except ImportError as exc:
        print(
            "ERROR: this script needs the real `madmom` package (not "
            "madmom_infer), including its NOTES_BRNN/NOTES_CNN pretrained "
            "model weights. Run it with the madmom-reference venv's "
            "interpreter -- see this file's module docstring for the exact "
            "command.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    if not WAVS_DIR.exists() or not any(WAVS_DIR.glob("*.wav")):
        print(
            "ERROR: tests/fixtures/wavs/ is empty -- run "
            "tools/generate_fixtures.py first.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"Using madmom {madmom.__version__} from {madmom.__file__}")

    print("4e-1: notes_brnn.pkl/notes_cnn.pkl structural digests ...")
    digest = generate_notes_structural_digest()
    (FIXTURES_DIR / "notes_structural_digest.json").write_text(
        json.dumps(digest, indent=2, sort_keys=True) + "\n"
    )

    print("4e-2: ReshapeLayer/TransposeLayer golden fixtures ...")
    layer_fixtures, layer_params = generate_notes_layer_fixtures()
    np.savez_compressed(FIXTURES_DIR / "notes_layers.npz", **layer_fixtures)
    (FIXTURES_DIR / "notes_layer_params.json").write_text(
        json.dumps(layer_params, indent=2, sort_keys=True) + "\n"
    )

    print("4e-3: RNN/CNN end-to-end + decode fixtures (real audio) ...")
    e2e_fixtures = generate_notes_end_to_end_fixtures()
    np.savez_compressed(FIXTURES_DIR / "notes_end_to_end.npz", **e2e_fixtures)

    print("4e-4: synthetic ADSR decode fixture ...")
    adsr_fixtures = generate_notes_adsr_synthetic_fixture()
    np.savez_compressed(FIXTURES_DIR / "notes_adsr_synthetic.npz",
                        **adsr_fixtures)

    print("4e-5: synthetic peak-picking fixture ...")
    peak_fixtures = generate_notes_peak_picking_synthetic_fixture()
    np.savez_compressed(FIXTURES_DIR / "notes_peak_picking_synthetic.npz",
                        **peak_fixtures)

    print("Done.")


if __name__ == "__main__":
    main()
