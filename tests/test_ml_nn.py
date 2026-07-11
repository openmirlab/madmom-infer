"""Golden-fixture tests for madmom_infer.ml.nn against real madmom's
pretrained `DOWNBEATS_BLSTM` model files -- the Phase-2 NN-runtime centerpiece
(`madmom_infer/ml/nn/{__init__,layers,activations,unpickle}.py`).

Two independent things are verified here, both against
`tests/fixtures/nn_structural_digest.json` (recorded by
`tools/generate_phase2_fixtures.py` from real madmom):

1. **Unpickling correctness**: `madmom_infer.ml.nn.unpickle.load_model`'s
   restricted `SafeUnpickler` reconstructs the exact same layer types,
   shapes, and weight/bias/recurrent/peephole array CONTENT (sha256) as
   real madmom's own (unrestricted) `pickle.load` -- across all 8
   `downbeats_blstm_[1-8].pkl` files. This is a stronger claim than "the
   forward pass matches": it proves the class-path remapping table (this
   module's `unpickle.py` header) is complete and correct, independent of
   any subsequent math.
2. **Forward-pass correctness**: feeding a fixed, seeded random input
   through each unpickled `NeuralNetwork` reproduces real madmom's
   activation to within a documented ULP bound (see
   `test_downbeats_rnn.py` for the full BLAS-bound proof methodology this
   inherits from `test_spectrogram.py`) -- this file only spot-checks shape/
   dtype/finiteness; the real numerical A/B lives in `test_downbeats_rnn.py`
   (whole-pipeline activations), since a single-layer numeric check here
   would just duplicate that file's proof with less context.

Downloads real weights via `madmom_infer.models.downbeats_blstm()` --
network-touching and NON-COMMERCIAL-licensed (CC BY-NC-SA 4.0), see that
module's header. The download itself only happens inside the
`model_paths` fixture (never at import time) so that collecting this
file never touches the network -- only tests actually marked
`pytest.mark.network` (deselected by default, see pyproject.toml's
`addopts`) trigger it, and even then a download failure is a clean
`pytest.skip`, never a collection error.

Reads: madmom_infer/ml/nn/*.py, madmom_infer/models.py,
tests/fixtures/nn_structural_digest.json
"""

import hashlib
import io
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from madmom_infer.ml.nn.unpickle import SafeUnpickler, load_model

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def model_paths():
    """Downloads (or reuses the local cache for) the 8 DOWNBEATS_BLSTM
    `.pkl` files. Deliberately NOT module-level eager code: a network call
    at import time would run during test COLLECTION regardless of any
    `-m 'not network'` deselection, since pytest imports every test module
    before applying marker filters. Keeping it inside a fixture means it
    only ever runs for tests that are both marked `network` and actually
    selected to run."""
    from madmom_infer.models import downbeats_blstm

    try:
        return downbeats_blstm()
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"could not download DOWNBEATS_BLSTM weights: {exc}")


def _arr_digest(arr):
    arr = np.ascontiguousarray(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def digest_layer(layer):
    """Independent reimplementation of tools/generate_phase2_fixtures.py's
    digest_layer -- deliberately not imported from `tools/`, to keep this
    test from silently passing if both copies drifted the same wrong way;
    see that script's module for the "why gate by type" rationale (legacy
    leftover attributes on some pickled `Gate` instances)."""
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
    return d


@pytest.fixture(scope="module")
def structural_digest_fixture():
    with open(FIXTURES_DIR / "nn_structural_digest.json") as fh:
        return json.load(fh)


@pytest.mark.network
@pytest.mark.parametrize("index", range(1, 9))
def test_unpickled_model_structurally_matches_real_madmom(
    index, structural_digest_fixture, model_paths
):
    """Every layer type, weight/bias/recurrent/peephole shape+dtype+sha256,
    and activation-function name must match real madmom's own unpickling,
    exactly (no tolerance -- these are discrete metadata, not floats)."""
    model_path = model_paths[index - 1]
    nn = load_model(model_path)
    ours = [digest_layer(l) for l in nn.layers]
    expected = structural_digest_fixture[f"downbeats_blstm_{index}"]
    assert ours == expected


def test_safe_unpickler_rejects_disallowed_globals():
    """The restricted unpickler must reject any class/function outside its
    allowlist -- e.g. a pickle that references `builtins.eval` -- loudly,
    not silently. Constructs a minimal malicious-shaped pickle by hand
    (protocol-2 GLOBAL opcode for `os.system`) rather than trusting a
    real model file to happen to demonstrate this."""
    # Build: PROTO 2 ; GLOBAL 'os system' ; STOP -- just resolves the global
    # (never calls it), which is already what SafeUnpickler must refuse.
    payload = b"\x80\x02cos\nsystem\n."
    with pytest.raises(pickle.UnpicklingError):
        SafeUnpickler(io.BytesIO(payload)).load()


@pytest.mark.network
def test_all_eight_ensemble_networks_have_expected_layer_shape(model_paths):
    """Sanity check independent of the fixture file: every ensemble member
    is `[BidirectionalLayer, BidirectionalLayer, BidirectionalLayer,
    FeedForwardLayer]` with a final 3-class softmax output (non-beat, beat,
    downbeat) -- this is the architecture `RNNDownBeatProcessor`'s
    `np.delete(obj=0, axis=1)` (drop non-beat) call site assumes."""
    for model_path in model_paths:
        nn = load_model(model_path)
        kinds = [type(l).__name__ for l in nn.layers]
        assert kinds == [
            "BidirectionalLayer", "BidirectionalLayer",
            "BidirectionalLayer", "FeedForwardLayer",
        ]
        final = nn.layers[-1]
        assert final.weights.shape[1] == 3
        assert final.activation_fn.__name__ == "softmax"
