"""Phase-2 golden-fixture generator: RNNDownBeatProcessor activations, the
chained end-to-end beat/downbeat decode, and the unpickled-model structural
digest, all recorded from REAL (compiled) madmom -- the Phase-1 sibling of
`tools/generate_fixtures.py`, split into its own file because it needs a
different (heavier, network-touching) dependency: madmom's own pretrained
`DOWNBEATS_BLSTM` model weights (CC BY-NC-SA 4.0, see madmom_infer/models.py
and this repo's README/NOTICE -- NEVER bundled, only read here transiently
from an already-installed real-madmom environment to produce comparison
fixtures, same as `tools/generate_fixtures.py`'s doc block explains for
Phase 1).

HOW TO RUN -- same real-madmom reference venv as Phase 1
(`madmom-reference/.venv`, Python 3.10.18, numpy 1.23.5, scipy 1.15.3 --
rebuilt 2026-07-12 to the same recorded versions as the original, now-gone
`all-in-one-fix/.venv`), which already has the `DOWNBEATS_BLSTM` weights
installed as vendored package data (a real, pip-installed madmom 0.17.dev0
wheel bundles its own `madmom/models` tree):

    /home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python \\
        tools/generate_phase2_fixtures.py

Reuses the SAME test wavs `tools/generate_fixtures.py` already generated
under `tests/fixtures/wavs/` (does not regenerate them -- run Phase-1's
script first if that directory is empty). Only the 44.1kHz-native wavs are
usable here: `RNNDownBeatProcessor` hard-codes `SignalProcessor(sample_rate=
44100)` and madmom_infer has no ffmpeg-backed resampling (see
`madmom_infer/audio/signal.py`'s module header) -- `stereo_48000.wav` is
skipped for that reason.

Reads: real `madmom` (features.downbeats.RNNDownBeatProcessor,
features.downbeats.DBNDownBeatTrackingProcessor, models.DOWNBEATS_BLSTM,
ml.nn.NeuralNetwork), numpy. Writes: tests/fixtures/rnn_downbeat.npz,
tests/fixtures/nn_structural_digest.json. Read by:
tests/test_ml_nn.py, tests/test_downbeats_rnn.py.
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

DBN_PARAMS = dict(beats_per_bar=[3, 4], fps=100)

# 44.1kHz-native cases only -- see module header (no resampling support).
RNN_CASES = {
    "mono_44100": "mono_44100.wav",
    "stereo_44100": "stereo_44100.wav",
    "float32_44100": "float32_44100.wav",
}


def _arr_digest(arr) -> dict:
    arr = np.ascontiguousarray(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def digest_layer(layer) -> dict:
    """Structural digest of one NN layer: type, weight/bias shapes+sha256,
    activation function name, recursing into known sub-layer attributes only
    (gated by type) -- see this file's module header re: legacy leftover
    attributes some pickled `Gate` instances carry that must NOT be recursed
    into naively."""
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


def generate_nn_digest() -> dict:
    import madmom
    from madmom.models import DOWNBEATS_BLSTM

    digests = {}
    for i, f in enumerate(DOWNBEATS_BLSTM, start=1):
        nn = madmom.ml.nn.NeuralNetwork.load(f)
        digests[f"downbeats_blstm_{i}"] = [digest_layer(l) for l in nn.layers]
    return digests


def generate_rnn_downbeat_fixtures() -> dict:
    from madmom.features.downbeats import (
        DBNDownBeatTrackingProcessor, RNNDownBeatProcessor,
    )

    rnn = RNNDownBeatProcessor()
    dbn = DBNDownBeatTrackingProcessor(**DBN_PARAMS)

    out = {}
    for case, wav_name in RNN_CASES.items():
        wav_path = WAVS_DIR / wav_name
        act = rnn(str(wav_path))
        beats = np.asarray(dbn(act))
        out[f"{case}_activations"] = act
        out[f"{case}_beat_times"] = beats
    return out


def main() -> None:
    try:
        import madmom  # noqa: F401
    except ImportError as exc:
        print(
            "ERROR: this script needs the real `madmom` package (not "
            "madmom_infer), including its DOWNBEATS_BLSTM pretrained model "
            "weights. Run it with the madmom-reference venv's interpreter -- "
            "see this file's module docstring for the exact command.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    if not WAVS_DIR.exists() or not any(WAVS_DIR.glob("*.wav")):
        print(
            "ERROR: tests/fixtures/wavs/ is empty -- run "
            "tools/generate_fixtures.py first (Phase-1 script, generates "
            "the shared test wavs this script reuses).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"Using madmom {madmom.__version__} from {madmom.__file__}")

    print("Phase 2a: unpickled-model structural digest (DOWNBEATS_BLSTM) ...")
    digest = generate_nn_digest()
    (FIXTURES_DIR / "nn_structural_digest.json").write_text(
        json.dumps(digest, indent=2, sort_keys=True) + "\n"
    )

    print("Phase 2b: RNNDownBeatProcessor + DBNDownBeatTrackingProcessor "
          "fixtures ...")
    rnn_fixtures = generate_rnn_downbeat_fixtures()
    np.savez_compressed(FIXTURES_DIR / "rnn_downbeat.npz", **rnn_fixtures)

    print("Done.")


if __name__ == "__main__":
    main()
