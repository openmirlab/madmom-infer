"""Wave-4a golden-fixture generator: `key_cnn.pkl`'s unpickled structural
digest, per-layer-type input/output recordings (one representative instance
per new layer class this wave ports: `PadLayer`, `ConvolutionalLayer`,
`BatchNormLayer`, `MaxPoolLayer`, `AverageLayer`), and the end-to-end
`CNNKeyRecognitionProcessor` activation + decoded key label, all recorded
from REAL (compiled) madmom -- the 4a sibling of
`tools/generate_phase2_fixtures.py`, kept in its own file (not folded into
that script) to keep the diff for this wave minimal and independently
regenerable without touching Phase-2's already-committed fixtures.

**Layer fixtures are fully self-contained (weights included), by design.**
`key_layers.npz` records not just each sampled layer's input/output but also
its own real trained parameters (`ConvolutionalLayer`'s `weights`/`bias`,
`BatchNormLayer`'s `beta`/`gamma`/`mean`/`inv_std`) and
`key_layer_params.json` records its non-array hyperparameters (`stride`,
`pad`, `size`, `axis`, `width`, `axes`, `value`, `dtype`, `keepdims`,
`activation_fn` name). This lets `tests/test_key.py`'s per-layer-type tests
construct THIS PORT's own layer classes directly and compare against the
golden output -- WITHOUT unpickling `key_cnn.pkl` at all, so those tests run
fully offline (no network, no local model-file dependency), unlike the
structural-digest/end-to-end tests below which need the real `.pkl` (via
`madmom_infer.models.key_cnn()`, `@pytest.mark.network`).

HOW TO RUN -- same real-madmom reference venv as Phase 1/2
(`madmom-reference/.venv`, Python 3.10.18, numpy 1.23.5, scipy 1.15.3), whose
already-installed madmom 0.17.dev0 wheel vendors `key/2018/key_cnn.pkl` as
package data (no network needed to run this script):

    /home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python \\
        tools/generate_key_fixtures.py

Reuses the SAME 44.1kHz-native test wavs `tools/generate_phase2_fixtures.py`
already established as the "no resampling support" subset
(`mono_44100`, `stereo_44100`, `float32_44100`) -- `CNNKeyRecognitionProcessor`
hard-codes `SignalProcessor(sample_rate=44100)` exactly like
`RNNDownBeatProcessor`, and madmom_infer has no ffmpeg-backed resampling (see
`madmom_infer/audio/signal.py`'s module header) -- `stereo_48000.wav` is
skipped for that reason.

Reads: real `madmom` (features.key.CNNKeyRecognitionProcessor,
features.key.key_prediction_to_label, models.KEY_CNN, ml.nn.NeuralNetwork),
numpy. Writes: tests/fixtures/key_structural_digest.json,
tests/fixtures/key_layers.npz, tests/fixtures/key_layer_params.json,
tests/fixtures/key_activations.npz. Read by: tests/test_key.py.
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
KEY_CASES = {
    "mono_44100": "mono_44100.wav",
    "stereo_44100": "stereo_44100.wav",
    "float32_44100": "float32_44100.wav",
}

# One representative layer index per new layer TYPE this wave ports, chosen
# from key_cnn.pkl's actual 30-layer stack (verified by direct inspection,
# see this repo's CLAUDE.md 4.0 audit): 0=PadLayer, 1=ConvolutionalLayer,
# 2=BatchNormLayer, 6=MaxPoolLayer, 29=AverageLayer (the network's final
# layer -- global-average-pooling head).
LAYER_FIXTURE_INDICES = {
    "PadLayer": 0,
    "ConvolutionalLayer": 1,
    "BatchNormLayer": 2,
    "MaxPoolLayer": 6,
    "AverageLayer": 29,
}


def _arr_digest(arr) -> dict:
    arr = np.ascontiguousarray(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def digest_layer(layer) -> dict:
    """Structural digest of one NN layer: type, weight/bias/beta/gamma/mean/
    inv_std array shapes+sha256, scalar hyperparameters (stride/pad/size/
    axis/width/axes/value/dtype/keepdims), and activation-function name --
    the 4a extension of `tools/generate_phase2_fixtures.py`'s `digest_layer`
    to cover the 5 new CNN-era layer classes `key_cnn.pkl` actually uses."""
    t = type(layer).__name__
    d = {"type": t}
    if hasattr(layer, "weights"):
        d["weights"] = _arr_digest(layer.weights)
    if hasattr(layer, "bias"):
        d["bias"] = _arr_digest(layer.bias)
    if getattr(layer, "activation_fn", None) is not None:
        d["activation_fn"] = layer.activation_fn.__name__
    if t == "ConvolutionalLayer":
        d["stride"] = layer.stride
        d["pad"] = layer.pad
    elif t == "BatchNormLayer":
        d["beta"] = _arr_digest(layer.beta)
        d["gamma"] = _arr_digest(layer.gamma)
        d["mean"] = _arr_digest(layer.mean)
        d["inv_std"] = _arr_digest(layer.inv_std)
    elif t == "MaxPoolLayer":
        d["size"] = list(layer.size)
        d["stride"] = list(layer.stride) if layer.stride is not None else None
        d["axis"] = layer.axis
    elif t == "PadLayer":
        d["width"] = layer.width
        d["axes"] = list(layer.axes)
        d["value"] = layer.value
    elif t == "AverageLayer":
        axis = layer.axis
        d["axis"] = list(axis) if isinstance(axis, tuple) else axis
        d["dtype"] = str(layer.dtype) if layer.dtype is not None else None
        d["keepdims"] = layer.keepdims
    return d


def generate_key_structural_digest() -> dict:
    import madmom
    from madmom.models import KEY_CNN

    assert len(KEY_CNN) == 1, (
        f"expected KEY_CNN to resolve to exactly 1 file, got {len(KEY_CNN)} "
        "-- see madmom_infer/models.py's key_cnn() header for why this "
        "port assumes a size-1 ensemble."
    )
    nn = madmom.ml.nn.NeuralNetwork.load(KEY_CNN[0])
    return {"key_cnn": [digest_layer(l) for l in nn.layers]}


def generate_key_layer_fixtures() -> "tuple[dict, dict]":
    """Run key_cnn.pkl's real forward pass on `mono_44100.wav`'s spectrogram
    feature, recording every layer's input AND output array in full (they
    are small -- largest is 13*192*4 bytes -- so no need to hash-only), then
    keep only the input/output pair at each `LAYER_FIXTURE_INDICES` entry:
    one full golden (input, output) pair per new layer TYPE this wave ports
    -- PLUS that same layer's own real trained parameters (array params into
    the returned npz-payload dict, non-array hyperparameters into a second,
    JSON-serializable dict), so `tests/test_key.py` can reconstruct this
    port's own layer instance and compare without ever unpickling
    `key_cnn.pkl` itself. Returns `(npz_payload, params_json)`."""
    import madmom
    from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
    from madmom.audio.spectrogram import (
        LogarithmicFilteredSpectrogramProcessor,
    )
    from madmom.audio.stft import ShortTimeFourierTransformProcessor
    from madmom.models import KEY_CNN

    sig = SignalProcessor(num_channels=1, sample_rate=44100)
    frames = FramedSignalProcessor(frame_size=8192, fps=5)
    stft = ShortTimeFourierTransformProcessor()
    spec = LogarithmicFilteredSpectrogramProcessor(
        num_bands=24, fmin=65, fmax=2100, unique_filters=True)

    wav_path = WAVS_DIR / KEY_CASES["mono_44100"]
    data = spec(stft(frames(sig(str(wav_path)))))
    data = np.array(data)

    nn = madmom.ml.nn.NeuralNetwork.load(KEY_CNN[0])
    assert len(nn.layers) == 30, (
        f"expected key_cnn.pkl to have 30 layers, got {len(nn.layers)} -- "
        "LAYER_FIXTURE_INDICES was chosen against a 30-layer stack, "
        "re-verify the indices if this assertion fires."
    )

    npz_payload = {"spec_input": data}
    params_json = {}
    for type_name, idx in LAYER_FIXTURE_INDICES.items():
        layer = nn.layers[idx]
        actual_type = type(layer).__name__
        assert actual_type == type_name, (
            f"LAYER_FIXTURE_INDICES[{type_name!r}] = {idx} but "
            f"nn.layers[{idx}] is actually a {actual_type} -- indices are "
            "stale, re-inspect key_cnn.pkl's layer stack."
        )
        # replay the forward pass up to (and including) this layer
        replay = np.array(data, copy=True)
        for prior in nn.layers[:idx]:
            replay = prior(replay)
        layer_out = layer(replay)
        npz_payload[f"{type_name}_input"] = np.asarray(replay)
        npz_payload[f"{type_name}_output"] = np.asarray(layer_out)

        activation_name = (layer.activation_fn.__name__
                            if getattr(layer, "activation_fn", None)
                            is not None else None)
        if type_name == "ConvolutionalLayer":
            npz_payload["ConvolutionalLayer_weights"] = np.asarray(layer.weights)
            npz_payload["ConvolutionalLayer_bias"] = np.asarray(layer.bias)
            params_json[type_name] = {
                "stride": layer.stride, "pad": layer.pad,
                "activation_fn": activation_name,
            }
        elif type_name == "BatchNormLayer":
            npz_payload["BatchNormLayer_beta"] = np.asarray(layer.beta)
            npz_payload["BatchNormLayer_gamma"] = np.asarray(layer.gamma)
            npz_payload["BatchNormLayer_mean"] = np.asarray(layer.mean)
            npz_payload["BatchNormLayer_inv_std"] = np.asarray(layer.inv_std)
            params_json[type_name] = {"activation_fn": activation_name}
        elif type_name == "MaxPoolLayer":
            params_json[type_name] = {
                "size": list(layer.size),
                "stride": list(layer.stride) if layer.stride is not None
                else None,
                "axis": layer.axis,
            }
        elif type_name == "PadLayer":
            params_json[type_name] = {
                "width": layer.width, "axes": list(layer.axes),
                "value": layer.value,
            }
        elif type_name == "AverageLayer":
            axis = layer.axis
            params_json[type_name] = {
                "axis": list(axis) if isinstance(axis, tuple) else axis,
                "dtype": str(layer.dtype) if layer.dtype is not None else None,
                "keepdims": layer.keepdims,
            }
    return npz_payload, params_json


def generate_key_end_to_end_fixtures() -> dict:
    from madmom.features.key import (
        CNNKeyRecognitionProcessor, key_prediction_to_label,
    )

    proc = CNNKeyRecognitionProcessor()
    out = {}
    for case, wav_name in KEY_CASES.items():
        wav_path = WAVS_DIR / wav_name
        prediction = proc(str(wav_path))
        out[f"{case}_prediction"] = prediction
        out[f"{case}_label"] = np.array(key_prediction_to_label(prediction))
    return out


def main() -> None:
    try:
        import madmom  # noqa: F401
    except ImportError as exc:
        print(
            "ERROR: this script needs the real `madmom` package (not "
            "madmom_infer), including its KEY_CNN pretrained model weights. "
            "Run it with the madmom-reference venv's interpreter -- see "
            "this file's module docstring for the exact command.",
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

    print("4a-1: key_cnn.pkl structural digest ...")
    digest = generate_key_structural_digest()
    (FIXTURES_DIR / "key_structural_digest.json").write_text(
        json.dumps(digest, indent=2, sort_keys=True) + "\n"
    )

    print("4a-2: per-layer-type golden input/output fixtures ...")
    layer_fixtures, layer_params = generate_key_layer_fixtures()
    np.savez_compressed(FIXTURES_DIR / "key_layers.npz", **layer_fixtures)
    (FIXTURES_DIR / "key_layer_params.json").write_text(
        json.dumps(layer_params, indent=2, sort_keys=True) + "\n"
    )

    print("4a-3: CNNKeyRecognitionProcessor end-to-end fixtures ...")
    e2e_fixtures = generate_key_end_to_end_fixtures()
    np.savez_compressed(FIXTURES_DIR / "key_activations.npz", **e2e_fixtures)

    print("Done.")


if __name__ == "__main__":
    main()
