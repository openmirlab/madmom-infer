"""Golden-fixture tests for Wave 4a: `madmom_infer.features.key`
(`CNNKeyRecognitionProcessor`) and the 5 new CNN-era layer classes
`madmom_infer.ml.nn.layers` gained to support it (`PadLayer`,
`ConvolutionalLayer`, `BatchNormLayer`, `MaxPoolLayer`, `AverageLayer`), all
recorded by `tools/generate_key_fixtures.py` from real (compiled) madmom.

Four independent things are verified here:

1. **Per-layer-type forward-pass correctness, fully OFFLINE** (no network,
   no local `key_cnn.pkl` dependency): `tests/fixtures/key_layers.npz` +
   `key_layer_params.json` carry each sampled layer's own real trained
   weights AND a real (input, output) pair recorded from that exact layer
   inside `key_cnn.pkl`'s real forward pass -- see
   `tools/generate_key_fixtures.py`'s module header for why this is
   self-contained. This is the strongest, most localized proof: one layer
   class, real weights, real input, real output, no pipeline noise.
2. **Unpickling correctness** (network): `madmom_infer.ml.nn.unpickle.
   load_model`'s restricted `SafeUnpickler` reconstructs the exact same
   layer types, shapes, hyperparameters, and weight/bias/beta/gamma/mean/
   inv_std array CONTENT (sha256) as real madmom's own unpickling of
   `key_cnn.pkl` -- same technique as `test_ml_nn.py`.
3. **End-to-end forward-pass correctness** (network): feeding the same
   44.1kHz test wavs through this port's `CNNKeyRecognitionProcessor`
   reproduces real madmom's 24-class key-probability vector to within a
   documented ULP bound, and the DECODED key label (`key_prediction_to_
   label`) EXACTLY.
4. **Cross-BLAS exactness** (the strongest claim, same shape as
   `test_downbeats_rnn.py::test_full_pipeline_is_exact_under_original_blas`):
   this port's own `CNNKeyRecognitionProcessor`, run under the ORIGINAL
   reference venv's numpy/scipy build (the same environment
   `tools/generate_key_fixtures.py` recorded `key_activations.npz` from),
   reproduces real madmom's activations AND decoded labels with ZERO
   differing elements -- proving the ULP drift measured by test 3 is BLAS
   non-associativity, not an algorithmic difference in this port's new CNN
   layer classes.

Downloads real weights via `madmom_infer.models.key_cnn()` for tests 2-4 --
network-touching and NON-COMMERCIAL-licensed (CC BY-NC-SA 4.0), see that
module's header. The download itself only happens inside the `model_paths`
fixture (never at import time), so collecting this file never touches the
network -- only tests actually marked `pytest.mark.network` (deselected by
default, see pyproject.toml's `addopts`) trigger it, and even then a
download failure is a clean `pytest.skip`, never a collection error.

Reads: madmom_infer/ml/nn/layers.py (the 5 new classes),
madmom_infer/features/key.py, madmom_infer/models.py,
tests/fixtures/key_layers.npz, tests/fixtures/key_layer_params.json,
tests/fixtures/key_structural_digest.json, tests/fixtures/key_activations.npz
"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from madmom_infer.ml.nn.activations import elu, linear
from madmom_infer.ml.nn.layers import (
    AverageLayer, BatchNormLayer, ConvolutionalLayer, MaxPoolLayer, PadLayer,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
WAVS_DIR = FIXTURES_DIR / "wavs"
REPO_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_PYTHON = Path(
    "/home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python"
)

# 44.1kHz-native cases only -- see module header (no resampling support).
KEY_CASES = ("mono_44100", "stereo_44100", "float32_44100")

# Measured worst case (see tests below) is 4 ULP end-to-end; this margin
# (4x) matches the repo's convention (test_beats_hmm.py's 4x, test_downbeats_
# rnn.py's ~2.7x) of "generous but not unlimited".
MAX_ULP = 16

_ACTIVATION_FNS = {"linear": linear, "elu": elu}


# ---------------------------------------------------------------------------
# 1. Per-layer-type forward-pass correctness (offline, no weights download)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def key_layers_fixture():
    return np.load(FIXTURES_DIR / "key_layers.npz")


@pytest.fixture(scope="module")
def key_layer_params():
    with open(FIXTURES_DIR / "key_layer_params.json") as fh:
        return json.load(fh)


def test_pad_layer_matches_fixture(key_layers_fixture, key_layer_params):
    params = key_layer_params["PadLayer"]
    layer = PadLayer(width=params["width"], axes=tuple(params["axes"]),
                      value=params["value"])
    out = layer.activate(key_layers_fixture["PadLayer_input"])
    expected = key_layers_fixture["PadLayer_output"]
    assert out.dtype == expected.dtype
    np.testing.assert_array_equal(out, expected)


def test_convolutional_layer_matches_fixture(key_layers_fixture, key_layer_params):
    params = key_layer_params["ConvolutionalLayer"]
    layer = ConvolutionalLayer(
        weights=key_layers_fixture["ConvolutionalLayer_weights"],
        bias=key_layers_fixture["ConvolutionalLayer_bias"],
        stride=params["stride"], pad=params["pad"],
        activation_fn=_ACTIVATION_FNS[params["activation_fn"]],
    )
    out = layer.activate(key_layers_fixture["ConvolutionalLayer_input"])
    expected = key_layers_fixture["ConvolutionalLayer_output"]
    assert out.shape == expected.shape
    assert out.dtype == expected.dtype
    np.testing.assert_array_max_ulp(out, expected, maxulp=MAX_ULP)


def test_batch_norm_layer_matches_fixture(key_layers_fixture, key_layer_params):
    params = key_layer_params["BatchNormLayer"]
    layer = BatchNormLayer(
        beta=key_layers_fixture["BatchNormLayer_beta"],
        gamma=key_layers_fixture["BatchNormLayer_gamma"],
        mean=key_layers_fixture["BatchNormLayer_mean"],
        inv_std=key_layers_fixture["BatchNormLayer_inv_std"],
        activation_fn=_ACTIVATION_FNS[params["activation_fn"]],
    )
    out = layer.activate(key_layers_fixture["BatchNormLayer_input"])
    expected = key_layers_fixture["BatchNormLayer_output"]
    assert out.shape == expected.shape
    np.testing.assert_array_max_ulp(out, expected, maxulp=MAX_ULP)


def test_max_pool_layer_matches_fixture(key_layers_fixture, key_layer_params):
    params = key_layer_params["MaxPoolLayer"]
    layer = MaxPoolLayer(size=tuple(params["size"]),
                          stride=tuple(params["stride"]) if params["stride"]
                          else None, axis=params["axis"])
    out = layer.activate(key_layers_fixture["MaxPoolLayer_input"])
    expected = key_layers_fixture["MaxPoolLayer_output"]
    assert out.dtype == expected.dtype
    np.testing.assert_array_equal(out, expected)


def test_average_layer_matches_fixture(key_layers_fixture, key_layer_params):
    params = key_layer_params["AverageLayer"]
    axis = tuple(params["axis"]) if isinstance(params["axis"], list) \
        else params["axis"]
    layer = AverageLayer(axis=axis, dtype=params["dtype"],
                          keepdims=params["keepdims"])
    out = layer.activate(key_layers_fixture["AverageLayer_input"])
    expected = key_layers_fixture["AverageLayer_output"]
    assert out.shape == expected.shape
    np.testing.assert_array_max_ulp(out, expected, maxulp=MAX_ULP)


# ---------------------------------------------------------------------------
# 2. Unpickling correctness (network -- needs the real key_cnn.pkl bytes)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model_paths():
    """Downloads (or reuses the local cache for) `key_cnn.pkl`. Deliberately
    NOT module-level eager code -- see test_ml_nn.py's identical fixture for
    why (pytest imports every test module before applying `-m` filters)."""
    from madmom_infer.models import key_cnn

    try:
        return key_cnn()
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"could not download KEY_CNN weights: {exc}")


def _arr_digest(arr):
    arr = np.ascontiguousarray(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def digest_layer(layer):
    """Independent reimplementation of
    tools/generate_key_fixtures.py's digest_layer -- deliberately not
    imported from tools/, same discipline as test_ml_nn.py's own copy."""
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


@pytest.fixture(scope="module")
def structural_digest_fixture():
    with open(FIXTURES_DIR / "key_structural_digest.json") as fh:
        return json.load(fh)


@pytest.mark.network
def test_unpickled_key_cnn_structurally_matches_real_madmom(
    structural_digest_fixture, model_paths
):
    """Every layer type, weight/bias/beta/gamma/mean/inv_std shape+dtype+
    sha256, scalar hyperparameter, and activation-function name must match
    real madmom's own unpickling, exactly (no tolerance -- discrete
    metadata, not floats)."""
    from madmom_infer.ml.nn.unpickle import load_model

    assert len(model_paths) == 1
    nn = load_model(model_paths[0])
    ours = [digest_layer(l) for l in nn.layers]
    expected = structural_digest_fixture["key_cnn"]
    assert ours == expected


# ---------------------------------------------------------------------------
# 3 + 4. End-to-end forward-pass correctness + cross-BLAS exactness (network)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _key_cnn_ready():
    from madmom_infer.models import key_cnn

    try:
        key_cnn()
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"could not download KEY_CNN weights: {exc}")


@pytest.fixture(scope="module")
def key_activations_fixture():
    return np.load(FIXTURES_DIR / "key_activations.npz")


@pytest.mark.network
def test_key_activations_match_fixture_within_ulp(
    key_activations_fixture, _key_cnn_ready
):
    from madmom_infer.features.key import CNNKeyRecognitionProcessor

    proc = CNNKeyRecognitionProcessor()
    for case in KEY_CASES:
        prediction = proc(str(WAVS_DIR / f"{case}.wav"))
        expected = key_activations_fixture[f"{case}_prediction"]
        assert prediction.shape == expected.shape, case
        assert prediction.dtype == expected.dtype, case
        np.testing.assert_array_max_ulp(prediction, expected, maxulp=MAX_ULP)


@pytest.mark.network
def test_key_label_is_exact(key_activations_fixture, _key_cnn_ready):
    """Despite activation-level ULP drift (previous test), the DECODED key
    label (an argmax-over-24-classes operation) must be EXACT -- absorbs
    float32-ULP-scale input noise, same shape of claim as
    test_downbeats_rnn.py's decoded-beat-times exactness."""
    from madmom_infer.features.key import (
        CNNKeyRecognitionProcessor, key_prediction_to_label,
    )

    proc = CNNKeyRecognitionProcessor()
    for case in KEY_CASES:
        prediction = proc(str(WAVS_DIR / f"{case}.wav"))
        label = key_prediction_to_label(prediction)
        expected_label = str(key_activations_fixture[f"{case}_label"])
        assert label == expected_label, case


def _reference_python_available():
    return REFERENCE_PYTHON.exists()


@pytest.mark.skipif(
    not _reference_python_available(),
    reason="reference madmom install (madmom-reference/.venv) not found on "
           "this machine; the cross-BLAS proof requires it",
)
def test_full_pipeline_is_exact_under_original_blas(key_activations_fixture):
    """THE proof: this project's own `CNNKeyRecognitionProcessor`, run under
    the ORIGINAL reference venv's numpy/scipy build (the same environment
    real madmom's own recorded fixture came from), reproduces both the
    activations AND the decoded key labels with ZERO differing elements,
    for all 3 cases -- proving the ULP drift measured above is caused
    entirely by BLAS-library non-associativity, not by any algorithmic
    difference in this port's new CNN layer classes. Uses the local
    `key_cnn.pkl` copy under `../madmom-upstream` directly (`nn_files=`
    override) so this test needs neither network nor a prior `-m network`
    run to have already populated the download cache.
    """
    upstream_key_cnn = (
        REPO_ROOT.parent / "madmom-upstream" / "madmom" / "models" / "key"
        / "2018" / "key_cnn.pkl"
    )
    if not upstream_key_cnn.exists():
        pytest.skip(f"local key_cnn.pkl not found at {upstream_key_cnn}")

    case_paths = ", ".join(repr(str(WAVS_DIR / f"{c}.wav")) for c in KEY_CASES)
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import numpy as np
from madmom_infer.features.key import (
    CNNKeyRecognitionProcessor, key_prediction_to_label,
)

cases = {list(KEY_CASES)!r}
wav_paths = [{case_paths}]
proc = CNNKeyRecognitionProcessor(nn_files=[{str(upstream_key_cnn)!r}])
fixture = np.load({str(FIXTURES_DIR / "key_activations.npz")!r})

for case, wav_path in zip(cases, wav_paths):
    prediction = proc(wav_path)
    label = key_prediction_to_label(prediction)
    expected_prediction = fixture[case + "_prediction"]
    expected_label = str(fixture[case + "_label"])
    assert np.array_equal(prediction, expected_prediction), \\
        f"{{case}}: activations differ"
    assert label == expected_label, f"{{case}}: label differs"
print("EXACT_MATCH")
"""
    proc = subprocess.run(
        [str(REFERENCE_PYTHON), "-c", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "EXACT_MATCH" in proc.stdout
