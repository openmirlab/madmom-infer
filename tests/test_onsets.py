"""Golden-fixture tests for Wave 4b: `madmom_infer.features.onsets` -- the
pure-DSP spectral-flux onset detection function family,
`SpectralOnsetProcessor`, `RNNOnsetProcessor`, `CNNOnsetProcessor`,
`peak_picking`/`OnsetPeakPickingProcessor`, and the one new NN layer class
(`StrideLayer`) `onsets_cnn.pkl` needs -- all recorded by
`tools/generate_onset_fixtures.py` from real (compiled) madmom.

Several independent things are verified here:

1. **Per-DSP-function correctness, fully OFFLINE** (no network, no `.pkl`
   dependency): each pure onset-detection function is deterministic given
   the shared test wav + a fixed pre-processing pipeline, and this
   project's own Phase-1 DSP chain is already golden-fixture-proven exact
   -- so these tests rebuild the input via THIS PORT'S OWN pipeline and
   compare only the new function's OUTPUT against real madmom's recorded
   value (see `tools/generate_onset_fixtures.py`'s module header).
2. **`StrideLayer` correctness, fully OFFLINE**: same self-contained
   (input, output, params) design as `test_key.py`'s per-layer-type tests.
3. **`correlation_diff` crashes under Python 3 in real madmom too** (see
   `madmom_infer/features/onsets.py`'s module header) -- pinned as an
   expected `TypeError`, not a golden output (there isn't one to record).
4. **Unpickling correctness** (network): `onsets_rnn_1.pkl`/
   `onsets_brnn_1.pkl`/`onsets_cnn.pkl` structural digests match real
   madmom's own unpickling exactly.
5. **RNN/CNN end-to-end activation correctness + decoded-onset-time
   exactness** (network): `RNNOnsetProcessor`/`CNNOnsetProcessor`
   activations match within a documented ULP bound; `OnsetPeakPickingProcessor`
   decoded onset TIMES are bit-exact (discrete argmax-style decode absorbs
   float32-ULP-scale input noise, same shape of claim as
   `test_downbeats_rnn.py`'s decoded beat times).
6. **Cross-BLAS exactness** (the strongest claim): this port's own
   `RNNOnsetProcessor`/`CNNOnsetProcessor` + `OnsetPeakPickingProcessor`,
   run under the ORIGINAL reference venv's numpy/scipy build, reproduce
   real madmom's activations AND decoded onset times with ZERO differing
   elements.

**Same "shared-instance-in-order" caching-gotcha discipline as
`test_downbeats_rnn.py`** (see that file's module header): `RNNOnsetProcessor`/
`CNNOnsetProcessor` each build ONE `ShortTimeFourierTransformProcessor`/
`FilteredSpectrogramProcessor` PER FRAME-SIZE BRANCH inside `__init__` and
reuse them across calls -- `tools/generate_onset_fixtures.py` processes all 3
cases through ONE shared processor instance, in order (`mono_44100` ->
`stereo_44100` -> `float32_44100`); every network test below replicates that
exact call order/instance-reuse, or it would silently compare against the
wrong numbers.

Reads: madmom_infer/features/onsets.py, madmom_infer/ml/nn/layers.py
(StrideLayer), madmom_infer/models.py, tests/fixtures/onset_dsp_functions.npz,
tests/fixtures/onset_stride_layer.npz,
tests/fixtures/onset_stride_layer_params.json,
tests/fixtures/onset_structural_digest.json, tests/fixtures/onset_activations.npz
"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from madmom_infer.audio.filters import LogarithmicFilterbank
from madmom_infer.audio.signal import FramedSignalProcessor, SignalProcessor
from madmom_infer.audio.spectrogram import (
    FilteredSpectrogramProcessor, SpectrogramProcessor,
)
from madmom_infer.audio.stft import ShortTimeFourierTransformProcessor
from madmom_infer.features.onsets import (
    OnsetPeakPickingProcessor, SpectralOnsetProcessor, complex_domain,
    complex_flux, correlation_diff, high_frequency_content, peak_picking,
    modified_kullback_leibler, normalized_weighted_phase_deviation,
    phase_deviation, rectified_complex_domain, spectral_diff, spectral_flux,
    superflux, weighted_phase_deviation, wrap_to_pi,
)
from madmom_infer.ml.nn.layers import StrideLayer

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
WAVS_DIR = FIXTURES_DIR / "wavs"
REPO_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_PYTHON = Path(
    "/home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python"
)

# 44.1kHz-native cases only -- see module header (no resampling support).
ONSET_CASES = ("mono_44100", "stereo_44100", "float32_44100")

UPSTREAM_ONSETS_DIR = REPO_ROOT.parent / "madmom-upstream" / "madmom" / "models" / "onsets" / "2013"

# Measured worst case for the pure-DSP functions is 17 ULP (complex_flux,
# filtered branch); 64 is a generous (~4x) but not unlimited margin, matching
# this repo's own convention (test_key.py's 4x, test_beats_hmm.py's 4x).
MAX_ULP_DSP = 64
# Measured worst case for the RNN/BRNN/CNN activations is 62 ULP; 256 is
# ~4x that, same order of magnitude as test_downbeats_rnn.py's 512 for its
# own (bigger, 8-network BLSTM) ensemble.
MAX_ULP_NN = 256


# ---------------------------------------------------------------------------
# 1. Per-DSP-function correctness (offline, own pipeline reconstructs input)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dsp_fixture():
    return np.load(FIXTURES_DIR / "onset_dsp_functions.npz")


@pytest.fixture(scope="module")
def dsp_spectrograms():
    """Rebuild the exact same (spec, spec_cs, spec_filt_cs) inputs
    `tools/generate_onset_fixtures.py` used, via THIS PORT'S OWN
    already-golden-fixture-proven DSP chain."""
    wav = str(WAVS_DIR / "mono_44100.wav")
    sig = SignalProcessor(num_channels=1, sample_rate=44100)
    frames = FramedSignalProcessor(frame_size=2048, fps=200)
    stft_ncs = ShortTimeFourierTransformProcessor()
    stft_cs = ShortTimeFourierTransformProcessor(circular_shift=True)
    spec_proc = SpectrogramProcessor()
    filt_proc = FilteredSpectrogramProcessor(num_bands=24, norm_filters=False)

    spec = spec_proc(stft_ncs(frames(sig(wav))))
    spec_cs = spec_proc(stft_cs(frames(sig(wav))))
    spec_filt_cs = filt_proc(spec_cs)
    return spec, spec_cs, spec_filt_cs


def test_wrap_to_pi_matches_fixture(dsp_fixture):
    out = wrap_to_pi(dsp_fixture["wrap_to_pi_input"])
    np.testing.assert_array_max_ulp(
        np.asarray(out), dsp_fixture["wrap_to_pi_output"], maxulp=4)


def test_high_frequency_content_matches_fixture(dsp_fixture, dsp_spectrograms):
    spec, _, _ = dsp_spectrograms
    out = high_frequency_content(spec)
    np.testing.assert_array_max_ulp(
        np.asarray(out), dsp_fixture["high_frequency_content"],
        maxulp=MAX_ULP_DSP)


def test_spectral_diff_matches_fixture(dsp_fixture, dsp_spectrograms):
    spec, _, _ = dsp_spectrograms
    out = spectral_diff(spec)
    np.testing.assert_array_max_ulp(
        np.asarray(out), dsp_fixture["spectral_diff"], maxulp=MAX_ULP_DSP)


def test_spectral_flux_matches_fixture(dsp_fixture, dsp_spectrograms):
    spec, _, _ = dsp_spectrograms
    out = spectral_flux(spec)
    np.testing.assert_array_max_ulp(
        np.asarray(out), dsp_fixture["spectral_flux"], maxulp=MAX_ULP_DSP)


def test_superflux_matches_fixture(dsp_fixture, dsp_spectrograms):
    spec, _, _ = dsp_spectrograms
    out = superflux(spec)
    np.testing.assert_array_max_ulp(
        np.asarray(out), dsp_fixture["superflux"], maxulp=MAX_ULP_DSP)


def test_modified_kullback_leibler_matches_fixture(dsp_fixture, dsp_spectrograms):
    spec, _, _ = dsp_spectrograms
    out = modified_kullback_leibler(spec)
    np.testing.assert_array_max_ulp(
        np.asarray(out), dsp_fixture["modified_kullback_leibler"],
        maxulp=MAX_ULP_DSP)


def test_phase_deviation_matches_fixture(dsp_fixture, dsp_spectrograms):
    _, spec_cs, _ = dsp_spectrograms
    out = phase_deviation(spec_cs)
    np.testing.assert_array_max_ulp(
        np.asarray(out), dsp_fixture["phase_deviation"], maxulp=MAX_ULP_DSP)


def test_weighted_phase_deviation_matches_fixture(dsp_fixture, dsp_spectrograms):
    _, spec_cs, _ = dsp_spectrograms
    out = weighted_phase_deviation(spec_cs)
    np.testing.assert_array_max_ulp(
        np.asarray(out), dsp_fixture["weighted_phase_deviation"],
        maxulp=MAX_ULP_DSP)


def test_normalized_weighted_phase_deviation_matches_fixture(
    dsp_fixture, dsp_spectrograms
):
    _, spec_cs, _ = dsp_spectrograms
    out = normalized_weighted_phase_deviation(spec_cs)
    np.testing.assert_array_max_ulp(
        np.asarray(out), dsp_fixture["normalized_weighted_phase_deviation"],
        maxulp=MAX_ULP_DSP)


def test_complex_domain_matches_fixture(dsp_fixture, dsp_spectrograms):
    _, spec_cs, _ = dsp_spectrograms
    out = complex_domain(spec_cs)
    np.testing.assert_array_max_ulp(
        np.asarray(out), dsp_fixture["complex_domain"], maxulp=MAX_ULP_DSP)


def test_rectified_complex_domain_matches_fixture(dsp_fixture, dsp_spectrograms):
    _, spec_cs, _ = dsp_spectrograms
    out = rectified_complex_domain(spec_cs)
    np.testing.assert_array_max_ulp(
        np.asarray(out), dsp_fixture["rectified_complex_domain"],
        maxulp=MAX_ULP_DSP)


def test_complex_flux_unfiltered_matches_fixture(dsp_fixture, dsp_spectrograms):
    _, spec_cs, _ = dsp_spectrograms
    out = complex_flux(spec_cs)
    np.testing.assert_array_max_ulp(
        np.asarray(out), dsp_fixture["complex_flux_unfiltered"],
        maxulp=MAX_ULP_DSP)


def test_complex_flux_filtered_matches_fixture(dsp_fixture, dsp_spectrograms):
    _, _, spec_filt_cs = dsp_spectrograms
    out = complex_flux(spec_filt_cs)
    np.testing.assert_array_max_ulp(
        np.asarray(out), dsp_fixture["complex_flux_filtered"],
        maxulp=MAX_ULP_DSP)


def test_spectral_onset_processor_default_matches_fixture(dsp_fixture):
    sodf = SpectralOnsetProcessor()
    out = sodf(str(WAVS_DIR / "mono_44100.wav"))
    np.testing.assert_array_max_ulp(
        np.asarray(out), dsp_fixture["sodf_default"], maxulp=MAX_ULP_DSP)


def test_spectral_onset_processor_superflux_matches_fixture(dsp_fixture):
    sodf = SpectralOnsetProcessor(
        onset_method="superflux", fps=200, filterbank=LogarithmicFilterbank,
        num_bands=24, log=np.log10)
    out = sodf(str(WAVS_DIR / "mono_44100.wav"))
    np.testing.assert_array_max_ulp(
        np.asarray(out), dsp_fixture["sodf_superflux"], maxulp=MAX_ULP_DSP)


def test_correlation_diff_raises_typeerror():
    """Real madmom's own `correlation_diff` crashes under Python 3 -- see
    this module's + `madmom_infer/features/onsets.py`'s module headers."""
    spec = np.random.RandomState(0).rand(10, 20).astype(np.float32)
    with pytest.raises(TypeError):
        correlation_diff(spec)


def test_peak_picking_finds_expected_peaks():
    """Small, hand-constructed, deterministic sanity check of the
    `peak_picking` algorithm itself (not fixture-based -- the algorithm's
    correctness doesn't depend on real madmom's trained weights, only on
    matching upstream's own thresholding/moving-max logic)."""
    activations = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.8, 0.0])
    peaks = peak_picking(activations, threshold=0.5)
    np.testing.assert_array_equal(peaks, np.array([2, 6]))


def test_onset_peak_picking_processor_offline():
    activations = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.8, 0.0])
    proc = OnsetPeakPickingProcessor(fps=100, combine=0.0)
    onsets = proc(activations)
    np.testing.assert_array_equal(onsets, np.array([0.02, 0.06]))


# ---------------------------------------------------------------------------
# 2. StrideLayer correctness (offline, self-contained fixture)
# ---------------------------------------------------------------------------
def test_stride_layer_matches_fixture():
    layer_fixtures = np.load(FIXTURES_DIR / "onset_stride_layer.npz")
    with open(FIXTURES_DIR / "onset_stride_layer_params.json") as fh:
        params = json.load(fh)["StrideLayer"]
    layer = StrideLayer(block_size=params["block_size"])
    out = layer.activate(layer_fixtures["StrideLayer_input"])
    expected = layer_fixtures["StrideLayer_output"]
    assert out.shape == expected.shape
    np.testing.assert_array_equal(out, expected)


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
    tools/generate_onset_fixtures.py's digest_layer -- deliberately not
    imported from tools/, same discipline as test_key.py's own copy."""
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
        d["beta"] = _arr_digest(np.asarray(layer.beta))
        d["gamma"] = _arr_digest(np.asarray(layer.gamma))
        d["mean"] = _arr_digest(layer.mean)
        d["inv_std"] = _arr_digest(layer.inv_std)
    elif t == "MaxPoolLayer":
        d["size"] = np.asarray(layer.size).tolist()
        d["stride"] = (np.asarray(layer.stride).tolist()
                        if layer.stride is not None else None)
        d["axis"] = layer.axis
    elif t == "StrideLayer":
        d["block_size"] = int(layer.block_size)
    return d


@pytest.fixture(scope="module")
def structural_digest_fixture():
    with open(FIXTURES_DIR / "onset_structural_digest.json") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def _onsets_models_ready():
    """Downloads (or reuses the local cache for) all 3 onset model
    families. Deliberately NOT module-level eager code -- see
    test_ml_nn.py's identical fixture for why."""
    from madmom_infer.models import onsets_brnn, onsets_cnn, onsets_rnn

    try:
        return {
            "rnn": onsets_rnn(), "brnn": onsets_brnn(), "cnn": onsets_cnn(),
        }
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"could not download onset model weights: {exc}")


@pytest.mark.network
def test_unpickled_onsets_rnn_structurally_matches_real_madmom(
    structural_digest_fixture, _onsets_models_ready
):
    from madmom_infer.ml.nn.unpickle import load_model

    nn = load_model(_onsets_models_ready["rnn"][0])
    ours = [digest_layer(l) for l in nn.layers]
    assert ours == structural_digest_fixture["onsets_rnn_1"]


@pytest.mark.network
def test_unpickled_onsets_brnn_structurally_matches_real_madmom(
    structural_digest_fixture, _onsets_models_ready
):
    from madmom_infer.ml.nn.unpickle import load_model

    nn = load_model(_onsets_models_ready["brnn"][0])
    ours = [digest_layer(l) for l in nn.layers]
    assert ours == structural_digest_fixture["onsets_brnn_1"]


@pytest.mark.network
def test_unpickled_onsets_cnn_structurally_matches_real_madmom(
    structural_digest_fixture, _onsets_models_ready
):
    from madmom_infer.ml.nn.unpickle import load_model

    nn = load_model(_onsets_models_ready["cnn"][0])
    ours = [digest_layer(l) for l in nn.layers]
    assert ours == structural_digest_fixture["onsets_cnn"]


# ---------------------------------------------------------------------------
# 4 + 5. End-to-end activation correctness + decoded-onset-time exactness
# (network) -- shared-instance-in-order, see module header
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def onset_activations_fixture():
    return np.load(FIXTURES_DIR / "onset_activations.npz")


@pytest.mark.network
def test_rnn_onset_activations_match_fixture_within_ulp(
    onset_activations_fixture, _onsets_models_ready
):
    from madmom_infer.features.onsets import RNNOnsetProcessor

    brnn = RNNOnsetProcessor(online=False)
    rnn = RNNOnsetProcessor(online=True)
    for case in ONSET_CASES:
        wav_path = str(WAVS_DIR / f"{case}.wav")
        act_brnn = brnn(wav_path)
        act_rnn = rnn(wav_path)
        expected_brnn = onset_activations_fixture[f"{case}_brnn_activations"]
        expected_rnn = onset_activations_fixture[f"{case}_rnn_activations"]
        assert act_brnn.shape == expected_brnn.shape, case
        assert act_brnn.dtype == expected_brnn.dtype, case
        np.testing.assert_array_max_ulp(act_brnn, expected_brnn,
                                         maxulp=MAX_ULP_NN)
        assert act_rnn.shape == expected_rnn.shape, case
        np.testing.assert_array_max_ulp(act_rnn, expected_rnn,
                                         maxulp=MAX_ULP_NN)


@pytest.mark.network
def test_cnn_onset_activations_match_fixture_within_ulp(
    onset_activations_fixture, _onsets_models_ready
):
    from madmom_infer.features.onsets import CNNOnsetProcessor

    cnn = CNNOnsetProcessor()
    for case in ONSET_CASES:
        wav_path = str(WAVS_DIR / f"{case}.wav")
        act_cnn = cnn(wav_path)
        expected_cnn = onset_activations_fixture[f"{case}_cnn_activations"]
        assert act_cnn.shape == expected_cnn.shape, case
        assert act_cnn.dtype == expected_cnn.dtype, case
        np.testing.assert_array_max_ulp(act_cnn, expected_cnn,
                                         maxulp=MAX_ULP_NN)


@pytest.mark.network
def test_onset_peak_picking_times_are_exact(
    onset_activations_fixture, _onsets_models_ready
):
    """Despite activation-level ULP drift (previous 2 tests), the DECODED
    onset TIMES must be EXACT -- same shape of claim as
    test_downbeats_rnn.py's/test_key.py's decoded discrete outputs."""
    from madmom_infer.features.onsets import CNNOnsetProcessor, RNNOnsetProcessor

    brnn = RNNOnsetProcessor(online=False)
    rnn = RNNOnsetProcessor(online=True)
    cnn = CNNOnsetProcessor()
    pp = OnsetPeakPickingProcessor(fps=100)
    for case in ONSET_CASES:
        wav_path = str(WAVS_DIR / f"{case}.wav")
        onsets_brnn = np.asarray(pp(brnn(wav_path)))
        onsets_rnn = np.asarray(pp(rnn(wav_path)))
        onsets_cnn = np.asarray(pp(cnn(wav_path)))
        assert np.array_equal(
            onsets_brnn, onset_activations_fixture[f"{case}_brnn_onsets"]
        ), case
        assert np.array_equal(
            onsets_rnn, onset_activations_fixture[f"{case}_rnn_onsets"]
        ), case
        assert np.array_equal(
            onsets_cnn, onset_activations_fixture[f"{case}_cnn_onsets"]
        ), case


# ---------------------------------------------------------------------------
# 6. Cross-BLAS exactness (the strongest claim)
# ---------------------------------------------------------------------------
def _reference_python_available():
    return REFERENCE_PYTHON.exists()


def _upstream_onset_models_available():
    return (
        UPSTREAM_ONSETS_DIR.exists()
        and (UPSTREAM_ONSETS_DIR / "onsets_cnn.pkl").exists()
    )


@pytest.mark.skipif(
    not _reference_python_available(),
    reason="reference madmom install (madmom-reference/.venv) not found on "
           "this machine; the cross-BLAS proof requires it",
)
@pytest.mark.skipif(
    not _upstream_onset_models_available(),
    reason="local ../madmom-upstream/madmom/models/onsets checkout not "
           "found; the cross-BLAS proof needs it (no network required this "
           "way, see nn_files= override)",
)
def test_full_pipeline_is_exact_under_original_blas():
    """THE proof: this port's own `RNNOnsetProcessor`(online=False/True) +
    `CNNOnsetProcessor` + `OnsetPeakPickingProcessor`, run under the
    ORIGINAL reference venv's numpy/scipy build, reproduce real madmom's
    activations AND decoded onset times with ZERO differing elements, for
    all 3 cases and all 3 model families -- proving the ULP drift measured
    above is BLAS non-associativity, not an algorithmic difference. Uses
    the local `../madmom-upstream` `.pkl` copies directly (`nn_files=`
    override) so this test needs neither network nor a prior `-m network`
    run.
    """
    rnn_paths = [str(UPSTREAM_ONSETS_DIR / f"onsets_rnn_{i}.pkl")
                 for i in range(1, 9)]
    brnn_paths = [str(UPSTREAM_ONSETS_DIR / f"onsets_brnn_{i}.pkl")
                  for i in range(1, 9)]
    cnn_path = str(UPSTREAM_ONSETS_DIR / "onsets_cnn.pkl")

    case_paths = ", ".join(repr(str(WAVS_DIR / f"{c}.wav")) for c in ONSET_CASES)
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import numpy as np
from madmom_infer.features.onsets import (
    CNNOnsetProcessor, OnsetPeakPickingProcessor, RNNOnsetProcessor,
)

cases = {list(ONSET_CASES)!r}
wav_paths = [{case_paths}]
brnn = RNNOnsetProcessor(online=False, nn_files={brnn_paths!r})
rnn = RNNOnsetProcessor(online=True, nn_files={rnn_paths!r})
cnn = CNNOnsetProcessor(nn_files=[{cnn_path!r}])
pp = OnsetPeakPickingProcessor(fps=100)
fixture = np.load({str(FIXTURES_DIR / "onset_activations.npz")!r})

for case, wav_path in zip(cases, wav_paths):
    act_brnn = brnn(wav_path)
    act_rnn = rnn(wav_path)
    act_cnn = cnn(wav_path)
    onsets_brnn = np.asarray(pp(act_brnn))
    onsets_rnn = np.asarray(pp(act_rnn))
    onsets_cnn = np.asarray(pp(act_cnn))
    assert np.array_equal(act_brnn, fixture[case + "_brnn_activations"]), \\
        f"{{case}}: brnn activations differ"
    assert np.array_equal(act_rnn, fixture[case + "_rnn_activations"]), \\
        f"{{case}}: rnn activations differ"
    assert np.array_equal(act_cnn, fixture[case + "_cnn_activations"]), \\
        f"{{case}}: cnn activations differ"
    assert np.array_equal(onsets_brnn, fixture[case + "_brnn_onsets"]), \\
        f"{{case}}: brnn onsets differ"
    assert np.array_equal(onsets_rnn, fixture[case + "_rnn_onsets"]), \\
        f"{{case}}: rnn onsets differ"
    assert np.array_equal(onsets_cnn, fixture[case + "_cnn_onsets"]), \\
        f"{{case}}: cnn onsets differ"
print("EXACT_MATCH")
"""
    proc = subprocess.run(
        [str(REFERENCE_PYTHON), "-c", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "EXACT_MATCH" in proc.stdout
