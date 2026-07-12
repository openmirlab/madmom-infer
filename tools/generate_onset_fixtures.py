"""Wave-4b golden-fixture generator: per-DSP-onset-function golden outputs,
`onsets_rnn_1.pkl`/`onsets_brnn_1.pkl`/`onsets_cnn.pkl`'s unpickled
structural digests, `StrideLayer`'s self-contained golden (input, output,
params) fixture, and end-to-end `RNNOnsetProcessor`/`CNNOnsetProcessor`
activations + `OnsetPeakPickingProcessor` decoded onset times -- the 4b
sibling of `tools/generate_key_fixtures.py`, same conventions (own file, own
fixture files, independently regenerable without touching prior waves'
already-committed fixtures).

**DSP-function fixtures need no wav-independent input serialization.** Every
pure onset-detection function (`high_frequency_content`, `spectral_diff`,
... ) is deterministic given (a) the shared test wav and (b) a fixed
pre-processing pipeline -- and this project's own Phase-1 DSP chain
(`SignalProcessor` -> `FramedSignalProcessor` -> `ShortTimeFourierTransform
Processor` -> `SpectrogramProcessor`[-> `FilteredSpectrogramProcessor`]) is
already golden-fixture-proven bit/ULP-exact against real madmom. So this
generator only needs to record each function's OUTPUT array (real madmom,
real pipeline, same wav); `tests/test_onsets.py` reconstructs the identical
input via THIS PORT'S OWN already-proven pipeline and feeds it through this
port's own new onset function -- no intermediate spectrogram/phase object
needs to survive serialization, and these tests run fully OFFLINE (no
network, no `.pkl` unpickling at all).

HOW TO RUN -- same real-madmom reference venv as Phase 1/2/4a
(`madmom-reference/.venv`, Python 3.10.18, numpy 1.23.5, scipy 1.15.3):

    /home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python \\
        tools/generate_onset_fixtures.py

Reuses the same 44.1kHz-native test-wav subset established by
`tools/generate_phase2_fixtures.py`/`tools/generate_key_fixtures.py`
(`mono_44100`, `stereo_44100`, `float32_44100`) for the model-dependent
(RNN/BRNN/CNN activation + decoded onset time) fixtures --
`RNNOnsetProcessor`/`CNNOnsetProcessor` hard-code `SignalProcessor(sample_
rate=44100)` exactly like `RNNDownBeatProcessor`/`CNNKeyRecognitionProcessor`
(no ffmpeg-backed resampling in this port, see `audio/signal.py`'s module
header). The DSP-function fixtures (model-independent, pure math) only need
ONE representative wav (`mono_44100.wav`), matching the economy
`generate_key_fixtures.py`'s per-layer-type fixtures already established.

Reads: real `madmom` (audio.signal/stft/spectrogram/filters,
features.onsets, models.ONSETS_RNN/ONSETS_BRNN/ONSETS_CNN, ml.nn.
NeuralNetwork), numpy. Writes: tests/fixtures/onset_dsp_functions.npz,
tests/fixtures/onset_structural_digest.json,
tests/fixtures/onset_stride_layer.npz,
tests/fixtures/onset_stride_layer_params.json,
tests/fixtures/onset_activations.npz. Read by: tests/test_onsets.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
WAVS_DIR = FIXTURES_DIR / "wavs"

# 44.1kHz-native cases only -- see module header (no resampling support).
ONSET_CASES = {
    "mono_44100": "mono_44100.wav",
    "stereo_44100": "stereo_44100.wav",
    "float32_44100": "float32_44100.wav",
}
DSP_CASE_WAV = "mono_44100.wav"


def digest_layer(layer) -> dict:
    """Structural digest of one NN layer -- the 4b extension of
    `tools/generate_key_fixtures.py`'s `digest_layer` to also cover
    `StrideLayer` (`block_size`) and the plain `RecurrentLayer`/
    `BidirectionalLayer` stack `onsets_rnn`/`onsets_brnn` use (already
    covered by the generic `weights`/`bias`/`activation_fn` handling, no
    extra fields needed for those two)."""
    import hashlib

    def _arr_digest(arr):
        arr = np.ascontiguousarray(arr)
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }

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


def generate_onset_structural_digest() -> dict:
    import madmom
    from madmom.models import ONSETS_BRNN, ONSETS_CNN, ONSETS_RNN

    nn_rnn = madmom.ml.nn.NeuralNetwork.load(ONSETS_RNN[0])
    nn_brnn = madmom.ml.nn.NeuralNetwork.load(ONSETS_BRNN[0])
    nn_cnn = madmom.ml.nn.NeuralNetwork.load(ONSETS_CNN[0])
    return {
        "onsets_rnn_1": [digest_layer(l) for l in nn_rnn.layers],
        "onsets_brnn_1": [digest_layer(l) for l in nn_brnn.layers],
        "onsets_cnn": [digest_layer(l) for l in nn_cnn.layers],
    }


def generate_stride_layer_fixture() -> "tuple[dict, dict]":
    """Self-contained golden (input, output, real weights-free params) for
    `onsets_cnn.pkl`'s one `StrideLayer` instance -- same self-contained
    design as `generate_key_fixtures.py`'s per-layer-type fixtures (offline
    reconstructable, no unpickling needed to test it)."""
    import madmom
    from madmom.audio.signal import FramedSignalProcessor, SignalProcessor
    from madmom.audio.spectrogram import (
        FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
    )
    from madmom.audio.stft import ShortTimeFourierTransformProcessor
    from madmom.audio.filters import MelFilterbank
    from madmom.features.onsets import EPSILON, _cnn_onset_processor_pad
    from madmom.models import ONSETS_CNN
    from madmom.processors import ParallelProcessor, SequentialProcessor

    sig = SignalProcessor(num_channels=1, sample_rate=44100)
    multi = ParallelProcessor([])
    for frame_size in [2048, 1024, 4096]:
        frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
        stft = ShortTimeFourierTransformProcessor()
        filt = FilteredSpectrogramProcessor(
            filterbank=MelFilterbank, num_bands=80, fmin=27.5, fmax=16000,
            norm_filters=True, unique_filters=False)
        spec = LogarithmicSpectrogramProcessor(log=np.log, add=EPSILON)
        multi.append(SequentialProcessor((frames, stft, filt, spec)))
    pre = SequentialProcessor(
        (sig, multi, np.dstack, _cnn_onset_processor_pad))

    data = np.array(pre(str(WAVS_DIR / DSP_CASE_WAV)))

    nn = madmom.ml.nn.NeuralNetwork.load(ONSETS_CNN[0])
    assert len(nn.layers) == 8, (
        f"expected onsets_cnn.pkl to have 8 layers, got {len(nn.layers)} -- "
        "the hardcoded StrideLayer index (5) below is stale, re-inspect."
    )
    idx = 5
    layer = nn.layers[idx]
    assert type(layer).__name__ == "StrideLayer", (
        f"nn.layers[{idx}] is a {type(layer).__name__}, not StrideLayer -- "
        "index is stale, re-inspect onsets_cnn.pkl's layer stack."
    )
    replay = np.array(data, copy=True)
    for prior in nn.layers[:idx]:
        replay = prior(replay)
    layer_out = layer(replay)

    npz_payload = {
        "StrideLayer_input": np.asarray(replay),
        "StrideLayer_output": np.asarray(layer_out),
    }
    params_json = {"StrideLayer": {"block_size": int(layer.block_size)}}
    return npz_payload, params_json


def generate_dsp_function_fixtures() -> dict:
    """Golden OUTPUT-only fixtures for every pure onset-detection function
    -- see module header for why no intermediate spectrogram/phase object
    needs to be serialized."""
    from madmom.audio.filters import LogarithmicFilterbank
    from madmom.audio.signal import FramedSignalProcessor, SignalProcessor
    from madmom.audio.spectrogram import (
        FilteredSpectrogramProcessor, SpectrogramProcessor,
    )
    from madmom.audio.stft import ShortTimeFourierTransformProcessor
    from madmom.features.onsets import (
        SpectralOnsetProcessor, complex_domain, complex_flux,
        high_frequency_content, modified_kullback_leibler, phase_deviation,
        normalized_weighted_phase_deviation, rectified_complex_domain,
        spectral_diff, spectral_flux, superflux, weighted_phase_deviation,
        wrap_to_pi,
    )

    wav_path = WAVS_DIR / DSP_CASE_WAV

    sig = SignalProcessor(num_channels=1, sample_rate=44100)
    frames = FramedSignalProcessor(frame_size=2048, fps=200)
    stft_ncs = ShortTimeFourierTransformProcessor()
    stft_cs = ShortTimeFourierTransformProcessor(circular_shift=True)
    spec_proc = SpectrogramProcessor()
    filt_proc = FilteredSpectrogramProcessor(num_bands=24, norm_filters=False)

    spec = spec_proc(stft_ncs(frames(sig(str(wav_path)))))
    spec_cs = spec_proc(stft_cs(frames(sig(str(wav_path)))))
    spec_filt_cs = filt_proc(spec_cs)

    out = {}

    wrap_input = np.array(
        [-4 * np.pi, -np.pi - 0.1, -0.5, 0.0, 0.5, np.pi, np.pi + 0.1,
         4 * np.pi], dtype=np.float64)
    out["wrap_to_pi_input"] = wrap_input
    out["wrap_to_pi_output"] = np.asarray(wrap_to_pi(wrap_input))

    out["high_frequency_content"] = high_frequency_content(spec)
    out["spectral_diff"] = spectral_diff(spec)
    out["spectral_flux"] = spectral_flux(spec)
    out["superflux"] = superflux(spec)
    out["modified_kullback_leibler"] = modified_kullback_leibler(spec)

    out["phase_deviation"] = phase_deviation(spec_cs)
    out["weighted_phase_deviation"] = weighted_phase_deviation(spec_cs)
    out["normalized_weighted_phase_deviation"] = \
        normalized_weighted_phase_deviation(spec_cs)
    out["complex_domain"] = complex_domain(spec_cs)
    out["rectified_complex_domain"] = rectified_complex_domain(spec_cs)

    out["complex_flux_unfiltered"] = complex_flux(spec_cs)
    out["complex_flux_filtered"] = complex_flux(spec_filt_cs)

    sodf_default = SpectralOnsetProcessor()
    out["sodf_default"] = sodf_default(str(wav_path))
    sodf_superflux = SpectralOnsetProcessor(
        onset_method="superflux", fps=200, filterbank=LogarithmicFilterbank,
        num_bands=24, log=np.log10)
    out["sodf_superflux"] = sodf_superflux(str(wav_path))

    return out


def generate_onset_nn_fixtures() -> dict:
    """End-to-end `RNNOnsetProcessor`(online=False/True)/`CNNOnsetProcessor`
    activations + `OnsetPeakPickingProcessor` decoded onset times, for all 3
    44.1kHz-native test-wav cases."""
    from madmom.features.onsets import (
        CNNOnsetProcessor, OnsetPeakPickingProcessor, RNNOnsetProcessor,
    )

    brnn_proc = RNNOnsetProcessor(online=False)
    rnn_proc = RNNOnsetProcessor(online=True)
    cnn_proc = CNNOnsetProcessor()
    pp = OnsetPeakPickingProcessor(fps=100)

    out = {}
    for case, wav_name in ONSET_CASES.items():
        wav_path = str(WAVS_DIR / wav_name)
        act_brnn = brnn_proc(wav_path)
        act_rnn = rnn_proc(wav_path)
        act_cnn = cnn_proc(wav_path)
        out[f"{case}_brnn_activations"] = act_brnn
        out[f"{case}_rnn_activations"] = act_rnn
        out[f"{case}_cnn_activations"] = act_cnn
        out[f"{case}_brnn_onsets"] = np.asarray(pp(act_brnn))
        out[f"{case}_rnn_onsets"] = np.asarray(pp(act_rnn))
        out[f"{case}_cnn_onsets"] = np.asarray(pp(act_cnn))
    return out


def main() -> None:
    try:
        import madmom  # noqa: F401
    except ImportError as exc:
        print(
            "ERROR: this script needs the real `madmom` package (not "
            "madmom_infer), including its ONSETS_RNN/ONSETS_BRNN/ONSETS_CNN "
            "pretrained model weights. Run it with the madmom-reference "
            "venv's interpreter -- see this file's module docstring for the "
            "exact command.",
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

    print("4b-1: onsets_rnn_1/onsets_brnn_1/onsets_cnn structural digest ...")
    digest = generate_onset_structural_digest()
    (FIXTURES_DIR / "onset_structural_digest.json").write_text(
        json.dumps(digest, indent=2, sort_keys=True) + "\n"
    )

    print("4b-2: StrideLayer self-contained golden fixture ...")
    layer_fixtures, layer_params = generate_stride_layer_fixture()
    np.savez_compressed(FIXTURES_DIR / "onset_stride_layer.npz",
                         **layer_fixtures)
    (FIXTURES_DIR / "onset_stride_layer_params.json").write_text(
        json.dumps(layer_params, indent=2, sort_keys=True) + "\n"
    )

    print("4b-3: per-DSP-function golden output fixtures ...")
    dsp_fixtures = generate_dsp_function_fixtures()
    np.savez_compressed(FIXTURES_DIR / "onset_dsp_functions.npz",
                         **dsp_fixtures)

    print("4b-4: RNN/CNN onset activations + decoded onset times ...")
    nn_fixtures = generate_onset_nn_fixtures()
    np.savez_compressed(FIXTURES_DIR / "onset_activations.npz", **nn_fixtures)

    print("Done.")


if __name__ == "__main__":
    main()
