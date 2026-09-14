"""Parity tests for `backend="torch"` wired into the numpy-facing NN
processors (`madmom_infer/backends.py` + each processor's own
`backend=`/`device=` kwargs) and `MadmomAnalyzer`.

`pytest.importorskip("torch")` at module scope, same convention as
`tests/test_torch_pipelines.py`. Every test that constructs a real
`backend="torch"` processor needs downloaded weights, so those parity
checks are marked `pytest.mark.network`.

Groups of checks:

1. **Per-processor parity**: for each of the 10 NN-backed processors,
   `Cls(backend="torch")(wav)` vs a fresh `Cls()(wav)` (numpy) on the 3
   44.1kHz fixture wavs -- fresh instances per case, same instance-reuse
   caching-artifact discipline `tests/test_torch_pipelines.py` and
   `tests/test_downbeats_rnn.py` already establish. `RNNBarProcessor` is
   compared on its own actual input form (a signal + decoded beat times),
   not a bare waveform, matching `tests/test_downbeats_rnn.py`'s "RNNBarProcessor
   end-to-end" section.
2. **`MadmomAnalyzer(backend="torch")` end-to-end**: decoded results equal
   the numpy backend's own decoded results (exact for event times/labels,
   allclose for chroma) on one fixture wav, across every NN-backed task.
3. **CUDA variant** of (1)/(2), skipped if no CUDA device is available.
4. **Validation**: an unknown `backend` raises `ValueError`; passing
   `device=` together with `backend="numpy"` raises `ValueError`; and MPS
   is rejected before model construction by the shared pipeline, direct
   adapter, and `RNNBarProcessor` paths.

The torch-free guard (a numpy-backend processor never imports torch) lives
in a separate file, `tests/test_backends_torch_free.py`, so it can assert
import purity without this file's module-level `importorskip("torch")`.

Reads: madmom_infer.backends (validate_backend), the 10 NN-backed
processor modules (imported lazily), madmom_infer.api (MadmomAnalyzer).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

FIXTURE_WAVS = [
    "tests/fixtures/wavs/mono_44100.wav",
    "tests/fixtures/wavs/float32_44100.wav",
    "tests/fixtures/wavs/stereo_44100.wav",
]
MONO_WAV = FIXTURE_WAVS[0]

# name -> (module, class, extra kwargs, tolerance) -- tolerance mirrors
# tests/test_torch_pipelines.py's _PARITY_TOLERANCE (same underlying
# pipelines, so the same measured drift applies).
_PROCESSORS = {
    "downbeats": ("madmom_infer.features.downbeats", "RNNDownBeatProcessor", {}, 6e-4),
    "beats": ("madmom_infer.features.beats", "RNNBeatProcessor", {}, 2e-7),
    "onsets_rnn": ("madmom_infer.features.onsets", "RNNOnsetProcessor", {}, 2e-6),
    "onsets_cnn": ("madmom_infer.features.onsets", "CNNOnsetProcessor", {}, 6e-6),
    "key": ("madmom_infer.features.key", "CNNKeyRecognitionProcessor", {}, 1.5e-7),
    "chroma": ("madmom_infer.audio.chroma", "DeepChromaProcessor", {}, 3e-6),
    "chords_feat": ("madmom_infer.features.chords", "CNNChordFeatureProcessor", {}, 3e-6),
    "notes_rnn": ("madmom_infer.features.notes", "RNNPianoNoteProcessor", {}, 5e-6),
    "notes_cnn": ("madmom_infer.features.notes", "CNNPianoNoteProcessor", {}, 2e-5),
}


def _cls(name):
    import importlib

    mod_name, cls_name, _kwargs, _tol = _PROCESSORS[name]
    return getattr(importlib.import_module(mod_name), cls_name)


# ---------------------------------------------------------------------
# 1. per-processor parity (CPU)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(_PROCESSORS))
@pytest.mark.parametrize("wav_path", FIXTURE_WAVS)
@pytest.mark.network
def test_processor_torch_backend_matches_numpy(name, wav_path):
    cls = _cls(name)
    _, _, kwargs, tolerance = _PROCESSORS[name]

    numpy_out = np.asarray(cls(**kwargs)(wav_path))
    torch_out = np.asarray(cls(backend="torch", **kwargs)(wav_path))

    assert numpy_out.shape == torch_out.shape
    diff = np.abs(numpy_out.astype(np.float64) - torch_out.astype(np.float64))
    max_diff = float(diff.max())
    assert max_diff < tolerance, (
        f"{name}/{wav_path}: max abs diff {max_diff!r} exceeds "
        f"tolerance {tolerance!r}"
    )


@pytest.mark.network
def test_rnn_bar_processor_torch_backend_matches_numpy():
    """`RNNBarProcessor` keeps its numpy frontend regardless of backend
    (see its docstring) -- only the two GRU ensembles run through torch.
    Compared on its real input shape: (wav_path, decoded beat times)."""
    import warnings

    from madmom_infer.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
    from madmom_infer.features.downbeats import RNNBarProcessor

    rnn = RNNBeatProcessor(online=False)
    act = rnn(MONO_WAV)
    dbn = DBNBeatTrackingProcessor(fps=100)
    beats = dbn(act)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        numpy_out = RNNBarProcessor()((MONO_WAV, beats))
        torch_out = RNNBarProcessor(backend="torch")((MONO_WAV, beats))

    assert numpy_out.shape == torch_out.shape
    np.testing.assert_allclose(
        numpy_out, torch_out, equal_nan=True, rtol=0, atol=1e-5)


# ---------------------------------------------------------------------
# 2. MadmomAnalyzer(backend="torch") end-to-end
# ---------------------------------------------------------------------
@pytest.mark.network
def test_analyzer_torch_backend_matches_numpy():
    from madmom_infer.api import MadmomAnalyzer

    tasks = ("onsets", "beats", "downbeats", "key", "chords", "chroma", "notes")
    numpy_result = MadmomAnalyzer(tasks=tasks)(MONO_WAV)
    torch_result = MadmomAnalyzer(tasks=tasks, backend="torch")(MONO_WAV)

    np.testing.assert_array_equal(numpy_result["onsets"], torch_result["onsets"])
    np.testing.assert_array_equal(numpy_result["beats"], torch_result["beats"])
    np.testing.assert_array_equal(numpy_result["downbeats"], torch_result["downbeats"])
    assert numpy_result["key"] == torch_result["key"]
    np.testing.assert_allclose(
        numpy_result["chroma"], torch_result["chroma"], rtol=0, atol=3e-6)
    numpy_chords, torch_chords = numpy_result["chords"], torch_result["chords"]
    assert numpy_chords.shape == torch_chords.shape
    np.testing.assert_allclose(numpy_chords["start"], torch_chords["start"], atol=1e-6)
    np.testing.assert_allclose(numpy_chords["end"], torch_chords["end"], atol=1e-6)
    assert list(numpy_chords["label"]) == list(torch_chords["label"])
    np.testing.assert_array_equal(numpy_result["notes"], torch_result["notes"])


# ---------------------------------------------------------------------
# 3. CUDA variant
# ---------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("name", sorted(_PROCESSORS))
@pytest.mark.network
def test_processor_torch_backend_cuda_matches_numpy(name):
    cls = _cls(name)
    _, _, kwargs, _tol = _PROCESSORS[name]

    numpy_out = np.asarray(cls(**kwargs)(MONO_WAV))
    torch_out = np.asarray(cls(backend="torch", device="cuda", **kwargs)(MONO_WAV))

    assert numpy_out.shape == torch_out.shape
    # looser than the CPU tolerance -- torch's default TF32 matmul/cudnn on
    # Ampere+ GPUs measurably drifts the CNN-heavy pipelines up to ~1e-3
    # from the fp32 numpy reference (see
    # madmom_infer/torch/features/adapter.py's module header).
    max_diff = float(
        np.abs(numpy_out.astype(np.float64) - torch_out.astype(np.float64)).max())
    assert max_diff < 5e-3, f"{name}: cuda max abs diff {max_diff!r}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.network
def test_analyzer_torch_backend_cuda_smoke():
    from madmom_infer.api import MadmomAnalyzer

    result = MadmomAnalyzer(tasks=("beats",), backend="torch", device="cuda")(MONO_WAV)
    assert np.asarray(result["beats"]).size > 0


# ---------------------------------------------------------------------
# 4. validation
# ---------------------------------------------------------------------
def test_unknown_backend_raises():
    from madmom_infer.features.beats import RNNBeatProcessor

    with pytest.raises(ValueError):
        RNNBeatProcessor(backend="jax")


def test_device_with_numpy_backend_raises():
    from madmom_infer.features.beats import RNNBeatProcessor

    with pytest.raises(ValueError):
        RNNBeatProcessor(backend="numpy", device="cpu")


def test_analyzer_unknown_backend_raises():
    from madmom_infer.api import MadmomAnalyzer

    with pytest.raises(ValueError):
        MadmomAnalyzer(tasks=("beats",), backend="jax")


def test_analyzer_device_with_numpy_backend_raises():
    from madmom_infer.api import MadmomAnalyzer

    with pytest.raises(ValueError):
        MadmomAnalyzer(tasks=("beats",), backend="numpy", device="cpu")


@pytest.mark.parametrize("device", ["mps", "mps:0"])
def test_torch_pipeline_processor_rejects_mps_before_build(device, monkeypatch):
    from madmom_infer.backends import torch_pipeline_processor
    import madmom_infer.torch.features as torch_features

    def fail_build_pipeline(*_args, **_kwargs):
        raise AssertionError("build_pipeline should not run for MPS devices")

    monkeypatch.setattr(torch_features, "build_pipeline", fail_build_pipeline)

    with pytest.raises(ValueError, match="MPS|mps"):
        torch_pipeline_processor("beats", device=device)


@pytest.mark.parametrize("device", ["mps", "mps:0"])
def test_torch_pipeline_processor_adapter_rejects_mps(device):
    from madmom_infer.torch.features import TorchPipelineProcessor

    with pytest.raises(ValueError, match="MPS|mps"):
        TorchPipelineProcessor(torch.nn.Identity(), device=device)


@pytest.mark.parametrize("device", ["mps", "mps:0"])
def test_rnn_bar_processor_rejects_mps_before_model_loading(device, monkeypatch):
    from madmom_infer.features.downbeats import RNNBarProcessor
    import madmom_infer.models as models

    def fail_model_lookup():
        raise AssertionError("model lookup should not run for MPS devices")

    monkeypatch.setattr(models, "downbeats_bgru", fail_model_lookup)

    with pytest.raises(ValueError, match="MPS|mps"):
        RNNBarProcessor(backend="torch", device=device)


def test_model_file_override_with_torch_backend_raises():
    from madmom_infer.features.beats import RNNBeatProcessor

    with pytest.raises(NotImplementedError):
        RNNBeatProcessor(backend="torch", nn_files=["not-a-real-file.pkl"])
