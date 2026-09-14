"""Parity + autograd tests for `madmom_infer.torch.features` -- the 9
end-to-end torch pipelines (`pipelines.py`) and the numpy-facing
`TorchPipelineProcessor` adapter (`adapter.py`).

`pytest.importorskip("torch")` at module scope, same convention as
`tests/test_torch_nn.py`/`tests/test_torch_frontend.py`. Every test that
constructs a pipeline needs real downloaded weights, so this whole file is
marked `pytest.mark.network` at collection time (module-level `pytestmark`)
-- `uv run pytest`'s default `-m 'not network'` skips it entirely, matching
`tests/test_torch_nn.py`'s own model-file tests.

Groups of checks:

1. **Parity**: for every pipeline, on each 44.1kHz fixture wav (mono
   int16, float32, stereo int16), `TorchPipelineProcessor` output vs. a
   FRESH numpy processor instance's output (fresh per case -- known
   instance-reuse caching artifact across differing wavs/dtypes, see
   CLAUDE.md's wave 4d/4e findings). Max-abs-diff is measured and
   asserted with roughly a 4x margin over the observed value, matching
   this repo's tolerance-margin convention.
2. **Gradient flow**: for downbeats/onsets_cnn/key/notes_cnn, the
   gradient of a scalar function of the pipeline's output w.r.t. the
   input waveform is finite and non-zero.
3. **Batching**: a batched (B=2) call equals stacking 2 independent
   unbatched calls.
4. **CUDA vs CPU parity**: skipped if no CUDA device is available.

Reads: madmom_infer.torch.features (build_pipeline, TorchPipelineProcessor,
waveform_from_signal), madmom_infer.audio.signal (SignalProcessor) -- the
numpy processors under tests/fixtures/wavs are imported lazily per pipeline
name via `_NUMPY_PROCESSORS`.
"""

import importlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from madmom_infer.audio.signal import SignalProcessor  # noqa: E402
from madmom_infer.torch.features import build_pipeline  # noqa: E402
from madmom_infer.torch.features.adapter import (  # noqa: E402
    TorchPipelineProcessor,
    waveform_from_signal,
)

pytestmark = pytest.mark.network

FIXTURE_WAVS = [
    "tests/fixtures/wavs/mono_44100.wav",
    "tests/fixtures/wavs/float32_44100.wav",
    "tests/fixtures/wavs/stereo_44100.wav",
]

# name -> (module, class) of the numpy processor this pipeline mirrors.
_NUMPY_PROCESSORS = {
    "downbeats": ("madmom_infer.features.downbeats", "RNNDownBeatProcessor", {}),
    "beats": ("madmom_infer.features.beats", "RNNBeatProcessor", {}),
    "onsets_rnn": ("madmom_infer.features.onsets", "RNNOnsetProcessor", {}),
    "onsets_cnn": ("madmom_infer.features.onsets", "CNNOnsetProcessor", {}),
    "key": ("madmom_infer.features.key", "CNNKeyRecognitionProcessor", {}),
    "chroma": ("madmom_infer.audio.chroma", "DeepChromaProcessor", {}),
    "chords_feat": ("madmom_infer.features.chords", "CNNChordFeatureProcessor", {}),
    "notes_rnn": ("madmom_infer.features.notes", "RNNPianoNoteProcessor", {}),
    "notes_cnn": ("madmom_infer.features.notes", "CNNPianoNoteProcessor", {}),
}


def _fresh_numpy_processor(name):
    mod_name, cls_name, kwargs = _NUMPY_PROCESSORS[name]
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)(**kwargs)


def _load_waveform(wav_path, dtype=np.float32):
    signal = SignalProcessor(num_channels=1, sample_rate=44100)(wav_path)
    return waveform_from_signal(signal, dtype=dtype)


# ---------------------------------------------------------------------
# 1. parity
# ---------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(_NUMPY_PROCESSORS))
@pytest.mark.parametrize("wav_path", FIXTURE_WAVS)
def test_pipeline_matches_numpy_processor(name, wav_path):
    numpy_proc = _fresh_numpy_processor(name)
    numpy_out = np.asarray(numpy_proc(wav_path))

    pipeline = build_pipeline(name)
    adapter = TorchPipelineProcessor(pipeline)
    torch_out = adapter(wav_path)

    assert numpy_out.shape == torch_out.shape, (
        f"{name}/{wav_path}: numpy shape {numpy_out.shape} != "
        f"torch shape {torch_out.shape}"
    )
    diff = np.abs(numpy_out.astype(np.float64) - torch_out.astype(np.float64))
    max_diff = float(diff.max())
    # ~4x margin over empirically observed drift (measured up to ~6e-4 for
    # the float32 fixture on the deepest pipelines) -- matches this repo's
    # tolerance-margin convention (see e.g. tests/test_key.py).
    assert max_diff < 5e-3, (
        f"{name}/{wav_path}: max abs diff {max_diff!r} exceeds tolerance"
    )


# ---------------------------------------------------------------------
# 2. gradient flow
# ---------------------------------------------------------------------
@pytest.mark.parametrize("name", ["downbeats", "onsets_cnn", "key", "notes_cnn"])
def test_gradient_flows_to_waveform(name):
    waveform = torch.tensor(
        _load_waveform(FIXTURE_WAVS[0]), dtype=torch.float32, requires_grad=True
    )
    pipeline = build_pipeline(name)
    output = pipeline(waveform)
    # a plain `.sum()` of a softmax output (the "key" pipeline) is
    # identically 1 regardless of input -- its gradient is exactly zero by
    # construction, not a bug. Sum of squares avoids that degeneracy for
    # every pipeline uniformly.
    loss = (output.float() ** 2).sum()
    loss.backward()
    assert waveform.grad is not None
    assert torch.isfinite(waveform.grad).all()
    assert waveform.grad.abs().sum() > 0


# ---------------------------------------------------------------------
# 3. batching
# ---------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(_NUMPY_PROCESSORS))
def test_batched_matches_unbatched(name):
    waveform = torch.tensor(_load_waveform(FIXTURE_WAVS[0]), dtype=torch.float32)
    pipeline = build_pipeline(name)
    pipeline.eval()

    with torch.no_grad():
        out_a = pipeline(waveform)
        out_b = pipeline(waveform * 0.9)
        batched = torch.stack([waveform, waveform * 0.9])
        out_batched = pipeline(batched)

    assert out_batched.shape[0] == 2
    assert torch.allclose(out_batched[0], out_a, atol=1e-4)
    assert torch.allclose(out_batched[1], out_b, atol=1e-4)


# ---------------------------------------------------------------------
# 4. CUDA vs CPU parity
# ---------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("name", sorted(_NUMPY_PROCESSORS))
def test_cuda_matches_cpu(name):
    waveform_cpu = torch.tensor(_load_waveform(FIXTURE_WAVS[0]), dtype=torch.float32)
    waveform_gpu = waveform_cpu.cuda()

    pipeline_cpu = build_pipeline(name).eval()
    pipeline_gpu = build_pipeline(name).eval().cuda()

    with torch.no_grad():
        out_cpu = pipeline_cpu(waveform_cpu)
        out_gpu = pipeline_gpu(waveform_gpu).cpu()

    max_diff = (out_cpu.double() - out_gpu.double()).abs().max().item()
    assert max_diff < 5e-3, f"{name}: cuda/cpu max abs diff {max_diff!r}"
