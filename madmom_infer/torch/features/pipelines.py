"""Differentiable, GPU-capable torch twins of the 9 audio-in-activations-out
numpy `SequentialProcessor`s in `madmom_infer/features/*.py` and
`madmom_infer/audio/chroma.py` (downbeats, beats, onset-RNN, onset-CNN,
key, chroma, chord-features, note-RNN, note-CNN).

Each pipeline is an `nn.Module` composing:

1. one or more `madmom_infer.torch.audio.frontend.SpectrogramFrontend`
   branches (framing -> STFT -> filterbank -> log-compress -> [temporal
   diff]), reusing that module's numpy-derived window/filterbank/diff
   construction -- never re-deriving DSP constants here;
2. a stacking step across branches when there is more than one
   (`torch.cat` for the `np.hstack`-style processors, `DstackModule` for
   `CNNOnsetProcessor`'s `np.dstack`);
3. zero or more of `common.py`'s small pad/context/superframe helpers,
   matching each numpy processor's own extra pre/post-NN step; and
4. the pretrained network itself, converted once at construction time via
   `madmom_infer.torch.ml.nn.to_torch(NeuralNetwork.load(...) |
   NeuralNetworkEnsemble.load(...))`.

Every `__init__` parameter that isn't a DSP constant (frame sizes,
`num_bands`, `fmin`/`fmax`, log `mul`/`add`, `diff_ratio`, pad widths) is
cited to the exact numpy `__init__` it mirrors -- see each class's
docstring. `build_pipeline(name, **kwargs)` is the one-stop registry the
adapter (`madmom_infer/torch/features/adapter.py`) and tests use.

Reads: torch, madmom_infer.audio.filters (LogarithmicFilterbank,
MelFilterbank, constants), madmom_infer.torch.audio.frontend
(SpectrogramFrontend, EPSILON), madmom_infer.torch.features.common (pad/
superframe helpers), madmom_infer.torch.ml.nn (to_torch,
DstackModule, StrideLayer), madmom_infer.ml.nn (NeuralNetwork,
NeuralNetworkEnsemble), madmom_infer.models (weight-file registry
functions); read by: madmom_infer/torch/features/__init__.py,
madmom_infer/torch/features/adapter.py, tests/test_torch_pipelines.py,
tools/compare_torch_backend.py.
"""

from __future__ import annotations

import torch
from torch import nn

from madmom_infer.audio.filters import A4, FMAX, FMIN, MelFilterbank
from madmom_infer.ml.nn import NeuralNetwork, NeuralNetworkEnsemble
from madmom_infer.torch.audio.frontend import EPSILON, SpectrogramFrontend, frame_signal
from madmom_infer.torch.ml.nn import DstackModule, to_torch

from .common import EdgeRepeatPad, SuperframeAverage, ZeroPad, ensure_batched, unbatch

__all__ = [
    "DownbeatsPipeline",
    "BeatsPipeline",
    "OnsetRNNPipeline",
    "OnsetCNNPipeline",
    "KeyPipeline",
    "ChromaPipeline",
    "ChordFeaturePipeline",
    "NoteRNNPipeline",
    "NoteCNNPipeline",
    "build_pipeline",
]

_SAMPLE_RATE = 44100


class _MultiBranchFrontend(nn.Module):
    """Run several `SpectrogramFrontend` branches over the same waveform
    and combine them, either `torch.cat` on the feature axis (the
    `np.hstack` shape every RNN-family processor uses) or `DstackModule`
    (the `np.dstack` shape `CNNOnsetProcessor` uses). Branch order is
    preserved exactly as given -- load-bearing for `CNNOnsetProcessor`,
    whose 3 branches feed 3 fixed input channels of one CNN.
    """

    def __init__(self, branch_kwargs, stack="cat"):
        super().__init__()
        self.branches = nn.ModuleList(
            [SpectrogramFrontend(**kw) for kw in branch_kwargs]
        )
        if stack not in ("cat", "dstack"):
            raise ValueError(f"stack must be 'cat' or 'dstack', got {stack!r}")
        self.stack = stack
        self._dstack = DstackModule() if stack == "dstack" else None

    def forward(self, waveform):
        outs = [branch(waveform) for branch in self.branches]
        if self.stack == "cat":
            return torch.cat(outs, dim=-1)
        return self._dstack(outs)


class _ContextStack(nn.Module):
    """Stack `frame_size` neighboring spectrogram frames (default hop 1)
    into one flattened context vector per output frame -- torch twin of
    `DeepChromaProcessor`'s `FramedSignalProcessor(frame_size=15,
    hop_size=1, fps=10)` + `_dcp_flatten` (`np.concatenate(fs).reshape
    (len(fs), -1)`).

    Reuses `madmom_infer.torch.audio.frontend.frame_signal` (the same
    origin/end-aware framing `SpectrogramFrontend` itself uses) rather
    than `madmom_infer.torch.ml.nn.StrideLayer`'s `Tensor.unfold`-based
    windowing -- `StrideLayer` mirrors `segment_axis`'s no-padding
    (`end='cut'`) semantics, which drops `frame_size - 1` frames off the
    end; `FramedSignalProcessor`'s default `origin=0, end='normal'`
    zero-pads at the boundaries instead, preserving the input frame
    count. Confirmed by comparing against `DeepChromaProcessor` directly
    (`tests/test_torch_pipelines.py`) -- using `StrideLayer` here silently
    produced a far shorter, mis-aligned output.
    """

    def __init__(self, frame_size=15, hop_size=1, origin=0, end="normal"):
        super().__init__()
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.origin = origin
        self.end = end

    def forward(self, x):
        # x: (B, T, F) -> window along T (frame_signal windows the LAST
        # axis) -> (B, F, T', frame_size) -> (B, T', frame_size, F) ->
        # (B, T', frame_size * F), matching np.concatenate(fs)'s
        # window-then-feature flattening order.
        x_t = x.transpose(1, 2)
        frames = frame_signal(
            x_t, self.frame_size, self.hop_size, origin=self.origin, end=self.end
        )
        frames = frames.permute(0, 2, 3, 1)
        b, t = frames.shape[0], frames.shape[1]
        return frames.reshape(b, t, -1)


def _branch_kwargs(frame_size, num_bands, fmin=FMIN, fmax=FMAX, mul=1, add=1,
                    diff_ratio=None, positive_diffs=True, fps=100,
                    sample_rate=_SAMPLE_RATE, dtype=torch.float32):
    kw = dict(
        sample_rate=sample_rate, frame_size=frame_size, fps=fps,
        num_bands=num_bands, fmin=fmin, fmax=fmax, fref=A4,
        norm_filters=True, unique_filters=True,
        log_mul=mul, log_add=add, dtype=dtype,
    )
    if diff_ratio is None:
        kw["include_diff"] = False
    else:
        kw["include_diff"] = True
        kw["diff_ratio"] = diff_ratio
        kw["positive_diffs"] = positive_diffs
    return kw


# ---------------------------------------------------------------------------
# RNNDownBeatProcessor (features/downbeats.py:161-182)
# ---------------------------------------------------------------------------
class DownbeatsPipeline(nn.Module):
    """Torch twin of `madmom_infer.features.downbeats.RNNDownBeatProcessor`
    up to and including the RNN ensemble and the "drop non-beat column"
    step (`np.delete(..., obj=0, axis=1)`) -- everything except the DBN
    decoder, which stays numpy-only (sequential, discrete, no GPU/autograd
    benefit -- see `torch/audio/frontend.py`'s module docstring).

    `forward(waveform)`: `(N,)` or `(B, N)` mono float @ 44.1kHz ->
    `(T, 2)` or `(B, T, 2)` `[beat, downbeat]` activations.
    """

    def __init__(self, dtype=torch.float32):
        super().__init__()
        from madmom_infer.models import downbeats_blstm

        branch_kwargs = [
            _branch_kwargs(fs, nb, diff_ratio=0.5, positive_diffs=True, dtype=dtype)
            for fs, nb in zip((1024, 2048, 4096), (3, 6, 12))
        ]
        self.frontend = _MultiBranchFrontend(branch_kwargs, stack="cat")
        ensemble = NeuralNetworkEnsemble.load(downbeats_blstm())
        self.nn = to_torch(ensemble)

    def forward(self, waveform):
        x, was_unbatched = ensure_batched(waveform)
        feats = self.frontend(x)
        preds = self.nn(feats)
        preds = preds[..., 1:]  # drop the "non-beat" column
        return unbatch(preds, was_unbatched)


# ---------------------------------------------------------------------------
# RNNBeatProcessor (features/beats.py:108-155)
# ---------------------------------------------------------------------------
class BeatsPipeline(nn.Module):
    """Torch twin of `madmom_infer.features.beats.RNNBeatProcessor`.
    `online=True` selects `BEATS_LSTM` (single 2048 frame, 12 bands);
    `online=False` (default) selects `BEATS_BLSTM` (1024/2048/4096, 6
    bands each) -- same offline-compatibility shape as upstream (see that
    class's docstring).

    **Measured numerical note**: the BLSTM ensemble is numerically
    ill-conditioned on some inputs -- on a percussion-free keyboard loop, a
    5e-7 perturbation of the INPUT changes the numpy reference's own output
    by up to 1.6e-2. Torch-vs-numpy activation diffs as large as ~1e-2 are
    therefore expected on such audio and are not a correctness bug in
    either backend; decoded beats still matched exactly in every case
    tested.

    `forward(waveform)`: `(N,)`/`(B, N)` -> `(T,)`/`(B, T)` beat activation.
    """

    def __init__(self, online=False, dtype=torch.float32):
        super().__init__()
        from madmom_infer.models import beats_blstm, beats_lstm

        if online:
            model_files = beats_lstm()
            frame_sizes, num_bands = (2048,), 12
        else:
            model_files = beats_blstm()
            frame_sizes, num_bands = (1024, 2048, 4096), 6

        branch_kwargs = [
            _branch_kwargs(fs, num_bands, diff_ratio=0.5, positive_diffs=True, dtype=dtype)
            for fs in frame_sizes
        ]
        self.frontend = _MultiBranchFrontend(branch_kwargs, stack="cat")
        ensemble = NeuralNetworkEnsemble.load(model_files)
        self.nn = to_torch(ensemble)

    def forward(self, waveform):
        x, was_unbatched = ensure_batched(waveform)
        feats = self.frontend(x)
        preds = self.nn(feats)
        return unbatch(preds, was_unbatched)


# ---------------------------------------------------------------------------
# RNNOnsetProcessor (features/onsets.py:422-447)
# ---------------------------------------------------------------------------
class OnsetRNNPipeline(nn.Module):
    """Torch twin of `madmom_infer.features.onsets.RNNOnsetProcessor`.
    `online=True` -> `ONSETS_RNN` (frame sizes 512/1024/2048);
    `online=False` (default) -> `ONSETS_BRNN` (1024/2048/4096). Both use
    `num_bands=6`, `fmin=30`, `fmax=17000`, `mul=5`, `diff_ratio=0.25`.

    `forward(waveform)`: `(N,)`/`(B, N)` -> `(T,)`/`(B, T)` onset activation.
    """

    def __init__(self, online=False, dtype=torch.float32):
        super().__init__()
        from madmom_infer.models import onsets_brnn, onsets_rnn

        frame_sizes = (512, 1024, 2048) if online else (1024, 2048, 4096)
        model_files = onsets_rnn() if online else onsets_brnn()

        branch_kwargs = [
            _branch_kwargs(fs, 6, mul=5, add=1, diff_ratio=0.25,
                            positive_diffs=True, dtype=dtype)
            for fs in frame_sizes
        ]
        self.frontend = _MultiBranchFrontend(branch_kwargs, stack="cat")
        ensemble = NeuralNetworkEnsemble.load(model_files)
        self.nn = to_torch(ensemble)

    def forward(self, waveform):
        x, was_unbatched = ensure_batched(waveform)
        feats = self.frontend(x)
        preds = self.nn(feats)
        return unbatch(preds, was_unbatched)


# ---------------------------------------------------------------------------
# CNNOnsetProcessor (features/onsets.py:459-492)
# ---------------------------------------------------------------------------
class OnsetCNNPipeline(nn.Module):
    """Torch twin of `madmom_infer.features.onsets.CNNOnsetProcessor`: 3
    Mel-filtered (80 bands, fmin 27.5, fmax 16000) branches at frame sizes
    `[2048, 1024, 4096]` (this exact order, load-bearing -- see
    `_cnn_onset_processor_pad`/`np.dstack` order in the numpy class),
    natural-log compressed (`mul=1, add=EPSILON`), `np.dstack`-ed into 3
    channels, edge-repeat-padded by 7 frames, fed to the single
    `onsets_cnn` network.

    `forward(waveform)`: `(N,)`/`(B, N)` -> `(T,)`/`(B, T)` onset activation.
    """

    def __init__(self, dtype=torch.float32):
        super().__init__()
        from madmom_infer.models import onsets_cnn

        branch_kwargs = [
            dict(
                sample_rate=_SAMPLE_RATE, frame_size=fs, fps=100,
                num_bands=80, fmin=27.5, fmax=16000, fref=A4,
                norm_filters=True, unique_filters=False,
                filterbank_cls=MelFilterbank,
                log_mul=1, log_add=EPSILON, natural_log=True,
                include_diff=False, dtype=dtype,
            )
            for fs in (2048, 1024, 4096)
        ]
        self.frontend = _MultiBranchFrontend(branch_kwargs, stack="dstack")
        self.pad = EdgeRepeatPad(7)
        network = NeuralNetwork.load(onsets_cnn()[0])
        self.nn = to_torch(network)

    def forward(self, waveform):
        x, was_unbatched = ensure_batched(waveform)
        feats = self.frontend(x)  # (B, T, 80, 3)
        feats = self.pad(feats)
        preds = self.nn(feats)
        return unbatch(preds, was_unbatched)


# ---------------------------------------------------------------------------
# CNNKeyRecognitionProcessor (features/key.py:147-163)
# ---------------------------------------------------------------------------
class KeyPipeline(nn.Module):
    """Torch twin of
    `madmom_infer.features.key.CNNKeyRecognitionProcessor`: frame 8192,
    fps 5, `LogarithmicFilterbank(24, fmin=65, fmax=2100, unique_filters=
    True)`, `log10(1 * x + 1)`, `key_cnn` ensemble, softmax over the
    24 key classes, then re-inserts the length-1 axis
    `madmom_infer.features.key.add_axis` prepends (an ensemble-of-1
    reshape trick, numpy-specific in origin but reproduced here purely
    for output-shape parity with `CNNKeyRecognitionProcessor`).

    `forward(waveform)`: `(N,)`/`(B, N)` -> `(1, 24)`/`(B, 1, 24)` key
    probabilities.
    """

    def __init__(self, dtype=torch.float32):
        super().__init__()
        from madmom_infer.models import key_cnn

        self.frontend = SpectrogramFrontend(
            sample_rate=_SAMPLE_RATE, frame_size=8192, fps=5,
            num_bands=24, fmin=65, fmax=2100, fref=A4,
            norm_filters=True, unique_filters=True,
            log_mul=1, log_add=1, include_diff=False, dtype=dtype,
        )
        ensemble = NeuralNetworkEnsemble.load(key_cnn())
        self.nn = to_torch(ensemble)

    def forward(self, waveform):
        x, was_unbatched = ensure_batched(waveform)
        feats = self.frontend(x)
        preds = self.nn(feats)
        preds = torch.softmax(preds, dim=-1)
        preds = preds.unsqueeze(-2)  # re-insert add_axis's length-1 axis
        return unbatch(preds, was_unbatched)


# ---------------------------------------------------------------------------
# DeepChromaProcessor (audio/chroma.py:229-292)
# ---------------------------------------------------------------------------
class ChromaPipeline(nn.Module):
    """Torch twin of `madmom_infer.audio.chroma.DeepChromaProcessor`:
    frame 8192, fps 10, `LogarithmicFilterbank(24, fmin, fmax,
    unique_filters)`, `log10(1 * x + 1)`, a 15-frame/hop-1 context window
    flattened per window (`_ContextStack(15)` -- see that class's
    docstring for why it reuses `frame_signal` rather than `StrideLayer`),
    fed to the `chroma_dnn` network.

    `forward(waveform)`: `(N,)`/`(B, N)` -> `(T, 24)`/`(B, T, 24)` chroma.
    """

    def __init__(self, fmin=65, fmax=2100, unique_filters=True, dtype=torch.float32):
        super().__init__()
        from madmom_infer.models import chroma_dnn

        self.frontend = SpectrogramFrontend(
            sample_rate=_SAMPLE_RATE, frame_size=8192, fps=10,
            num_bands=24, fmin=fmin, fmax=fmax, fref=A4,
            norm_filters=True, unique_filters=unique_filters,
            log_mul=1, log_add=1, include_diff=False, dtype=dtype,
        )
        self.context = _ContextStack(15, hop_size=1, origin=0, end="normal")
        ensemble = NeuralNetworkEnsemble.load(chroma_dnn())
        self.nn = to_torch(ensemble)

    def forward(self, waveform):
        x, was_unbatched = ensure_batched(waveform)
        feats = self.frontend(x)
        feats = self.context(feats)
        preds = self.nn(feats)
        return unbatch(preds, was_unbatched)


# ---------------------------------------------------------------------------
# CNNChordFeatureProcessor (features/chords.py:191-250)
# ---------------------------------------------------------------------------
class ChordFeaturePipeline(nn.Module):
    """Torch twin of
    `madmom_infer.features.chords.CNNChordFeatureProcessor`: frame 8192,
    fps 10, `LogarithmicFilterbank(24, fmin=60, fmax=2600,
    unique_filters=True)`, `log10(1 * x + 1)`, 11-zero-frame padding on
    both ends (`_cnncfp_pad`), `chords_cnn_feat` network, then a
    superframe average (`SuperframeAverage(3)` -- `_cnncfp_superframes`
    (`segment_axis(3, 1)`) + `_cnncfp_avg` (`.mean((1, 2))`) combined).

    `forward(waveform)`: `(N,)`/`(B, N)` -> `(T, 128)`/`(B, T, 128)`
    chord features.
    """

    def __init__(self, dtype=torch.float32):
        super().__init__()
        from madmom_infer.models import chords_cnn_feat

        self.frontend = SpectrogramFrontend(
            sample_rate=_SAMPLE_RATE, frame_size=8192, fps=10,
            num_bands=24, fmin=60, fmax=2600, fref=A4,
            norm_filters=True, unique_filters=True,
            log_mul=1, log_add=1, include_diff=False, dtype=dtype,
        )
        self.pad = ZeroPad(11)
        network = NeuralNetwork.load(chords_cnn_feat()[0])
        self.nn = to_torch(network)
        self.superframes = SuperframeAverage(3)

    def forward(self, waveform):
        x, was_unbatched = ensure_batched(waveform)
        feats = self.frontend(x)
        feats = self.pad(feats)
        preds = self.nn(feats)
        preds = self.superframes(preds)
        return unbatch(preds, was_unbatched)


# ---------------------------------------------------------------------------
# RNNPianoNoteProcessor (features/notes.py:92-141)
# ---------------------------------------------------------------------------
class NoteRNNPipeline(nn.Module):
    """Torch twin of `madmom_infer.features.notes.RNNPianoNoteProcessor`:
    3 branches (1024/2048/4096), `num_bands=12`, `mul=5`, `diff_ratio=0.5`,
    `np.hstack`-ed, fed to the single `notes_brnn` network.

    `forward(waveform)`: `(N,)`/`(B, N)` -> `(T, 88)`/`(B, T, 88)` note
    onset activations.
    """

    def __init__(self, dtype=torch.float32):
        super().__init__()
        from madmom_infer.models import notes_brnn

        branch_kwargs = [
            _branch_kwargs(fs, 12, mul=5, add=1, diff_ratio=0.5,
                            positive_diffs=True, dtype=dtype)
            for fs in (1024, 2048, 4096)
        ]
        self.frontend = _MultiBranchFrontend(branch_kwargs, stack="cat")
        network = NeuralNetwork.load(notes_brnn()[0])
        self.nn = to_torch(network)

    def forward(self, waveform):
        x, was_unbatched = ensure_batched(waveform)
        feats = self.frontend(x)
        preds = self.nn(feats)
        return unbatch(preds, was_unbatched)


# ---------------------------------------------------------------------------
# CNNPianoNoteProcessor (features/notes.py:289-342)
# ---------------------------------------------------------------------------
class NoteCNNPipeline(nn.Module):
    """Torch twin of `madmom_infer.features.notes.CNNPianoNoteProcessor`:
    single frame_size=4096, fps=50, `LogarithmicFilterbank(24, fmin=30,
    fmax=10000)`, `log10(1 * x + 1)`, edge-repeat-padded by 5 frames, fed
    to the multi-task `notes_cnn` graph.

    `forward(waveform)`: `(N,)`/`(B, N)` -> `(T, 88, 3)`/`(B, T, 88, 3)`
    `[note, onset, offset]` activations.
    """

    def __init__(self, dtype=torch.float32):
        super().__init__()
        from madmom_infer.models import notes_cnn

        self.frontend = SpectrogramFrontend(
            sample_rate=_SAMPLE_RATE, frame_size=4096, fps=50,
            num_bands=24, fmin=30, fmax=10000, fref=A4,
            norm_filters=True, unique_filters=True,
            log_mul=1, log_add=1, include_diff=False, dtype=dtype,
        )
        self.pad = EdgeRepeatPad(5)
        ensemble = NeuralNetworkEnsemble.load(notes_cnn())
        self.nn = to_torch(ensemble)

    def forward(self, waveform):
        x, was_unbatched = ensure_batched(waveform)
        feats = self.frontend(x)
        feats = self.pad(feats)
        preds = self.nn(feats)
        return unbatch(preds, was_unbatched)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
_PIPELINES = {
    "downbeats": DownbeatsPipeline,
    "beats": BeatsPipeline,
    "onsets_rnn": OnsetRNNPipeline,
    "onsets_cnn": OnsetCNNPipeline,
    "key": KeyPipeline,
    "chroma": ChromaPipeline,
    "chords_feat": ChordFeaturePipeline,
    "notes_rnn": NoteRNNPipeline,
    "notes_cnn": NoteCNNPipeline,
}


def build_pipeline(name, **kwargs):
    """Construct a pipeline `nn.Module` by name -- one of `"downbeats"`,
    `"beats"`, `"onsets_rnn"`, `"onsets_cnn"`, `"key"`, `"chroma"`,
    `"chords_feat"`, `"notes_rnn"`, `"notes_cnn"` -- downloading (and
    sha256-verifying) its pretrained weights via `madmom_infer.models` on
    first use, same as the numpy processor it mirrors.
    """
    try:
        cls = _PIPELINES[name]
    except KeyError as exc:
        raise ValueError(
            f"build_pipeline: unknown pipeline {name!r}, expected one of "
            f"{sorted(_PIPELINES)}."
        ) from exc
    return cls(**kwargs)
