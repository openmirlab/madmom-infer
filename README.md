# madmom-infer

**A from-scratch, modernized reimplementation of [madmom](https://github.com/CPJKU/madmom)'s inference-relevant algorithms**

[![PyPI](https://img.shields.io/pypi/v/madmom-infer.svg)](https://pypi.org/project/madmom-infer/)
[![License: BSD-2-Clause](https://img.shields.io/badge/License-BSD--2--Clause-blue.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Why this exists

[madmom](https://github.com/CPJKU/madmom) is a well-regarded MIR (Music Information
Retrieval) / audio-DSP research library out of CPJKU (Johannes Kepler University,
Linz) and OFAI (Vienna). Its algorithms -- spectrogram feature extraction, beat
and downbeat tracking, onset detection, tempo estimation, and more -- are still
widely used and cited. But madmom's PyPI release is roughly 8 years stale, ships
compiled Cython extensions, and is difficult or impossible to install cleanly on
modern Python / numpy / scipy versions.

**madmom-infer** re-derives madmom's inference-relevant algorithms from scratch
against current Python tooling. It is an independent reimplementation, not an
official fork -- see [NOTICE](./NOTICE). It does not reuse or redistribute any
of madmom's original source code; it reimplements the published algorithms.

---

## Acknowledgments

This project stands entirely on the research and engineering of the original
madmom team. madmom-infer reimplements their published algorithms from
scratch -- it does not exist without their work:

- **[madmom](https://github.com/CPJKU/madmom)** by the Department of
  Computational Perception, Johannes Kepler University (JKU), Linz, Austria,
  and the Austrian Research Institute for Artificial Intelligence (OFAI),
  Vienna, Austria -- the original library this project reimplements the
  inference-relevant algorithms of
- **Sebastian Böck, Filip Korzeniowski, Jan Schlüter, Florian Krebs, and
  Gerhard Widmer** -- authors of the algorithms and the paper this project's
  DSP, HMM/DBN decoding, and RNN ensemble code re-derive (see Citation below)
- **[CPJKU/madmom_models](https://github.com/CPJKU/madmom_models)** -- the
  official upstream repository this project downloads pretrained
  `RNNDownBeatProcessor`/`CNNKeyRecognitionProcessor` weights from at
  runtime (never bundled, see
  [What this project will NEVER bundle](#what-this-project-will-never-bundle))

See [NOTICE](./NOTICE) for the full attribution statement, including why this
is an independent reimplementation and not an official fork.

---

## Citation

If you use madmom-infer, please cite the original madmom paper whose
algorithms it reimplements:

```bibtex
@inproceedings{madmom,
   Title = {{madmom: a new Python Audio and Music Signal Processing Library}},
   Author = {B{\"o}ck, Sebastian and Korzeniowski, Filip and Schl{\"u}ter, Jan and Krebs, Florian and Widmer, Gerhard},
   Booktitle = {Proceedings of the 24th ACM International Conference on Multimedia},
   Month = {10},
   Year = {2016},
   Pages = {1174--1178},
   Address = {Amsterdam, The Netherlands},
   Doi = {10.1145/2964284.2973795}
}
```

---

## Features

- **Bit-identical numpy backend**: the reference implementation, verified against
  real (compiled) madmom output via golden-fixture tests -- not "close enough,"
  proven exact or exact-to-a-documented-ULP-bound where BLAS non-associativity
  is the only source of drift
- **Optional differentiable torch frontend**: a batched, autograd-differentiable,
  device-agnostic reimplementation of the framing -> STFT -> filterbank ->
  log-compression -> temporal-difference chain (`torch` extra)
- **Restricted, class-allowlisted unpickling** for madmom's own `.pkl` model
  files -- never a bare `pickle.load` against a downloaded, lower-trust artifact
- **Runtime-only weight downloads**, sha256-verified against a pinned known-good
  table, cached under `$XDG_CACHE_HOME/madmom_infer/models/` -- never bundled or
  vendored into this project (see [What this project will NEVER bundle](#what-this-project-will-never-bundle))

---

## Scope

This project targets madmom's **inference** code only:

- Signal processing and feature extraction (framing, STFT, filterbanks,
  log-spectrograms)
- Decoding algorithms (Viterbi-based HMM/DBN beat and downbeat tracking)
- The NN runtime and `RNNDownBeatProcessor` end-to-end (spectrogram frontend ->
  BLSTM ensemble -> DBN decode)
- The CNN runtime (convolution/max-pool/batch-norm/pad/global-average/stride
  layers) and `CNNKeyRecognitionProcessor` end-to-end (spectrogram frontend
  -> CNN -> 24-class major/minor key probabilities + decoded label)
- Onset detection: the full spectral-flux/phase-deviation/complex-domain DSP
  function family, `SpectralOnsetProcessor`, `RNNOnsetProcessor` (online and
  offline RNN ensembles), `CNNOnsetProcessor`, and `OnsetPeakPickingProcessor`
  end-to-end (spectrogram frontend -> onset activation function -> decoded
  onset times)
- Beat tracking: `RNNBeatProcessor` (online and offline RNN ensembles),
  `DBNBeatTrackingProcessor` (beat-only, reusing the same HMM machinery as
  downbeat tracking), `MultiModelSelectionProcessor` end-to-end (spectrogram
  frontend -> beat activation function -> decoded beat times),
  `BeatTrackingProcessor`/`BeatDetectionProcessor` (tempo-driven, look-
  aside/look-ahead beat alignment, no HMM), and `CRFBeatDetectionProcessor`
  (a numpy-ported Conditional-Random-Field Viterbi decode over several
  candidate tempo intervals)
- Tempo estimation: `TempoEstimationProcessor` with all 3 shipped histogram
  modes -- autocorrelation (`ACFTempoHistogramProcessor`), resonating comb
  filters (`CombFilterTempoHistogramProcessor`, backed by a numpy port of
  madmom's comb-filter Cython module), and DBN-based
  (`DBNTempoHistogramProcessor`, reusing `DBNBeatTrackingProcessor`)
- Chroma extraction: classic pitch-class-profile chroma (`PitchClassProfile`/
  `HarmonicPitchClassProfile`), a deep-neural-network chroma extractor
  (`DeepChromaProcessor`), and Compressed Log Pitch chroma
  (`CLPChroma`/`CLPChromaProcessor`, a pure-DSP, time-domain-filterbank-based
  chroma feature)
- Chord recognition: a numpy Conditional Random Field decoder
  (`ConditionalRandomField`) backing two full audio-in, chord-segments-out
  pipelines -- `DeepChromaProcessor` -> `DeepChromaChordRecognitionProcessor`,
  and `CNNChordFeatureProcessor` -> `CRFChordRecognitionProcessor`
- `RNNBarProcessor`: an alternative, GRU-based joint beat/downbeat model
  (beat-synchronous percussive + harmonic features, the harmonic branch built
  on `CLPChromaProcessor` above) as a full audio-in alternative to
  `RNNDownBeatProcessor`
- Piano note transcription: `RNNPianoNoteProcessor` (RNN onset activations,
  decoded by `NoteOnsetPeakPickingProcessor`/`NotePeakPickingProcessor`) and
  `CNNPianoNoteProcessor` (a multi-task CNN producing per-pitch note/onset/
  offset activations, decoded by `ADSRNoteTrackingProcessor`'s
  attack-decay-sustain-release HMM, reusing the same Viterbi decoder as beat/
  downbeat tracking)
- Rhythmic pattern tracking: `PatternTrackingProcessor` (a Gaussian-Mixture-
  Model observation model -- `ml/gmm.py`'s forward-inference-only numpy `GMM`
  -- combined with a multi-band spectral-flux frontend
  (`MultiBandSpectrogram`/`MultiBandSpectrogramProcessor`) to jointly track
  beats, downbeats, AND the rhythmic pattern/meter itself) and
  `DBNBarTrackingProcessor` (decode downbeats from pre-determined beat
  positions plus a downbeat activation function, e.g. `RNNBarProcessor`'s
  output, across several candidate bar lengths)
- MFCC/cepstral features: `Cepstrogram`/`CepstrogramProcessor` (a generic
  DCT-of-spectrogram transform) and `MFCC`/`MFCCProcessor` (the standard
  Mel-filter -> log -> DCT pipeline, reusing the Mel filterbank above)
- Harmonic/percussive source separation: `HarmonicPercussiveSourceSeparation`
  (alias `HPSS`) -- median-filter-based `slices()`/`masks()` (Fitzgerald
  2010)

Out of scope, forever:

- **`madmom.evaluation.*`** -- madmom's F-measure/precision-recall research
  evaluation metrics (~4447 lines). This is tooling for *scoring* MIR research
  output, not for inference, and is not part of this project's scope.
- Training-only code. Madmom itself has essentially no gradient-based training
  code to port (its neural-net layers are forward-inference-only already).

Every inference-relevant public class/function in upstream madmom's
`features/`, `audio/`, and `ml/` packages is now ported (or documented as
permanently excluded, see above) -- the numpy backend is feature-complete
against a real madmom 0.17.dev0 install's own surface. The only remaining
gap is a torch reimplementation of the RNN ensemble forward pass itself
(blocked on a real design question -- madmom's LSTM layers use peephole
connections `torch.nn.LSTM` does not implement, so this needs a custom
cell, not a drop-in swap). Viterbi/DBN decoding is sequential and
discrete-state, so it is not planned for a torch port -- no GPU benefit to
speak of.

---

## Install

```bash
pip install madmom-infer

# with the optional differentiable torch frontend
pip install "madmom-infer[torch]"
```

---

## Quick Start

The task-level API is the shortest route from audio to a musical result. It
accepts paths, NumPy arrays, and metadata-bearing `Signal` objects; always pass
`sample_rate` for arrays. Audio is converted to the rate required by the
pretrained pipelines automatically.

```python
import madmom_infer as mm

beats = mm.detect_beats("track.wav", sample_rate=44100)
tempo = mm.estimate_tempo("track.wav", sample_rate=44100)
key = mm.detect_key("track.wav", sample_rate=44100)

# Reuse loaded models and shared intermediate results for several tasks.
result = mm.Analyzer(tasks=["beats", "tempo", "key"])(
    "track.wav", sample_rate=44100
)
print(result.beats, result.tempo, result.key)
```

The complete task vocabulary is `onsets`, `beats`, `downbeats`, `tempo`,
`key`, `chords`, `notes`, `chroma`, `mfcc`, and `hpss`. The original
madmom-style processors below remain supported for custom models, decoders,
and pipeline composition.

```python
from madmom_infer.features.downbeats import (
    RNNDownBeatProcessor,
    DBNDownBeatTrackingProcessor,
)

rnn = RNNDownBeatProcessor()
activations = rnn("track.wav")

dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
beats = dbn(activations)  # (time, beat_number_in_bar) pairs
```

The `RNNDownBeatProcessor`'s BLSTM ensemble weights are fetched from the
official [CPJKU/madmom_models](https://github.com/CPJKU/madmom_models)
repository on first use, sha256-verified, and cached locally -- no separate
download step needed. See
[What this project will NEVER bundle](#what-this-project-will-never-bundle)
for the licensing terms that apply to those weights.

### Key detection

```python
from madmom_infer.features.key import (
    CNNKeyRecognitionProcessor,
    key_prediction_to_label,
)

key_proc = CNNKeyRecognitionProcessor()
prediction = key_proc("track.wav")  # (1, 24) major/minor key probabilities
key_prediction_to_label(prediction)  # e.g. "E major"
```

Same runtime-download/sha256/caching story as `RNNDownBeatProcessor` above,
via `madmom_infer/models.py`'s `key_cnn()`.

### Onset detection

```python
from madmom_infer.features.onsets import (
    CNNOnsetProcessor,
    OnsetPeakPickingProcessor,
)

onset_proc = CNNOnsetProcessor()
activations = onset_proc("track.wav")

peak_picking = OnsetPeakPickingProcessor(fps=100)
onset_times = peak_picking(activations)  # onset times [seconds]
```

`RNNOnsetProcessor()` (bidirectional ensemble) and `RNNOnsetProcessor(online=True)`
(a smaller, causal ensemble) are drop-in alternatives to `CNNOnsetProcessor`
above -- same `activations -> OnsetPeakPickingProcessor` decode step. For
direct access to the underlying pure-DSP onset detection functions (no
pretrained weights needed at all), see `SpectralOnsetProcessor`:

```python
from madmom_infer.features.onsets import SpectralOnsetProcessor

sodf = SpectralOnsetProcessor(onset_method="superflux")
activations = sodf("track.wav")
```

Same runtime-download/sha256/caching story as `RNNDownBeatProcessor`/
`CNNKeyRecognitionProcessor` above for `RNNOnsetProcessor`/`CNNOnsetProcessor`,
via `madmom_infer/models.py`'s `onsets_rnn()`/`onsets_brnn()`/`onsets_cnn()`.

### Beat tracking

```python
from madmom_infer.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor

beat_proc = RNNBeatProcessor()
activations = beat_proc("track.wav")

dbn = DBNBeatTrackingProcessor(fps=100)
beat_times = dbn(activations)  # beat times [seconds]
```

`RNNBeatProcessor(online=True)` is a smaller, causal ensemble, a drop-in
alternative to the bidirectional default above -- same
`activations -> DBNBeatTrackingProcessor` decode step. Same runtime-download/
sha256/caching story as the processors above, via `madmom_infer/models.py`'s
`beats_lstm()`/`beats_blstm()`.

`CRFBeatDetectionProcessor` and `BeatTrackingProcessor`/
`BeatDetectionProcessor` are drop-in alternatives to `DBNBeatTrackingProcessor`
above -- same `activations -> beat_times` decode step, different decoding
algorithm (a Conditional-Random-Field Viterbi decode over several candidate
tempo intervals, or a non-HMM tempo-driven look-aside/look-ahead alignment,
respectively):

```python
from madmom_infer.features.beats import CRFBeatDetectionProcessor

crf = CRFBeatDetectionProcessor(fps=100)
beat_times = crf(activations)
```

### Pattern tracking

```python
from madmom_infer.audio.signal import SignalProcessor
from madmom_infer.audio.spectrogram import (
    LogarithmicSpectrogramProcessor,
    MultiBandSpectrogramProcessor,
    SpectrogramDifferenceProcessor,
)
from madmom_infer.features.downbeats import PatternTrackingProcessor
from madmom_infer.models import patterns_ballroom
from madmom_infer.processors import SequentialProcessor

pre_proc = SequentialProcessor([
    SignalProcessor(num_channels=1, sample_rate=44100),
    LogarithmicSpectrogramProcessor(),
    SpectrogramDifferenceProcessor(positive_diffs=True),
    MultiBandSpectrogramProcessor(crossover_frequencies=[270]),
])
features = pre_proc("track.wav")

pattern_proc = PatternTrackingProcessor(patterns_ballroom(), fps=50)
beats = pattern_proc(features)  # (time, beat_number_in_bar) pairs
```

Jointly tracks beats, downbeats, AND the rhythmic pattern/meter itself, via a
Gaussian-Mixture-Model observation model (`madmom_infer.ml.gmm.GMM`,
forward-inference-only) rather than an RNN. `patterns_ballroom()` downloads
(and sha256-verifies) madmom's `PATTERNS_BALLROOM` GMM files -- NOT neural
network weights, but still CC BY-NC-SA-licensed, same
runtime-download/caching story as every other model family above (see
[What this project will NEVER bundle](#what-this-project-will-never-bundle)).

### Tempo estimation

```python
from madmom_infer.features.beats import RNNBeatProcessor
from madmom_infer.features.tempo import TempoEstimationProcessor

beat_proc = RNNBeatProcessor()
activations = beat_proc("track.wav")

tempo_proc = TempoEstimationProcessor(fps=100)  # comb-filter histogram, default
tempi = tempo_proc(activations)  # [[bpm, strength], ...], strongest first
```

Pass `method="acf"` or `method="dbn"` for the autocorrelation or DBN-based
histogram modes instead of the default resonating-comb-filter one.

### Chroma extraction

```python
from madmom_infer.audio.chroma import DeepChromaProcessor

chroma_proc = DeepChromaProcessor()
chroma = chroma_proc("track.wav")  # (num_frames, 12) chroma vectors
```

Two more chroma flavors are available: `CLPChromaProcessor()` (pure DSP, no
pretrained weights -- Compressed Log Pitch chroma, needs `ffmpeg` on `PATH`
for its internal semitone-filterbank resampling), and
`PitchClassProfile`/`HarmonicPitchClassProfile` (hand-designed filterbank
weighting on top of a plain `Spectrogram`, no pretrained weights either):

```python
from madmom_infer.audio.chroma import CLPChromaProcessor
from madmom_infer.audio.spectrogram import Spectrogram
from madmom_infer.audio.chroma import PitchClassProfile

clp_proc = CLPChromaProcessor(fps=50)
clp_chroma = clp_proc("track.wav")

spec = Spectrogram("track.wav", frame_size=2048, fps=100)
pcp = PitchClassProfile(spec)
```

Same runtime-download/sha256/caching story as the processors above for
`DeepChromaProcessor`, via `madmom_infer/models.py`'s `chroma_dnn()`.

### Chord recognition

```python
from madmom_infer.audio.chroma import DeepChromaProcessor
from madmom_infer.features.chords import DeepChromaChordRecognitionProcessor
from madmom_infer.processors import SequentialProcessor

chroma_proc = DeepChromaProcessor()
decode = DeepChromaChordRecognitionProcessor()
chord_rec = SequentialProcessor([chroma_proc, decode])

chords = chord_rec("track.wav")
# structured array of (start, end, label) segments, e.g.:
# [(0.0, 1.6, 'F:maj'), (1.6, 2.5, 'A:maj'), (2.5, 4.1, 'D:maj')]
```

`CNNChordFeatureProcessor` -> `CRFChordRecognitionProcessor` is a drop-in
alternative chain (a learned CNN feature extractor instead of deep chroma,
decoded by a separately-trained CRF model). Same runtime-download/sha256/
caching story as the processors above, via `madmom_infer/models.py`'s
`chords_dccrf()`/`chords_cnn_feat()`/`chords_cfcrf()`.

### Piano note transcription

```python
from madmom_infer.features.notes import (
    CNNPianoNoteProcessor,
    ADSRNoteTrackingProcessor,
)

cnn_proc = CNNPianoNoteProcessor()
activations = cnn_proc("track.wav")  # (num_frames, 88, 3): note/onset/offset

adsr = ADSRNoteTrackingProcessor()
notes = adsr(activations)  # (time, MIDI_pitch, duration) triples
```

An RNN alternative is also available -- `RNNPianoNoteProcessor()` produces a
`(num_frames, 88)` onset activation function (one column per piano key,
MIDI note 21..108), decoded by peak-picking instead of an HMM:

```python
from madmom_infer.features.notes import (
    RNNPianoNoteProcessor,
    NoteOnsetPeakPickingProcessor,
)

rnn_proc = RNNPianoNoteProcessor()
activations = rnn_proc("track.wav")

peak_pick = NoteOnsetPeakPickingProcessor(fps=100, pitch_offset=21)
onsets = peak_pick(activations)  # (time, MIDI_pitch) pairs
```

Same runtime-download/sha256/caching story as the processors above, via
`madmom_infer/models.py`'s `notes_brnn()`/`notes_cnn()`.

### MFCC / cepstral features

Pure DSP, no pretrained weights:

```python
from madmom_infer.audio.spectrogram import Spectrogram
from madmom_infer.audio.cepstrogram import MFCC, Cepstrogram

mfcc = MFCC("track.wav", sample_rate=44100)  # (num_frames, 30) MFCCs

spec = Spectrogram("track.wav")
cepstrogram = Cepstrogram(spec)  # DCT of the plain magnitude spectrogram
```

### Harmonic/percussive source separation

```python
import numpy as np
from madmom_infer.audio.spectrogram import Spectrogram
from madmom_infer.audio.hpss import HPSS

spec = Spectrogram("track.wav")
hpss = HPSS()  # alias for HarmonicPercussiveSourceSeparation
harmonic_slice, percussive_slice = hpss.slices(np.asarray(spec))
harmonic_mask, percussive_mask = hpss.masks(harmonic_slice, percussive_slice)
harmonic = np.asarray(spec) * harmonic_mask
percussive = np.asarray(spec) * percussive_mask
```

`HPSS().process()` composes the same `slices()` and `masks()` operations and
returns the harmonic and percussive spectrograms directly.

### Torch frontend

```python
import torch
from madmom_infer.torch import rnn_downbeat_frontend

# mono, float, 44.1kHz waveform: (batch, num_samples)
waveform = torch.randn(2, 44100)

frontend = rnn_downbeat_frontend(dtype=torch.float32)  # matches
                                                        # RNNDownBeatProcessor's
                                                        # 3-branch DSP cascade
features = frontend(waveform)  # (2, 100, 314), differentiable w.r.t. waveform
```

Because it is a plain differentiable `nn.Module`, the frontend can sit
inside a training loop -- e.g. learning a per-band gain on top of the fixed
filterbank (a toy illustration, not a claim that madmom's own filterbank
should be retrained):

```python
import torch
from madmom_infer.torch.audio.frontend import SpectrogramFrontend

frontend = SpectrogramFrontend(frame_size=2048, fps=100, num_bands=12,
                                dtype=torch.float32)
band_gain = torch.nn.Parameter(torch.ones(2 * frontend.num_bands))
optimizer = torch.optim.Adam([band_gain], lr=1e-2)

waveform = torch.randn(4, 44100)
target = torch.zeros(4, 100, 2 * frontend.num_bands)

for _ in range(50):
    optimizer.zero_grad()
    feats = frontend(waveform) * band_gain  # gradient flows through the STFT
    loss = torch.nn.functional.mse_loss(feats, target)
    loss.backward()
    optimizer.step()
```

**What it covers** (`madmom_infer/torch/audio/frontend.py`): framing (exact
`FramedSignal` hop/origin semantics), complex STFT (window, FFT size,
optional circular shift), filterbank application, log compression, and the
temporal difference stage, individually as functional building blocks
(`frame_signal`, `stft`, `apply_filterbank`, `log_compress`,
`temporal_difference`) plus the composed `SpectrogramFrontend` module and
the `rnn_downbeat_frontend()` factory that mirrors `RNNDownBeatProcessor`'s
3-branch (1024/2048/4096-sample, 3/6/12-bands-per-octave) cascade exactly,
producing the same 314-dimensional feature vector real madmom's RNN
ensemble consumes.

**What it explicitly does NOT cover** (see `madmom_infer/torch/__init__.py`):
- The RNN ensemble forward pass -- madmom's LSTM layers use peephole
  connections `torch.nn.LSTM` does not implement, so this needs a custom cell.
- Viterbi/DBN decoding -- sequential, per-frame, discrete-state recursion, no
  batching or GPU benefit to speak of.
- Audio loading/downmixing/resampling (`SignalProcessor`) -- the frontend
  takes an already-mono, already-resampled float waveform tensor directly.
- Byte-identical numeric parity with the numpy backend at every precision:
  the two use different underlying FFT/BLAS libraries (`torch.fft` vs
  `scipy.fft`/BLAS), so outputs agree to a documented tolerance
  (`tests/test_torch_frontend.py`: ~2.3e-6 max absolute difference at
  float32, ~1e-10 at float64 against a float64-only numpy test harness --
  see that file's module docstring for why numpy's *shipped* classes cannot
  produce a genuine float64 baseline to begin with), not bit-for-bit.

---

## What this project will NEVER bundle

madmom's own pretrained model weights (`.pkl` and similar files) are licensed
**CC BY-NC-SA 4.0 (non-commercial)** by the original authors -- a separate,
more restrictive license than madmom's BSD-2-Clause source code. **This is a
permanent policy**: madmom-infer will never bundle, vendor, or redistribute
any of madmom's own pretrained weights, for any reason. See
[NOTICE](./NOTICE) for the full statement.

Instead (`madmom_infer/models.py`), weights are downloaded at **runtime**, on
demand, directly from the official
[CPJKU/madmom_models](https://github.com/CPJKU/madmom_models) GitHub
repository, cached locally under `$XDG_CACHE_HOME/madmom_infer/models/`
(never inside this project's own package or git history), with sha256
verification against a pinned known-good table. **The downloaded weight
bytes remain CC BY-NC-SA 4.0 -- non-commercial use only -- regardless of
madmom-infer's own BSD-2-Clause license**, which covers only this project's
source code, never the weights it fetches at runtime. See
[NOTICE](./NOTICE) and `madmom_infer/models.py`'s module docstring for the
full statement.

---

## Development

This project uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python -c "import madmom_infer; print(madmom_infer.__version__)"
uv run pytest -v
```

The default `pytest` run above is fully offline (network-marked tests
deselected by `pyproject.toml`; this is what CI runs). To also exercise the
network-dependent tests against real, freshly-downloaded madmom weights, run
`uv run pytest -m network -v`. To exercise the optional torch frontend, install
it first (`uv sync --extra dev --extra torch`), then run
`uv run pytest tests/test_torch_frontend.py -v`. See CLAUDE.md for the full
verification picture, including the reference-venv cross-BLAS proof.

---

## License

BSD-2-Clause. See [LICENSE](./LICENSE). Note: this covers madmom-infer's
source code only -- see
[What this project will NEVER bundle](#what-this-project-will-never-bundle)
above regarding madmom's separately-licensed pretrained weights.

---

## Support

For issues and questions:
- **GitHub Issues**: [github.com/openmirlab/madmom-infer/issues](https://github.com/openmirlab/madmom-infer/issues)

---
