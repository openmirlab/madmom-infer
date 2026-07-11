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

## Scope

This project targets madmom's **inference** code only:

- Signal processing and feature extraction (framing, STFT, filterbanks,
  log-spectrograms)
- Decoding algorithms (Viterbi-based HMM/DBN beat and downbeat tracking)
- Onset/tempo/chord/key/note feature extraction (Phase 2, see below)

Out of scope, forever:

- **`madmom.evaluation.*`** -- madmom's F-measure/precision-recall research
  evaluation metrics (~4447 lines). This is tooling for *scoring* MIR research
  output, not for inference, and is not part of this project's scope under any
  phase.
- Training-only code. Madmom itself has essentially no gradient-based training
  code to port (its neural-net layers are forward-inference-only already).

## Dual backend design

- **numpy backend** (default, required): the reference implementation.
  Verified bit-identical to original madmom via **golden-fixture tests** --
  recordings of real madmom output, checked against this port's output. This
  is the same testing philosophy used by the sibling
  [all-in-one-infer](https://github.com/openmirlab/all-in-one-infer) package's
  pure-Python NATTEN replacement.
- **torch backend** (optional, via the `torch` extra): GPU-accelerated batch
  processing. Most valuable for the spectrogram/STFT stage, which batches
  trivially across frames. The Viterbi decoder is an inherently sequential,
  per-frame recursion, so GPU gains there are expected to be **marginal** --
  we're not going to oversell that in these docs.

Install the optional torch backend with:

```bash
pip install "madmom-infer[torch]"
```

## Roadmap

### Phase 1 -- **complete**

Driven by the sibling all-in-one-infer package's needs. All of the original
madmom code in this phase is pure numpy/scipy (no Cython), so most of it was a
near-mechanical port, verified against madmom via golden fixtures:

- `Signal`, `FramedSignalProcessor` (`madmom_infer/audio/signal.py`)
- `ShortTimeFourierTransformProcessor` (`madmom_infer/audio/stft.py`)
- Filterbank construction (`madmom_infer/audio/filters.py`)
- `FilteredSpectrogramProcessor`, `LogarithmicSpectrogramProcessor`
  (`madmom_infer/audio/spectrogram.py`)
- `Processor` / `SequentialProcessor` composition (`madmom_infer/processors.py`)
- A **numpy reimplementation of madmom's Cython Viterbi decoder**
  (`madmom/ml/hmm.pyx` -> `HiddenMarkovModel.viterbi()`), the phase-1
  centerpiece (`madmom_infer/ml/hmm.py`), plus the bar-length state-space and
  DBN downbeat tracker that consume it (`madmom_infer/features/beats_hmm.py`,
  `madmom_infer/features/downbeats.py`).

  This is feasible in pure numpy because the beat/downbeat state space is
  small (~11k-15k states per bar-length HMM), transitions are sparse (~1-2
  incoming edges per state), and the recursion is log-domain -- well within
  numpy vectorization territory, no Cython/C/GPU required.

Every stage above has a golden-fixture test (`tests/test_*.py`, 84 tests
total) proving bit-identical output against a real, compiled madmom 0.17.dev0
install -- except the one stage where bit-identity is bounded by BLAS library
non-associativity rather than achievable in principle: the filterbank
matrix-multiply (`np.dot(spectrogram, filterbank)`) rounds differently by a
handful of float32 ULPs depending on which OpenBLAS build numpy resolves to.
This is proven, not assumed -- `tests/test_spectrogram.py`'s
`test_filtered_spectrogram_algorithm_is_exact_under_original_blas` exports
this port's own computed arrays and re-runs the same matmul through the
original reference venv's numpy/BLAS build, reproducing the golden fixture
with zero differing elements. Everything downstream of the STFT and
filterbank-matrix stages (which ARE bit-identical) is verified to within 64
ULPs (worst case observed: 12, on the committed fixtures).

**End-to-end acceptance**: run against the sibling all-in-one-infer package
(3 stems separated once, shared between two isolated environments -- one with
real madmom, one with madmom-infer via a thin `import madmom` compatibility
shim) on a 90-second real-music excerpt, `harmonix-all` model, CUDA: **bpm,
beats, downbeats, beat_positions, and segments (boundaries + labels) came out
byte-for-byte IDENTICAL**. The intermediate spectrogram `.npy` arrays differed
by up to ~1778 float32 ULPs (max abs diff ~2.4e-7) on this longer, more
complex real track -- the same proven BLAS-non-associativity source as
above, confirmed to have zero effect on the final decoded output. A
same-environment rerun (real madmom vs. itself) reproduced both the fields
*and* the spectrogram bit-for-bit, confirming the pipeline itself is fully
deterministic and isolating the spectrogram delta to the BLAS difference
between environments.

### Phase 2 -- **NN runtime + `RNNDownBeatProcessor` end-to-end: complete**

The weights-bundling question is resolved as: **never bundle, always
download at runtime** (see "What this project will NEVER bundle" below).
Phase 2 ships:

- **A forward-pass-only NN runtime** (`madmom_infer/ml/nn/{__init__,layers,
  activations}.py`), porting exactly the layer/activation types the target
  ensemble needs: `NeuralNetwork`/`NeuralNetworkEnsemble`,
  `FeedForwardLayer`, `RecurrentLayer`, `BidirectionalLayer`, `Gate`/`Cell`/
  `LSTMLayer`, and the `linear`/`tanh`/`sigmoid`/`relu`/`elu`/`softmax`
  activations. Everything in `madmom/ml/nn/*` is already forward-inference-
  only (no `backward`/`train`/`fit`/`grad` anywhere, confirmed by grep) and
  pure Python/NumPy (no Cython), so this was a near-mechanical port, same as
  Phase 1's spectrogram chain.
- **A restricted, class-allowlisted unpickler** (`madmom_infer/ml/nn/
  unpickle.py`) for madmom's own `.pkl` model files -- deliberately NOT a
  bare `pickle.load`, since unpickling is inherently code execution and a
  downloaded model file is a lower-trust artifact than this project's own
  source. Only the exact class/function paths the target models reference
  are allowed (see `unpickle.py`'s mapping table); anything else raises
  loudly.
- **A runtime weights-download layer** (`madmom_infer/models.py`): fetches
  madmom's `DOWNBEATS_BLSTM` ensemble (8 `downbeats_blstm_[1-8].pkl` files)
  from the official `CPJKU/madmom_models` GitHub repository over HTTPS,
  caches them under `$XDG_CACHE_HOME/madmom_infer/models/` (never inside
  this project), and verifies each download's sha256 against a pinned
  known-good table before use.
- **`RNNDownBeatProcessor`** (`madmom_infer/features/downbeats.py`): the
  multi-frame-size (1024/2048/4096) spectrogram + `SpectrogramDifference`
  pre-processing cascade, feeding an 8-network BLSTM ensemble, chained into
  the already-ported `DBNDownBeatTrackingProcessor` -- audio in, beat/
  downbeat times out, matching real madmom exactly (see below).

**Verification**: unpickled-model structural digest (layer types, shapes,
every weight/bias/recurrent/peephole-weight array's sha256, activation
function names) matches real madmom's own unpickling **exactly**, across
all 8 ensemble networks (`tests/test_ml_nn.py`). Activations match to within
a documented ULP bound (up to 190 ULP observed, compounding Phase 1's
proven BLAS non-associativity across dozens of `np.dot` calls per ensemble
member) -- proven algorithm-exact, not just "close", by the same technique
Phase 1 established: this project's own code, re-run under the original
reference venv's numpy/BLAS build, reproduces real madmom's recorded
activations AND decoded beat/downbeat times with **zero** differing
elements (`tests/test_downbeats_rnn.py::test_full_pipeline_is_exact_under_original_blas`).
Decoded beat/downbeat times are **exact** in every environment tested
(the DBN decode is an integer-domain argmax, which absorbs float32-ULP-scale
input noise). Onset/tempo/chord/key/note feature extraction beyond
`RNNDownBeatProcessor` remains out of scope for now (see roadmap below).

### Phase 3 (odds and ends)

Remaining audio submodules (chroma, HPSS, cepstrogram) and two more small
Cython units: `features/beats_crf.pyx` and `audio/comb_filters.pyx`.

## What this project will NEVER bundle

madmom's own pretrained model weights (`.pkl` and similar files) are licensed
**CC BY-NC-SA 4.0 (non-commercial)** by the original authors -- a separate,
more restrictive license than madmom's BSD-2-Clause source code. **This is a
permanent policy, not a phase-2-only caveat**: madmom-infer will never bundle,
vendor, or redistribute any of madmom's own pretrained weights, in any phase,
for any reason. See [NOTICE](./NOTICE) for the full statement.

Instead (Phase 2, `madmom_infer/models.py`), weights are downloaded at
**runtime**, on demand, directly from the official
[CPJKU/madmom_models](https://github.com/CPJKU/madmom_models) GitHub
repository, cached locally under `$XDG_CACHE_HOME/madmom_infer/models/`
(never inside this project's own package or git history), with sha256
verification against a pinned known-good table. **The downloaded weight
bytes remain CC BY-NC-SA 4.0 -- non-commercial use only -- regardless of
madmom-infer's own BSD-2-Clause license**, which covers only this project's
source code, never the weights it fetches at runtime. See
[NOTICE](./NOTICE) and `madmom_infer/models.py`'s module docstring for the
full statement.

## Install

```bash
pip install madmom-infer
```

Phase 1 -- the DSP feature-extraction pipeline and numpy Viterbi/DBN
downbeat decoder -- and Phase 2 -- the NN runtime, restricted model
unpickling, runtime weights download, and `RNNDownBeatProcessor`
end-to-end -- are complete and golden-fixture verified; Phase 3 is not yet
started. See the Roadmap above.

## Attribution

madmom-infer reimplements algorithms originally published by CPJKU/madmom:

> https://github.com/CPJKU/madmom

See [LICENSE](./LICENSE) and [NOTICE](./NOTICE) for full attribution and
licensing details.

## Development

This project uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python -c "import madmom_infer; print(madmom_infer.__version__)"
uv run pytest -v
```

The default `pytest` run above is fully offline (99 tests, network-marked
tests deselected by `pyproject.toml`; this is what CI runs). To also
exercise the network-dependent A/B tests against real, freshly-downloaded
madmom weights, run `uv run pytest -m network -v`. See CLAUDE.md's
"Phase-2 verification commands" for the full picture, including the
reference-venv cross-BLAS proof.

## License

BSD-2-Clause. See [LICENSE](./LICENSE). Note: this covers madmom-infer's
source code only -- see "What this project will NEVER bundle" above regarding
madmom's separately-licensed pretrained weights.
