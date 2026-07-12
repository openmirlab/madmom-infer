# Changelog

All notable changes to madmom-infer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-12

**Phase 3a: optional, differentiable torch spectrogram frontend.** The
numpy backend stays the default and is unchanged -- zero modifications to
any existing `madmom_infer/` numpy module in this release, verified by
running the full pre-existing suite unchanged before and after (68 passed,
20 skipped, 11 deselected both times).

### Added
- `madmom_infer/torch/` (opt-in, `pip install "madmom-infer[torch]"`):
  a batched, autograd-differentiable, device-agnostic torch reimplementation
  of the framing -> STFT -> filterbank -> log-compression -> temporal-
  difference chain (`madmom_infer.audio.{signal,stft,filters,spectrogram}`'s
  tensor operations), plus a `rnn_downbeat_frontend()` factory mirroring
  `RNNDownBeatProcessor`'s 3-branch (1024/2048/4096-sample,
  3/6/12-bands-per-octave) pre-processing cascade exactly, producing the
  same 314-dimensional feature vector real madmom's RNN ensemble consumes.
  Reuses the numpy backend's own window (`np.hanning`), filterbank matrix
  (`LogarithmicFilterbank`), bin frequencies (`fft_frequencies`), and
  diff-frame count (`_diff_frames`) -- computed via the existing numpy code
  and converted to tensors, not re-derived. `import madmom_infer` never
  imports torch; `import madmom_infer.torch` raises a clear, actionable
  `ImportError` if torch is not installed.
  - Explicitly NOT covered (see module docstrings, README "Torch frontend
    (Phase 3a)"): the RNN ensemble forward pass (madmom's LSTM peephole
    connections have no `torch.nn.LSTM` equivalent -- Phase 3b, not
    started) and Viterbi/DBN decoding (sequential, discrete-state -- no
    torch benefit expected, not planned).
  - 50 new tests (`tests/test_torch_frontend.py`), `pytest.importorskip
    ("torch")`-gated so they skip cleanly on a torch-less install:
    framing-index correctness against `FramedSignal.__getitem__` directly;
    float32 parity against the real, shipped numpy processor chain on
    synthetic sine-mix/noise/click/silence signals across 4 signal lengths
    (including non-multiples of the 441-sample hop) -- max absolute
    difference ~2.3e-6, tolerance atol-dominated rather than ULP-based
    (the temporal-difference stage's zero-clamping makes relative/ULP
    metrics blow up on tiny absolute differences, a measurement artifact,
    not a bigger error -- see that file's module docstring); float64
    parity against a bespoke float64-throughout numpy test harness (numpy's
    shipped classes hardcode a `complex64`/`float32` ceiling regardless of
    input dtype and cannot produce a genuine float64 baseline) -- within
    ~1e-10; `torch.autograd.gradcheck`-verified differentiability on a
    small float64 instance; batched output matches per-item output to
    float32 matmul/FFT batch-shape non-associativity tolerance; CPU and
    (parametrized, skipped if unavailable) CUDA device tests; dtype/shape
    mismatch error-handling tests.
- Wheel-from-sdist install smoke test (org constitution art. 7): built via
  `python -m build`, installed into a clean venv without torch (`import
  madmom_infer` works, public symbol touch succeeds) and into a second
  clean venv with the `torch` extra (`import madmom_infer.torch` works,
  `rnn_downbeat_frontend()` runs).

## [0.1.1] - 2026-07-12

### Fixed
- **Doc-honesty fix: removed all "torch backend" present-tense claims.** There
  is zero `import torch` anywhere in `madmom_infer/` -- the torch backend was
  always a Phase 3 proposal (`docs/DESIGN.md`), never implemented, but
  shipped docs described it as an existing, installable feature. Corrected:
  - `README.md`'s "Dual backend design" section claimed a "torch backend
    (optional, via the `torch` extra): GPU-accelerated batch..." and
    instructed `pip install "madmom-infer[torch]"`. Rewritten as "Backend"
    describing numpy as the only backend that exists today, torch as planned
    Phase 3 (with a pointer to `docs/DESIGN.md`).
  - `pyproject.toml` declared a `torch = ["torch>=2.0.0"]` optional-dependency
    extra that nothing in the package imports or uses -- removed. Installing
    it would have pulled a multi-gigabyte unused dependency.
  - `CLAUDE.md`'s "Dual-backend + golden-fixture testing philosophy" section
    asserted "An optional torch backend exists for GPU-accelerated batch
    processing, gated behind the `torch` extra" -- rewritten to state plainly
    that no torch backend exists yet and it's a planned, not-yet-implemented
    Phase 3 item.
  - `madmom_infer/__init__.py`'s module docstring described "current
    numpy/scipy/torch" and a "dual numpy/torch backend design" -- corrected
    to numpy/scipy only, with torch noted as planned but unimplemented.
  - `docs/DESIGN.md` and `madmom_infer/audio/signal.py`'s module docstring
    were left as-is: both already use honest proposal/roadmap language
    ("we recommend", "a future `madmom_infer.torch.audio.signal.Signal`")
    rather than claiming the torch backend currently exists.
  - The `[0.1.0]` entry below is left unedited as a historical record of what
    that release's docs said at the time, imperfections included.

### Changed
- **CI test matrix now covers the full classifiers-claimed range, Python
  3.9-3.13** (`.github/workflows/publish.yml`'s `test` job, which gates the
  `publish` job) -- previously only 3.9-3.11. Added the
  `Programming Language :: Python :: 3.13` classifier to `pyproject.toml` to
  match. All five versions verified green locally via fresh
  `uv venv --python 3.X` + editable install + `pytest`. 3.9-3.11 resolve
  scipy 1.17.1 by default: 68 passed, 20 skipped (env-guarded golden
  fixtures needing the real-madmom reference venv or network access,
  expected outside the recording environment), 11 deselected
  (`network`-marked), 0 failed. 3.12-3.13 resolve scipy 1.18.0 by default:
  66 passed, 22 skipped, 11 deselected, 0 failed -- 2 extra skips relative
  to 3.9-3.11, explained below.
- **scipy 1.18.0 compatibility: ULP-tolerant STFT fixture assertions**
  (`tests/test_stft.py`). scipy 1.18.0 (the first scipy release requiring
  Python>=3.12, and what `uv sync`/`pip install` resolve to by default on
  3.12/3.13) changed `scipy.fft.fft`'s float32 rounding relative to scipy
  1.13.1-1.17.1 (all bit-identical to the golden fixtures), previously
  causing 3 test failures under default dependency resolution on
  Python 3.12/3.13 (`test_stft_matches_fixture[float32_44100]`,
  `[stereo_48000_mono]`, and
  `test_window_caching_gotcha_reproduces_exact_bug`). Measured (not
  assumed): diffing this port's own STFT output computed under scipy
  1.17.1 (bit-identical to `tests/fixtures/stft.npz`, including its
  whole-array SHA-256 fingerprints) against the same computation under
  scipy 1.18.0 shows the two builds differ by exactly 1 float32 ULP in
  exactly 1 of 153,600 values per affected case, always landing outside the
  three frames (`frame0`/`frame1`/`frame_last`) the fixture stores
  per-element ground truth for. Fixed by relaxing exactly the 2 assertions
  that needed it, no more: `test_window_caching_gotcha_reproduces_exact_bug`
  now uses `np.testing.assert_array_max_ulp(maxulp=4)` (4x the measured
  1-ULP worst case, rounded up to the next power of two -- the same margin
  convention `test_spectrogram.py` already established for this exact class
  of build-dependent float32 non-associativity) instead of bit-exact
  `array_equal`; `test_stft_matches_fixture`'s whole-array SHA-256 check
  (which, being a hash, has no notion of "close") now skips with an
  explicit reason on the 2 affected cases specifically when it doesn't
  match, rather than failing or silently passing -- frame0/frame1/frame_last
  stay bit-exact assertions in every case, unrelaxed, since they still pass
  on both scipy builds. No fixtures were regenerated and no dependency was
  pinned; this project's golden-fixture philosophy (CLAUDE.md) still holds:
  bit-exactness now holds within one scipy build, documented plainly rather
  than papered over, the same class as the org constitution's art.2
  env-scoped-fixture clause.
- **CI-hardware ULP tolerance for HMM transition-model fixtures**
  (`tests/test_beats_hmm.py`). GitHub Actions' py3.11 test job (run
  29173379978) failed `test_bar_transition_model_csr_exact[3]` and `[4]`,
  passing bit-exact on the local dev machine. Root cause: `BarTransitionModel`
  computes its `probabilities` array at runtime via
  `exponential_transition`'s `np.exp()` (`madmom_infer/features/beats_hmm.py`),
  and libm's `exp` last bit isn't guaranteed identical across CPU/OS/libc
  builds -- CI differed from the fixture by exactly 1 float64 ULP (max abs
  diff 1.11e-16, max rel diff 3.5e-16) in 536/21648 elements. Same class of
  env-dependence as the scipy 1.18 STFT fix above and the org constitution's
  art.2 clause. Fixed by relaxing exactly that one assertion to
  `np.testing.assert_array_max_ulp(maxulp=4)` (4x the measured 1-ULP worst
  case, matching this project's established margin convention); every other
  assertion in the file (CSR `states`/`pointers`, state positions/intervals,
  observation-model pointers -- all integer or non-transcendental) audited
  and left bit-exact, since none of them depend on libm. Full suite
  reverified green on a fresh py3.11 venv: 68 passed, 20 skipped, 11
  deselected, 0 failed.

## [0.1.0] - 2026-07-11

Initial public release. Phase 1 (spectrogram/STFT/filterbank chain plus the
HMM/Viterbi decoder and DBN downbeat-tracking state space) and Phase 2 (NN
runtime, restricted model unpickling, sha256-verified runtime download of
madmom's own pretrained weights, and `RNNDownBeatProcessor` end-to-end) are
both complete and bit-identical/BLAS-proven against a real madmom install,
per this project's golden-fixture testing philosophy (see CLAUDE.md). 99
tests, all green.

### Changed
- Release-prep: added a `pytest.mark.network` marker (deselected by default
  via `pyproject.toml`'s `addopts`, matching the sibling maest-infer repo's
  convention) for the tests that download real madmom pretrained weights,
  and moved that download out of module-import-time code into a fixture so
  collecting the test suite never touches the network regardless of marker
  selection. Added `.github/workflows/publish.yml` (trusted-publishing
  release pipeline, gated on the test suite passing first).

### Added
- **Phase 2: NN runtime + restricted model unpickling + runtime weights
  download + `RNNDownBeatProcessor` end-to-end**, proving one real madmom
  model (`DOWNBEATS_BLSTM`, an 8-network BLSTM ensemble) end-to-end:
  - `madmom_infer/ml/nn/{__init__,layers,activations}.py`: forward-pass-only
    NN runtime port -- `NeuralNetwork`/`NeuralNetworkEnsemble`,
    `FeedForwardLayer`/`RecurrentLayer`/`BidirectionalLayer`/`Gate`/`Cell`/
    `LSTMLayer`, and `linear`/`tanh`/`sigmoid`/`relu`/`elu`/`softmax` --
    exactly the layer/activation subset the target ensemble needs, ported
    from `madmom.ml.nn.*` (already forward-inference-only, no Cython).
  - `madmom_infer/ml/nn/unpickle.py`: a restricted, class-allowlisted
    `SafeUnpickler` for madmom's own `.pkl` model files (never a bare
    `pickle.load`), with a full `(madmom module, name) -> madmom_infer
    object` mapping table derived by `pickletools`-inspecting all 8 target
    pickles (not guessed from source).
  - `madmom_infer/models.py`: runtime weights-download layer for madmom's
    `DOWNBEATS_BLSTM` ensemble from the official `CPJKU/madmom_models`
    GitHub repo, with an XDG-respecting local cache and sha256 verification
    against a pinned known-good table (cross-checked against a real
    pip-installed madmom 0.17.dev0 wheel's vendored copy -- identical).
  - `madmom_infer/processors.py`: added `ParallelProcessor` (sequential,
    not `multiprocessing.Pool`-backed -- see module header) and
    `BufferProcessor`.
  - `madmom_infer/audio/signal.py`: added `SignalProcessor`.
  - `madmom_infer/audio/spectrogram.py`: added `SpectrogramDifference`/
    `SpectrogramDifferenceProcessor` (the SuperFlux-style temporal
    difference `RNNDownBeatProcessor` stacks onto each frame-size branch),
    skipped in Phase 1.
  - `madmom_infer/features/downbeats.py`: added `RNNDownBeatProcessor`,
    chaining the multi-frame-size (1024/2048/4096) spectrogram+diff cascade
    into the 8-network BLSTM ensemble into the already-ported
    `DBNDownBeatTrackingProcessor`.
  - 13 new tests (`test_ml_nn.py`, `test_downbeats_rnn.py`), all green (see
    this release's top-level summary for the final 99-test count).
    Unpickled-model structural digest
    (layer types/shapes/every weight-array sha256/activation names) matches
    real madmom's own unpickling EXACTLY across all 8 ensemble networks.
    Activations match real madmom to within a documented, empirically
    measured ULP bound (up to 190 ULP, compounding Phase 1's proven BLAS
    non-associativity across dozens of `np.dot` calls per ensemble member)
    -- proven algorithm-exact (not just "close"), same technique as Phase
    1: this project's own code, re-run under the original reference venv's
    numpy/BLAS build, reproduces real madmom's recorded activations AND
    decoded beat/downbeat times with ZERO differing elements. Decoded
    beat/downbeat times are exact in every environment tested regardless of
    activation-level ULP drift.
  - Found and pinned a fourth caching gotcha (same shape as Phase 1's two
    documented ones): reusing one `RNNDownBeatProcessor` instance across
    differing-dtype input wavs silently keeps the first call's dtype-scaled
    STFT window / sample-rate-scoped filterbank, now visible one level up
    from where Phase 1 found it.
- Initial project scaffold: package layout (`madmom_infer/audio`, `ml`,
  `features`), `pyproject.toml` (hatchling, dynamic version via
  `__about__.py`), `LICENSE` (BSD-2-Clause, dual-copyright), `NOTICE`,
  `README.md` with the 3-phase roadmap and dual numpy/torch backend design,
  and stub modules for all Phase 1 targets.
- **Phase 1 spectrogram chain, completing Phase 1**:
  `ShortTimeFourierTransform(Processor)` (`madmom_infer/audio/stft.py`),
  `Filterbank`/`LogarithmicFilterbank` (`madmom_infer/audio/filters.py`), and
  `Spectrogram`/`FilteredSpectrogram(Processor)`/
  `LogarithmicSpectrogram(Processor)` (`madmom_infer/audio/spectrogram.py`).
  Faithfully reproduces three real madmom bugs, all confirmed empirically
  against a live madmom install (not just from reading source): (1) the
  int16-signal FFT-window-scaling convention, (2) `ShortTimeFourierTransformProcessor`'s
  window-caching gotcha (a reused instance silently keeps a stale,
  wrong-dtype-scaled window across differing-dtype calls), and (3) a
  previously-undocumented `FilteredSpectrogramProcessor` filterbank-caching
  gotcha found while porting (a reused instance silently keeps a stale,
  wrong-sample-rate filterbank across differing-sample-rate calls -- this is
  why `tests/fixtures/filterbank.npz`'s `filterbank_matrix_48000` is actually
  a copy of `filterbank_matrix_44100`). All three are pinned by dedicated
  regression tests, not just narratively documented.
- 22 new golden-fixture tests (`tests/test_stft.py`, `test_filters.py`,
  `test_spectrogram.py`; 84 tests total, 62 pre-existing + 22 new, all
  green). STFT and filterbank-matrix construction are bit-identical
  (`np.array_equal` + exact dtype). Stages depending on `np.dot` (BLAS
  matmul) are verified to within 64 float32 ULPs, root-caused (not assumed)
  to BLAS-library non-associativity across OpenBLAS builds via a dedicated
  cross-environment proof test that re-runs this port's own computed arrays
  through the original reference venv's numpy/BLAS and gets a zero-diff
  match.
- **End-to-end acceptance test** against the sibling all-in-one-infer
  package: isolated environments (one with real madmom, one with
  madmom-infer via a thin `import madmom` compatibility shim re-exporting
  madmom_infer 1:1) running the identical pipeline (shared, pre-separated
  stems) on a real 90-second music excerpt, `harmonix-all` model, CUDA.
  `bpm`/`beats`/`downbeats`/`beat_positions`/`segments` came out identical;
  intermediate spectrogram arrays differed by up to ~1778 float32 ULPs
  (same proven BLAS root cause), with zero effect on the decoded output. A
  same-environment rerun confirmed the pipeline itself is fully
  deterministic (bit-identical fields AND spectrogram), isolating the
  cross-environment spectrogram delta to the BLAS difference alone.
