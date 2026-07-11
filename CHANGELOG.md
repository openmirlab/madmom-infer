# Changelog

All notable changes to madmom-infer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  - 13 new tests (`test_ml_nn.py`, `test_downbeats_rnn.py`; 97 total, 84
    pre-existing + 13 new, all green). Unpickled-model structural digest
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
