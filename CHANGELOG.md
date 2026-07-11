# Changelog

All notable changes to madmom-infer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
