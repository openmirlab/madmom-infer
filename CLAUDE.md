# madmom-infer -- CLAUDE.md

## Scope and status

madmom-infer is a from-scratch reimplementation of CPJKU/madmom's
inference-relevant algorithms (not training-only code, not
`madmom.evaluation.*`). See [README.md](./README.md) for the public-facing
scope and feature description -- it no longer uses phase framing, don't
reintroduce it there.

Internally, this file still tracks work by phase (matching test/script
names like `tools/generate_phase2_fixtures.py`), since that's a stable way
to refer to a specific chunk of already-shipped work:

- **Phase 1** (complete): DSP pipeline (framing, STFT, filterbanks,
  log-spectrograms) + numpy Viterbi decoder
- **Phase 2** (complete): forward-pass-only NN runtime + restricted model
  unpickling + runtime weights download + `RNNDownBeatProcessor`
  end-to-end -- the pretrained-weights question is resolved as "never
  bundle, always download at runtime", see `madmom_infer/models.py`
- **Phase 3a** (complete): optional, differentiable torch spectrogram
  frontend (`madmom_infer/torch/`)
- **Phase 3b** (not started): torch NN forward pass, blocked on madmom's
  LSTM peephole connections having no `torch.nn.LSTM` equivalent
- **Phase 4 — complete-port campaign** (started 2026-07-12, branch
  `feat/complete-port`): port every remaining inference-relevant madmom
  capability. Target surface = what the reference madmom install exposes
  (0.17.dev0, built from `../madmom-upstream`). Waves, each gated on
  golden fixtures + the full suite green before its commit:
  - **4.0 reference-env rebuild + gap audit**: the original reference venv
    (`all-in-one-fix/.venv`) no longer exists on this machine. Rebuild at
    `../madmom-reference/.venv` (Python 3.10, numpy==1.23.5,
    scipy==1.15.3 — matching the recorded environment of the committed
    fixtures), install madmom from `../madmom-upstream` (populate its
    `madmom/models` submodule first), repoint `REFERENCE_PYTHON` in
    tests/tools/docs, and prove faithfulness by running the
    previously-skipped cross-BLAS exactness tests against the already-
    committed fixtures — if they fail on BLAS-build differences, STOP and
    surface it; never silently regenerate Phase-1/2 fixtures. Then a
    class-level gap audit: every public processor/class in upstream
    `features/`, `audio/`, `ml/` → ported / to-port (which wave) /
    excluded (why), recorded below.
  - **4a CNN infra + key detection**: `ConvolutionalLayer`,
    `MaxPoolLayer`, `BatchNormLayer`, `PadLayer`, `AverageLayer` (pure
    numpy — the classes `key_cnn.pkl` actually pickles) + unpickler
    allowlist entries + models-registry entries (key/2018) +
    `CNNKeyRecognitionProcessor` end-to-end with `key_prediction_to_label`.
  - **4b onsets**: the spectral-flux DSP family (superflux, complex
    domain, high-frequency content, …), `RNNOnsetProcessor`,
    `CNNOnsetProcessor` (reuses 4a's conv layers),
    `OnsetPeakPickingProcessor`.
  - **4c beats completion + tempo**: `RNNBeatProcessor`, beat-only
    `DBNBeatTrackingProcessor`, `MultiModelSelectionProcessor`;
    `TempoEstimationProcessor` (acf/dbn/comb — comb via a numpy port of
    `audio/comb_filters.pyx`; TCN layers only if the reference surface
    actually ships a TCN model).
  - **4d chroma + chords**: `ml/crf.py` (numpy CRF Viterbi),
    `DeepChromaProcessor`, `CLPChroma`,
    `DeepChromaChordRecognitionProcessor`, `CNNChordFeatureProcessor` +
    `CRFClassifier` chord decoding.
  - **4e notes/piano**: `RNNPianoNoteProcessor`,
    `ADSRNoteTrackingProcessor` (`notes_hmm.py` state spaces on the
    existing HMM machinery), `NotePeakPickingProcessor`.
  - **4f CRF beats + patterns**: numpy port of `features/beats_crf.pyx`
    (same playbook as Phase 1's `hmm.pyx` port), `ml/gmm.py`,
    `GMMPatternTrackingProcessor`.
  - **4g leftovers + closure**: `audio/cepstrogram.py` (MFCC),
    `audio/hpss.py`, anything the 4.0 audit flags; closure audit (every
    upstream inference class → ported or documented-excluded); README/
    CHANGELOG sync; version bump; merge to main.
  - **Permanent exclusions** (existing ones unchanged, plus): `bin/` CLI
    programs (this package is a library — processors are the API) and
    layer classes no shipped model needs (no speculative TCN ports —
    **correction, 4.0 audit**: GRU turned out NOT speculative, see
    below — the shipped `DOWNBEATS_BGRU` models do need `GRULayer`/
    `GRUCell`).
  - Weights discipline is unchanged: every new model family goes through
    `models.py`'s sha256-pinned runtime download, never bundled; the
    CC-BY-NC-SA weights license is one models-repo-wide fact and applies
    to key/chords/onsets/notes exactly as it does to the downbeat models.
  - **4.0 status: DONE (2026-07-12).** Reference venv rebuilt at
    `../madmom-reference/.venv` (Python 3.10.18, numpy 1.23.5, scipy
    1.15.3, cython 0.29.37, setuptools 83.0.0, madmom 0.17.dev0 built
    from `../madmom-upstream` after populating its `madmom/models`
    submodule). **Faithfulness proof: PASSED, not fudged** -- both
    previously-skipped cross-BLAS exactness tests
    (`test_spectrogram.py::test_filtered_spectrogram_algorithm_is_exact_
    under_original_blas` and `test_downbeats_rnn.py::
    test_full_pipeline_is_exact_under_original_blas`) now run (path
    exists) and pass with **zero differing elements** against the
    already-committed Phase-1/2 fixtures -- the rebuilt venv's OpenBLAS
    build reproduces the original bit-for-bit for this port's actual
    computations, so no fixture was touched or regenerated. Full offline
    suite: 88 passed, 1 skipped (torch extra not installed, expected),
    11 deselected (network-marked). See the audit table below for the
    class-level gap inventory this wave produced.

### 4.0 audit result (2026-07-12)

Ground truth = the rebuilt reference venv's actual installed surface
(`madmom-reference/.venv`, introspected live) cross-checked against
`../madmom-upstream` source and `../madmom-upstream`'s `setup.py`
`package_data` (which pins exactly which model subdirs a real madmom
*install* ships -- narrower than the full models submodule checkout).

**Corrections to the wave plan's assumptions, found by this audit:**
- The "no speculative GRU/TCN ports" exclusion in Permanent exclusions
  is right for TCN but **wrong for GRU**: `setup.py`'s `package_data`
  ships `models/beats/201[56]/*` only (2015 BLSTM + 2016 LSTM), **not**
  `beats/2019` (TCN) -- confirmed empirically, `BEATS_TCN`'s pkl files
  are absent from the installed tree even though present in the
  `madmom_models` submodule checkout. So `TCNBeatProcessor`,
  `TCNTempoHistogramProcessor`, `TCNLayer`, `TCNBlock` stay EXCLUDED
  (no shipped model can ever reach them). But `models/downbeats/*/*` IS
  fully shipped, including the `downbeats_bgru_{harmonic,rhythmic}_*.pkl`
  ensemble (12 files) that `DOWNBEATS_BGRU`/`RNNBarProcessor` load --
  and unpickling one of those files (pickletools-walked, see below)
  references `GRULayer`/`GRUCell`, which **no wave currently ports**.
  This is a real gap, not a speculative one: flagging `GRULayer`,
  `GRUCell`, `RNNBarProcessor`, `SyncronizeFeaturesProcessor` as
  TO-PORT, tentatively slotted into 4c (closest existing "beats family"
  wave) pending an explicit amendment when 4c is planned in detail.
- `PitchClassProfile`/`HarmonicPitchClassProfile` (classic,
  non-neural-net chroma features in `audio/chroma.py`, backing
  `PitchClassProfileFilterbank`/`HarmonicPitchClassProfileFilterbank` in
  `audio/filters.py`) are public, user-facing classes the 4d bullet's
  text doesn't name (it only calls out the DNN/CLP chroma path). Added
  to 4d below as a scope addition.
- `key/2017` models exist in the submodule checkout but are **not**
  installed by a real madmom (`package_data` pins `key/2018/*` only) --
  `CNNKeyRecognitionProcessor` always resolves to `KEY_CNN` =
  `key/2018/key_cnn.pkl` regardless, so this has no effect on 4a, noted
  for completeness only.

**Status legend:** PORTED (in madmom_infer today) · TO-PORT (wave) ·
EXCLUDE (why).

| Upstream module | Class / function | Status | Model file(s) loaded | Notes |
|---|---|---|---|---|
| `audio/signal.py` | `Signal`, `SignalProcessor`, `FramedSignal`, `FramedSignalProcessor`, `Stream`, `LoadAudioFileError`, `remix`, `normalize`, `adjust_gain`, `attenuate`, `rescale`, `resample`, `root_mean_square`, `sound_pressure_level`, `energy`, `smooth`, `trim`, `signal_frame`, `load_audio_file`, `load_wave_file`, `write_wave_file` | PORTED | -- | Phase 1, complete |
| `audio/filters.py` | `Filterbank`, `LogarithmicFilterbank`, `log_frequencies`, `frequencies2bins`, `bins2frequencies`, freq-conversion helpers (`hz2mel` etc.) | PORTED | -- | Phase 1 |
| `audio/filters.py` | `MelFilterbank` | TO-PORT (4g) | -- | feeds `cepstrogram.py` MFCC + `CNNOnsetProcessor`'s 80-band mel input |
| `audio/filters.py` | `BarkFilterbank`, `RectangularFilter`, `RectangularFilterbank` | TO-PORT (4f) | -- | feeds `MultiBandSpectrogramProcessor`, used by `PatternTrackingProcessor` |
| `audio/filters.py` | `PitchClassProfileFilterbank`, `HarmonicPitchClassProfileFilterbank`, `SimpleChromaFilterbank`, `SemitoneBandpassFilterbank` | TO-PORT (4d, scope addition -- see corrections above) | -- | feed `audio/chroma.py`'s classic (non-DNN) chroma path |
| `audio/filters.py` | `HarmonicFilterbank` | TO-PORT (4g) | -- | used by `SemitoneBandpassSpectrogram`/harmonic feature paths; low priority, no processor in the named waves depends on it alone |
| `audio/stft.py` | `ShortTimeFourierTransform`, `ShortTimeFourierTransformProcessor`, `stft`, `fft_frequencies` | PORTED | -- | Phase 1 |
| `audio/stft.py` | `Phase`, `LocalGroupDelay`/`LGD`, `phase`, `local_group_delay`, `lgd`, `rfft_builder` | TO-PORT (4b) | -- | feeds onset phase-deviation family |
| `audio/spectrogram.py` | `Spectrogram`, `SpectrogramProcessor`, `FilteredSpectrogram(Processor)`, `LogarithmicSpectrogram(Processor)`, `SpectrogramDifference(Processor)` | PORTED | -- | Phase 1 |
| `audio/spectrogram.py` | `SuperFluxProcessor` | TO-PORT (4b) | -- | onset family |
| `audio/spectrogram.py` | `MultiBandSpectrogram`, `MultiBandSpectrogramProcessor` | TO-PORT (4f) | -- | `PatternTrackingProcessor` input |
| `audio/spectrogram.py` | `SemitoneBandpassSpectrogram` | TO-PORT (4d) | -- | `CLPChromaProcessor` input |
| `audio/cepstrogram.py` | `Cepstrogram`, `CepstrogramProcessor`, `MFCC`, `MFCCProcessor` | TO-PORT (4g) | -- | needs `MelFilterbank` first |
| `audio/chroma.py` | `DeepChromaProcessor` | TO-PORT (4d) | `CHROMA_DNN` = `chroma/2016/chroma_dnn.pkl` | pickle refs: `NeuralNetwork`, `FeedForwardLayer`, `relu`/`sigmoid` -- no new layer classes needed beyond 4a's set |
| `audio/chroma.py` | `CLPChroma`, `CLPChromaProcessor` | TO-PORT (4d) | -- | pure DSP, no NN weights; needs `SemitoneBandpassSpectrogram` |
| `audio/chroma.py` | `PitchClassProfile`, `HarmonicPitchClassProfile` | TO-PORT (4d, scope addition) | -- | classic chroma, not DNN-based |
| `audio/comb_filters.pyx` | `feed_forward_comb_filter`, `feed_backward_comb_filter`, `comb_filter`, `CombFilterbankProcessor` | TO-PORT (4c) | -- | numpy port (same playbook as `hmm.pyx`); feeds `TempoEstimationProcessor`'s comb-filter histogram mode |
| `audio/hpss.py` | `HPSS`/`HarmonicPercussiveSourceSeparation` | TO-PORT (4g) | -- | not consumed by any other TO-PORT processor in this audit; standalone preprocessing utility |
| `ml/hmm.py` | `TransitionModel`, `ObservationModel`, `DiscreteObservationModel`, `HiddenMarkovModel`/`HMM` | PORTED | -- | Phase 1 |
| `ml/crf.py` | `ConditionalRandomField` | TO-PORT (4d) | -- | chord decoding (`CRFChordRecognitionProcessor`, `DeepChromaChordRecognitionProcessor`) |
| `ml/gmm.py` | `GMM`, `log_multivariate_normal_density`, `logsumexp`, `pinvh` | TO-PORT (4f) | -- | backs `GMMPatternTrackingObservationModel` |
| `ml/nn/__init__.py` | `NeuralNetwork`, `NeuralNetworkEnsemble`, `average_predictions` | PORTED | -- | Phase 2 |
| `ml/nn/layers.py` | `Layer`, `FeedForwardLayer`, `RecurrentLayer`, `BidirectionalLayer`, `Gate`, `Cell`, `LSTMLayer` | PORTED | -- | Phase 2 |
| `ml/nn/layers.py` | `ConvolutionalLayer`, `MaxPoolLayer`, `BatchNormLayer`, `PadLayer`, `AverageLayer` | TO-PORT (4a) | -- | confirmed pickletools-walked as exactly what `key_cnn.pkl` (`AverageLayer`,`BatchNormLayer`,`ConvolutionalLayer`,`MaxPoolLayer`,`PadLayer`,`elu`,`linear`), `onsets_cnn.pkl`, `notes_cnn*.pkl`, `chords_cnnfeat.pkl` reference |
| `ml/nn/layers.py` | `GRULayer`, `GRUCell` | TO-PORT (tentative 4c, scope addition -- see corrections above) | -- | `downbeats_bgru_*.pkl` (12 files) reference these; no wave currently plans them |
| `ml/nn/layers.py` | `ReshapeLayer`, `TransposeLayer`, `StrideLayer` | TO-PORT (4a/4b, alongside the CNN infra that needs them) | -- | `notes_cnn.pkl` needs Reshape+Transpose; `onsets_cnn.pkl` needs Stride |
| `ml/nn/layers.py` | `TCNBlock`, `TCNLayer` | EXCLUDE | -- | no shipped model references them (`BEATS_TCN` not in `package_data`; confirmed by attempted load, file absent from installed tree) |
| `ml/nn/layers.py` | `MultiTaskLayer`, `ParallelLayer`, `SequentialLayer` | EXCLUDE | -- | only used by TCN multi-task models, which aren't shipped |
| `ml/nn/activations.py` | `linear`, `tanh`, `sigmoid`, `relu`, `elu`, `softmax` | PORTED | -- | Phase 2 |
| `features/beats_hmm.py` | `BeatStateSpace`, `BarStateSpace`, `BeatTransitionModel`, `BarTransitionModel`, `RNNBeatTrackingObservationModel`, `RNNDownBeatTrackingObservationModel`, `exponential_transition` | PORTED | -- | Phase 2 |
| `features/beats_hmm.py` | `MultiPatternStateSpace`, `MultiPatternTransitionModel`, `GMMPatternTrackingObservationModel` | TO-PORT (4f) | -- | pattern-tracking HMM machinery |
| `features/downbeats.py` | `RNNDownBeatProcessor`, `DBNDownBeatTrackingProcessor` | PORTED | `DOWNBEATS_BLSTM` | Phase 2, cross-BLAS-proven exact |
| `features/downbeats.py` | `RNNBarProcessor`, `SyncronizeFeaturesProcessor` | TO-PORT (tentative 4c, scope addition) | `DOWNBEATS_BGRU` | needs `GRULayer`/`GRUCell` (above) + `CLPChromaProcessor` (4d) |
| `features/downbeats.py` | `DBNBarTrackingProcessor`, `PatternTrackingProcessor` | TO-PORT (4f) | `PATTERNS_BALLROOM` (no NN globals -- GMM-only) | upstream's actual class name is `PatternTrackingProcessor`, not `GMMPatternTrackingProcessor` as the 4f bullet names it -- same processor, correcting the name here |
| `features/downbeats.py` | `LoadBeatsProcessor` | EXCLUDE | -- | file/STDIN batch-loading plumbing for `bin/` CLI scripts, not an inference algorithm |
| `features/beats.py` | `RNNBeatProcessor`, `DBNBeatTrackingProcessor`, `MultiModelSelectionProcessor` | TO-PORT (4c) | `BEATS_LSTM`, `BEATS_BLSTM` | pickle refs confirm no new layer classes beyond Phase-2's LSTM/BLSTM set |
| `features/beats.py` | `CRFBeatDetectionProcessor` | TO-PORT (4f) | -- | needs `features/beats_crf.pyx` numpy port |
| `features/beats.py` | `TCNBeatProcessor`, `detect_beats`, `threshold_activations` (TCN-specific parts) | EXCLUDE | `BEATS_TCN` (not shipped) | see corrections above; `threshold_activations` itself is already ported (`features/downbeats.py`) and reused, not duplicated |
| `features/tempo.py` | `TempoEstimationProcessor`, `TempoHistogramProcessor`, `ACFTempoHistogramProcessor`, `CombFilterTempoHistogramProcessor`, `DBNTempoHistogramProcessor`, `detect_tempo`, `dominant_interval`, `interval_histogram_acf`, `interval_histogram_comb`, `smooth_histogram` | TO-PORT (4c) | -- | comb variant needs `audio/comb_filters.py` port first |
| `features/tempo.py` | `TCNTempoHistogramProcessor` | EXCLUDE | -- | only consumes `TCNBeatProcessor` output, which can't exist (no shipped model) |
| `features/onsets.py` | `SpectralOnsetProcessor`, `spectral_diff`, `spectral_flux`, `superflux`, `complex_flux`, `complex_domain`, `rectified_complex_domain`, `high_frequency_content`, `modified_kullback_leibler`, `phase_deviation`, `weighted_phase_deviation`, `normalized_weighted_phase_deviation`, `correlation_diff`, `wrap_to_pi`, `peak_picking`, `OnsetPeakPickingProcessor` | TO-PORT (4b) | -- | pure DSP, no NN weights |
| `features/onsets.py` | `RNNOnsetProcessor` | TO-PORT (4b) | `ONSETS_RNN`, `ONSETS_BRNN`, `ONSETS_BRNN_PP` | pickle refs: `FeedForwardLayer`/`RecurrentLayer`/`BidirectionalLayer` -- all already PORTED (Phase 2), no new layer classes |
| `features/onsets.py` | `CNNOnsetProcessor` | TO-PORT (4b, reuses 4a's conv layers) | `ONSETS_CNN` | pickle refs: `ConvolutionalLayer`,`MaxPoolLayer`,`BatchNormLayer`,`StrideLayer` + `MelFilterbank` input |
| `features/key.py` | `CNNKeyRecognitionProcessor`, `key_prediction_to_label`, `add_axis` | TO-PORT (4a) | `KEY_CNN` = `key/2018/key_cnn.pkl` | pickle refs confirmed above |
| `features/chords.py` | `DeepChromaChordRecognitionProcessor` | TO-PORT (4d) | `CHORDS_DCCRF` | pickle has **no** NN globals -- CRF-only (`ml/crf.py`), confirms 4d's CRF-first framing |
| `features/chords.py` | `CNNChordFeatureProcessor` | TO-PORT (4d) | `CHORDS_CNN_FEAT` | pickle refs: `ConvolutionalLayer`,`BatchNormLayer`,`MaxPoolLayer` (4a's set, no new classes) |
| `features/chords.py` | `CRFChordRecognitionProcessor` | TO-PORT (4d) | `CHORDS_CFCRF` | pickle has no NN globals -- CRF-only |
| `features/chords.py` | `majmin_targets_to_chord_labels` | TO-PORT (4d) | -- | label-decoding helper alongside the chord processors |
| `features/notes_hmm.py` | `ADSRObservationModel`, `ADSRStateSpace`, `ADSRTransitionModel` | TO-PORT (4e) | -- | HMM state spaces on existing `ml/hmm.py` machinery |
| `features/notes.py` | `RNNPianoNoteProcessor` | TO-PORT (4e) | `NOTES_BRNN` = `notes/2013/notes_brnn.pkl` | pickle refs: `BidirectionalLayer`,`FeedForwardLayer`,`RecurrentLayer` -- already PORTED, no new classes |
| `features/notes.py` | `CNNPianoNoteProcessor` | TO-PORT (4e, reuses 4a's conv layers) | `NOTES_CNN`, `NOTES_CNN_MIREX` | pickle refs: `ConvolutionalLayer`,`BatchNormLayer`,`ReshapeLayer`(+`TransposeLayer` for `NOTES_CNN`) |
| `features/notes.py` | `ADSRNoteTrackingProcessor`, `NotePeakPickingProcessor`, `NoteOnsetPeakPickingProcessor` | TO-PORT (4e) | -- | decode/peak-picking, no NN weights |
| **`evaluation/*`** | (entire subpackage) | EXCLUDE | -- | out of scope per this repo's stated scope (see top of this file) |
| **`bin/*`** | (CLI scripts, installed as `console_scripts`-style `scripts=` entries by upstream `setup.py`) | EXCLUDE | -- | this package is a library, processors are the API (Permanent exclusions) |
| **`io/*`, `utils/*`** | `io.audio`, `io.midi`, `utils.midi`, `utils.stats` | EXCLUDE (out of this audit's stated scope: `features/`, `audio/`, `ml/` only) | -- | I/O/annotation-file helpers, not inference algorithms; flagged here rather than silently dropped, revisit only if a TO-PORT processor is found to need one (none currently do) |

## File-top header convention

Every module in this codebase starts with a header of this shape (as a Python
module docstring):

```python
"""One-line title.

2-3 sentences: what this file is for, and *why* it exists this way -- the
design constraint or decision it embodies, not just a restatement of the
code. For stubs, say plainly that it's not yet implemented and which phase
it belongs to.

Reads: <files/libraries this module reads or depends on>; read by: <files
that depend on this one>, where useful
"""
```

Thinner files (e.g. a filterbank leaf module) get a shorter, honest version
of the same shape -- don't pad it out artificially. Keep headers in sync as
files change; this is what lets a future session (or the `/nav:sync` skill)
grasp any file from its first ~12 lines without reading the whole thing.

## Golden-fixture testing philosophy (numpy backend is the reference; torch frontend is Phase 3a, shipped)

This project follows the same pattern established by the sibling
all-in-one-infer package's pure-Python NATTEN replacement:

- A **numpy backend** is the default and required implementation, treated as
  the reference. It must be verified **bit-identical to original madmom**
  using golden fixtures -- recorded input/output pairs captured from running
  the real (compiled) madmom, checked against this port's output in tests.
  Do not consider a Phase 1/2/3 module "done" until it has a golden-fixture
  test, not just a hand-written unit test.
- An **optional torch backend** (`madmom_infer/torch/`, gated behind the
  `torch` extra, guarded/lazy -- `import madmom_infer` never touches torch)
  exists as of Phase 3a: a batched, autograd-differentiable, device-agnostic
  spectrogram frontend (framing/STFT/filterbank/log-compression/temporal-
  diff), reusing the numpy backend's own window/filterbank-matrix/diff-
  frame-count construction rather than re-deriving that DSP knowledge (see
  `madmom_infer/torch/audio/frontend.py`'s module docstring for the exact
  reuse boundary). Verified against the numpy backend per
  `tests/test_torch_frontend.py` (float32 vs the real shipped numpy
  processor chain, float64 vs a bespoke float64-throughout numpy test
  harness -- since numpy's own classes hardcode a complex64/float32
  ceiling and cannot produce a genuine float64 baseline), plus
  `gradcheck`, batching, and CPU/CUDA device tests. The RNN ensemble
  forward pass is NOT covered (Phase 3b, not started -- madmom's LSTM
  peephole connections have no `torch.nn.LSTM` equivalent, needs a custom
  cell), nor is Viterbi/DBN decoding (inherently sequential, discrete-state
  -- no torch benefit expected there, ever). Don't oversell torch-backend
  speedups for sequential algorithms in docs or commit messages.
- Never bundle madmom's own pretrained weights (CC BY-NC-SA 4.0) -- see
  README.md's "What this project will NEVER bundle" section. This is a
  permanent policy, not a phase-gate detail to relax later. Phase 2
  downloads them at runtime instead (`madmom_infer/models.py`) -- see that
  module's docstring before touching it.
- Unpickling madmom's own `.pkl` model files must go through
  `madmom_infer/ml/nn/unpickle.py`'s restricted, class-allowlisted
  `SafeUnpickler` -- never a bare `pickle.load`. A downloaded model file is
  a lower-trust artifact than this project's own source; extending the
  allowlist (e.g. to support a new model family in a later phase) means
  adding an explicit `(module, name) -> object` entry, never widening
  `find_class` to accept unlisted globals.

## Phase-2 verification commands

The network-dependent tests in `tests/test_ml_nn.py` and
`tests/test_downbeats_rnn.py` (they download real madmom weights via
`madmom_infer.models.downbeats_blstm()`) are marked `pytest.mark.network`
and **deselected by default** (`pyproject.toml`'s `addopts = "-m 'not
network'"`, same convention as the sibling maest-infer repo) so plain `uv
run pytest` never needs network access -- CI runs exactly that. Run them
explicitly with `-m network` for real before considering a Phase-2 change
"done" (skips cleanly if the network is unavailable -- don't be alarmed by
skips in an offline sandbox):

```bash
uv run pytest -m network tests/test_ml_nn.py tests/test_downbeats_rnn.py -v
```

The strongest Phase-2 acceptance check is `test_downbeats_rnn.py`'s
`test_full_pipeline_is_exact_under_original_blas`, which shells out to the
reference venv (`madmom-reference/.venv`, numpy 1.23.5 -- the same
technique `test_spectrogram.py` established in Phase 1) and asserts this
project's own `RNNDownBeatProcessor` -> `DBNDownBeatTrackingProcessor`
output is bit-identical to real madmom's, not just within a tolerance.
Regenerate the Phase-2 fixtures it and `test_ml_nn.py` depend on with:

```bash
/home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python \
    tools/generate_phase2_fixtures.py
```

The reference venv originally lived at `all-in-one-fix/.venv`; that
directory (and the whole `all-in-one-fix` checkout) no longer exists on
this machine. It was rebuilt 2026-07-12 at `madmom-reference/.venv`
(Wave 4.0) to the exact same recorded versions (Python 3.10.18, numpy
1.23.5, scipy 1.15.3) from `../madmom-upstream`, and every
`REFERENCE_PYTHON` path in tests/tools now points there -- see Wave 4.0's
entry above for the faithfulness-proof outcome.

## Phase-3a verification commands

`tests/test_torch_frontend.py` needs torch (`uv sync --extra dev --extra
torch` or `pip install "madmom-infer[dev,torch]"`); it uses
`pytest.importorskip("torch")` at module scope, so plain `uv run pytest`
skips it cleanly on a torch-less install rather than failing collection --
don't be alarmed by that one skip in a core-only environment. Run it
explicitly once torch is installed:

```bash
uv run pytest tests/test_torch_frontend.py -v
```

Before considering a Phase-3a torch-frontend change "done", also verify the
opt-in boundary itself still holds -- both a torch-less venv (`import
madmom_infer` works, `import madmom_infer.torch` raises a clear guarded
`ImportError`) and the full non-network suite pass in both a torch-less and
a torch-installed environment with identical non-torch test counts.
