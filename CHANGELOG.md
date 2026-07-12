# Changelog

All notable changes to madmom-infer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

**Wave 4e of the complete-port campaign (`feat/complete-port` branch): piano
note transcription.** Adds `RNNPianoNoteProcessor` (RNN onset activations)
and `CNNPianoNoteProcessor` (multi-task CNN note/onset/offset activations),
`NoteOnsetPeakPickingProcessor`/`NotePeakPickingProcessor` (peak-picking
decode, reusing the onset-detection family's `peak_picking`), and
`ADSRNoteTrackingProcessor` (a new attack-decay-sustain-release HMM,
`features/notes_hmm.py`, on the existing Phase-1 `ml/hmm.py` Viterbi
decoder).

### Added
- `madmom_infer/features/notes_hmm.py` (new module): `ADSRStateSpace`,
  `ADSRTransitionModel`, `ADSRObservationModel` -- a near-line-for-line port
  of upstream's per-pitch ADSR HMM state space (silence -> attack -> decay ->
  sustain -> release), built on the same `ml/hmm.py` `TransitionModel`/
  `ObservationModel` base classes `features/beats_hmm.py` already uses.
- `madmom_infer/ml/nn/layers.py`: `ReshapeLayer`, `TransposeLayer` --
  confirmed by `pickletools`-walking all 4 target note-CNN pickles
  (`notes_cnn.pkl` = `NOTES_CNN`, `notes_cnn_{1,2}.pkl` = `NOTES_CNN_MIREX`,
  unused by any ported processor but walked anyway) to be exactly the 2 new
  layer classes needed on top of already-ported ones.
- **Real surprise, found by `pickletools`-walking `notes_cnn.pkl` directly,
  not guessed**: it does not pickle a bare `NeuralNetwork` the way every
  other target `.pkl` in this project does -- it pickles the model's ENTIRE
  multi-task `madmom.processors.SequentialProcessor`/`ParallelProcessor`
  branch-and-`dstack` graph directly (3 parallel note/onset/offset branches,
  each `ConvolutionalLayer` -> `TransposeLayer` -> `ReshapeLayer` ->
  `FeedForwardLayer`, merged by a plain `numpy.dstack` reference). This
  turned out to need zero new code in `ml/nn/__init__.py` --
  `NeuralNetworkEnsemble.load`/`NeuralNetwork.load` were already fully
  generic (matching upstream's own equally generic `Processor.load`) -- only
  new `ml/nn/unpickle.py` allowlist entries: `madmom.processors.
  {SequentialProcessor,ParallelProcessor}`, `numpy.dstack` (two module-path
  spellings across the target pickles), and two Python-2-pickle-compat
  primitives (`_codecs.encode`, `itertools.imap` -> Python 3's builtin
  `map`) that real madmom's own bare `pickle.load` resolves transparently
  via `pickle._compat_pickle.NAME_MAPPING` but this project's
  allowlist-only `SafeUnpickler.find_class` does not, needing explicit
  entries (same shape of gap wave 4c's `copy_reg._reconstructor`/
  `__builtin__.object` entries already closed for the older-format
  `downbeats_bgru_*.pkl` files).
- `madmom_infer/features/notes.py` (new module): `RNNPianoNoteProcessor`,
  `NoteOnsetPeakPickingProcessor`, `NotePeakPickingProcessor` (upstream's
  deprecated-since-0.17 alias, ported anyway -- the audit table lists it as
  real, not dead code), `_cnn_pad`, `CNNPianoNoteProcessor`,
  `ADSRNoteTrackingProcessor`.
- `madmom_infer/models.py`: `notes_brnn()`/`NOTES_BRNN` (1 file),
  `notes_cnn()`/`NOTES_CNN` (1 file) -- both sha256s computed from the local
  `../madmom-upstream` submodule checkout AND cross-checked byte-for-byte
  against fresh `raw.githubusercontent.com/CPJKU/madmom_models` downloads,
  identical. `NOTES_CNN_MIREX` (`notes/2018/notes_cnn_[12].pkl`) is real,
  `package_data`-shipped, and was `pickletools`-walked for completeness, but
  -- like wave 4b's `ONSETS_BRNN_PP` -- no ported processor loads it
  (upstream's own `CNNPianoNoteProcessor.__init__` hardcodes `NOTES_CNN`
  only), so it has no registry entry.
- New `tools/generate_notes_fixtures.py`: `notes_brnn.pkl`/`notes_cnn.pkl`
  structural digests (the latter a recursive digest of the nested processor
  graph above), self-contained `ReshapeLayer`/`TransposeLayer` golden
  fixtures, `RNNPianoNoteProcessor`/`CNNPianoNoteProcessor` end-to-end
  activations plus decoded notes for all 3 usable 44.1kHz test-wav cases
  (fresh processor instances per case -- a shared instance across differing-
  dtype wavs was found, empirically, to make REAL madmom itself silently
  produce a materially wrong `float32_44100` RNN activation array, a real
  upstream `FilteredSpectrogramProcessor`/`ShortTimeFourierTransformProcessor`
  caching artifact, same category wave 4d already documented for
  `RNNBarProcessor`), and two SYNTHETIC (hand-crafted, deterministic)
  fixtures for `ADSRNoteTrackingProcessor`/`NoteOnsetPeakPickingProcessor`'s
  decode logic -- the real-audio test wavs decode to EMPTY output on every
  case, too weak a fixture to exercise the segmentation branches, so the
  synthetic ones include a deliberately INCOMPLETE note that must be
  discarded.
- **Faithfulness proof: PASSED.** `tests/test_notes.py::
  test_full_pipeline_is_exact_under_original_blas` reproduces real madmom's
  `RNNPianoNoteProcessor`/`CNNPianoNoteProcessor` activations AND decoded
  notes (peak-picked and ADSR-HMM-decoded) with **zero differing elements**,
  for all 3 44.1kHz test-wav cases, both model families -- confirmed also
  independently for the synthetic ADSR fixture
  (`test_adsr_synthetic_decode_is_exact_under_original_blas`). In-process
  (differing-BLAS-build) drift: CNN activations up to 247 ULP (asserted at a
  1024-ULP margin, ~4x observed, matching this repo's convention); RNN
  activations (a raw, near-zero-centered linear-layer output, not a
  bounded-[0,1] probability -- an ULP-view metric is unstable that close to
  zero) measured up to ~7.15e-7 absolute (asserted at `atol=1e-5`, ~14x
  observed, same "documented absolute tolerance, not ULP" precedent as wave
  4d's `SemitoneBandpassSpectrogram` finding).
- 23 new tests (`tests/test_notes.py`: 12 offline + 11 network);
  `tests/test_fixtures_exist.py`: +6 (2 new structural-digest/params test
  functions + 4 new parametrized fixture-file cases). Full offline suite:
  223 passed, 1 skipped, 51 deselected (was 205/1/40 after 4d); network
  suite: 51 passed, 1 skipped, 223 deselected, all green.

**Wave 4d of the complete-port campaign (`feat/complete-port` branch): chroma
+ chords, and closing wave 4c's `RNNBarProcessor` loop.** Adds a numpy
Conditional Random Field decoder (`ml/crf.py`), all three chroma paths
(classic `PitchClassProfile`/`HarmonicPitchClassProfile`, DNN
`DeepChromaProcessor`, and CLP `CLPChroma`/`CLPChromaProcessor`), two full
audio-in chord-recognition pipelines, and (unblocked by `CLPChromaProcessor`)
`RNNBarProcessor`'s first real audio-in end-to-end proof.

### Added
- `madmom_infer/ml/crf.py` (new module): `ConditionalRandomField` -- a
  pure-numpy, matrix-formulation Viterbi decoder (forward-inference only).
  Adds a `.load()` classmethod (not in upstream, which inherits `Processor.
  load`'s bare `pickle.load`) delegating to `unpickle.load_model`, matching
  `NeuralNetwork.load`'s restricted-unpickling convention.
- `madmom_infer/ml/nn/unpickle.py`: 1 new `ALLOWED_GLOBALS` entry
  (`madmom.ml.crf.ConditionalRandomField`) -- confirmed by `pickletools` to
  be the ONLY new global all 4 target `.pkl` files need beyond already-
  ported classes (`chroma_dnn.pkl` needs only `FeedForwardLayer`/`relu`/
  `sigmoid`; `chords_cnnfeat.pkl` needs only wave 4a's CNN layer set;
  `chords_dccrf.pkl`/`chords_cnncrf.pkl` need only `ConditionalRandomField`).
- `madmom_infer/audio/filters.py`: `hz2midi`, `midi2hz`, `semitone_frequencies`,
  `PitchClassProfileFilterbank`, `HarmonicPitchClassProfileFilterbank`,
  `SemitoneBandpassFilterbank` -- verbatim/composition ports. Also
  `SimpleChromaFilterbank`, ported **including its unconditional
  `raise NotImplementedError`**: confirmed by reading upstream directly that
  it is not actually implemented in real madmom either (dead code below the
  raise), so this port reproduces that not-implemented state rather than
  "fixing" it.
- `madmom_infer/audio/signal.py`: `resample()` -- a **policy correction, not
  silently made**: this project's "no ffmpeg dependency" stance (Phase 1
  through 4c) does not survive `SemitoneBandpassFilterbank`, whose ~78
  semitone bands each filter at a FIXED sample rate (882/4410/22050 Hz)
  unconditionally different from this project's 44100 Hz convention --
  resampling is load-bearing on every call here, not an optional
  convenience. Implemented as a narrow ffmpeg-subprocess call (only the
  exact shape this one caller needs), shelling out to the system `ffmpeg`
  binary with the same command shape real madmom's own `_ffmpeg_call`
  builds -- verified **bit-identical** (`np.array_equal`) against real
  madmom's own `resample()` output, both sides invoking the literal same
  system `ffmpeg` binary.
- `madmom_infer/audio/spectrogram.py`: `SemitoneBandpassSpectrogram` -- own
  composition class (not a `FilteredSpectrogram` subclass: no STFT stage,
  a time-domain IIR filterbank instead). Measured (not bit-identical):
  matches real madmom's output to within ~1e-5 absolute across the 3 usable
  test-wav cases when run under a different scipy version than the one that
  recorded the reference fixture (`scipy.signal.filtfilt`'s recursive
  nature amplifies tiny per-version `ellip()` coefficient differences) --
  documented plainly rather than claimed exact.
- `madmom_infer/audio/chroma.py` (new module): `PitchClassProfile`/
  `HarmonicPitchClassProfile` (composition subclasses of `Spectrogram`, not
  upstream's ndarray-view hierarchy), `DeepChromaProcessor`, `CLPChroma`/
  `CLPChromaProcessor`. Found and fixed a genuine latent bug in wave 4c's
  own `SyncronizeFeaturesProcessor` (`features/downbeats.py`), surfaced by
  actually exercising `RNNBarProcessor` end-to-end for the first time: it
  called `features.T` directly, which works on upstream's `np.ndarray`-
  subclass spectrograms but raises `AttributeError` on this project's own
  composition-style ones -- fixed with an explicit `np.asarray(features).T`.
- `madmom_infer/features/chords.py` (new module): `majmin_targets_to_chord_
  labels`, `DeepChromaChordRecognitionProcessor`, `CNNChordFeatureProcessor`
  (`nn_file=` override added, not in upstream, for testability -- matches
  the `nn_files=`/`models=` convention `CNNKeyRecognitionProcessor`/
  `DeepChromaProcessor` already establish), `CRFChordRecognitionProcessor`.
  Confirmed by reading upstream directly: NEITHER chord-recognition path
  touches `CLPChroma` at all, so full audio-in chord recognition is
  achievable and EXACT-testable independent of the CLP-chroma precision
  caveat above.
- `madmom_infer/models.py`: `chroma_dnn()`/`CHROMA_DNN`, `chords_dccrf()`/
  `CHORDS_DCCRF`, `chords_cnn_feat()`/`CHORDS_CNN_FEAT`, `chords_cfcrf()`/
  `CHORDS_CFCRF` -- 4 sha256-pinned registry entries, cross-checked against
  both a fresh `raw.githubusercontent.com/CPJKU/madmom_models` download and
  the local `../madmom-upstream` submodule checkout (identical).
- **`RNNBarProcessor` (wave 4c, `features/downbeats.py`) is now instantiable
  and provably correct end-to-end from raw audio**, closing the loop that
  wave was left open. `tests/test_downbeats_rnn.py`'s new "RNNBarProcessor
  end-to-end" section: decoded beat times EXACT, decoded downbeat activation
  within ~4e-8 absolute (both in-process and cross-BLAS) -- the GRU
  ensemble's own forward pass was already proven bit-exact in wave 4c
  independent of this wave's CLP-chroma feature-extraction noise.
- 35 new tests total (`tests/test_crf.py`: 4; `tests/test_chroma.py`: 22;
  `tests/test_chords.py`: 9; `tests/test_downbeats_rnn.py`: +2; `tests/
  test_fixtures_exist.py`: +7 more) and `tools/generate_chroma_chord_
  fixtures.py` (the fixture-generation script, reference-venv-only).
  Faithfulness proofs: `tests/test_crf.py`/`test_chords.py`'s cross-BLAS
  tests reproduce real madmom's CRF-decoded chord label sequences AND
  segment boundaries with **zero differing elements**, both recognition
  paths, all 3 cases; `tests/test_chroma.py`'s `DeepChromaProcessor`
  cross-BLAS test likewise **zero differing elements**. Full offline suite
  now 205 passed, 1 skipped, 40 deselected (was 174/1/25); network suite 40
  passed, 1 skipped, 205 deselected, all green.

**Wave 4c of the complete-port campaign (`feat/complete-port` branch): beat
tracking + tempo estimation + GRU support.** Adds `RNNBeatProcessor`,
`DBNBeatTrackingProcessor` (beat-only), `MultiModelSelectionProcessor`,
the full tempo histogram family (`TempoEstimationProcessor`, `acf`/`comb`/
`dbn` modes), a numpy port of `audio/comb_filters.pyx`, and `GRULayer`/
`GRUCell` (the shipped `DOWNBEATS_BGRU` models' layer family, an audit
correction -- see CLAUDE.md).

### Added
- `madmom_infer/audio/comb_filters.py` (new module): `feed_forward_comb_filter`,
  `feed_backward_comb_filter` (+ 1D/2D helpers), `comb_filter`,
  `CombFilterbankProcessor` -- numpy port of `audio/comb_filters.pyx`,
  proven **bit-identical** to real madmom (`np.array_equal`, both
  in-process and cross-BLAS, no tolerance needed at all -- neither function
  touches BLAS). Found and reproduced two real precision quirks, both
  confirmed empirically against the reference venv: (1) real madmom's
  `feed_backward_comb_filter` silently rounds `alpha` to float32 precision
  before its accumulation loop (a Cython `cdef ... float alpha` parameter
  coercion); (2) `comb_filter`'s per-tau dispatch extracting `alpha[i]`
  from a numpy array is a numpy-2.x-vs-1.23.5 scalar-promotion divergence
  (NEP 50) -- fixed with an explicit `float(alpha[i])` cast, same class of
  fix as `features/onsets.py`'s `normalized_weighted_phase_deviation`.
- `madmom_infer/ml/nn/layers.py`: `GRUCell`, `GRULayer` -- the layer family
  all 12 `downbeats_bgru_{harmonic,rhythmic}_[0-5].pkl` files need
  (confirmed by `pickletools`). Found the real shipped files use an
  OLDER pickle format than every other target `.pkl` in this project
  (loading one emits real madmom's own "please update your GRU models"
  `RuntimeWarning`) -- ported `GRULayer.__setstate__`'s legacy
  `hid_init` -> `init` rename branch verbatim (initially dropped as
  presumed-dead code; empirically NOT dead, all 12 real files exercise it).
- `madmom_infer/ml/nn/unpickle.py`: 4 new `ALLOWED_GLOBALS` entries --
  `GRUCell`, `GRULayer`, plus 2 generic old-style-class reconstruction
  primitives (`copy_reg._reconstructor` -> `copyreg._reconstructor`,
  `__builtin__.object` -> `builtins.object`) the older-format
  `downbeats_bgru_*.pkl` files also reference.
- `madmom_infer/features/beats.py` (new module): `RNNBeatProcessor`
  (online + offline LSTM/BLSTM ensembles), `DBNBeatTrackingProcessor`
  (beat-only, offline-only -- reuses `beats_hmm.py`'s existing
  `BeatStateSpace`/`BeatTransitionModel`/`RNNBeatTrackingObservationModel`),
  `MultiModelSelectionProcessor`. Found and fixed a genuine Phase-2 latent
  bug this wave surfaced: `RNNBeatTrackingObservationModel.log_densities`
  (`features/beats_hmm.py`) called `np.asarray(observations, ndmin=1)`,
  which is not valid on ANY numpy version (`asarray` has no `ndmin`
  parameter) -- a previous wave's numpy-2.x-compat note wrongly claimed
  this worked; never exercised until `DBNBeatTrackingProcessor` (this
  class was the only caller). Fixed as `np.array(observations, ndmin=1)`.
  `CRFBeatDetectionProcessor` stays out of scope (wave 4f); `BeatTrackingProcessor`/
  `BeatDetectionProcessor`/`detect_beats` are real upstream classes with no
  wave assignment in the audit table (flagged, not silently ported).
- `madmom_infer/features/tempo.py` (new module): `smooth_histogram`,
  `interval_histogram_acf`, `interval_histogram_comb`, `dominant_interval`,
  `detect_tempo`, `TempoHistogramProcessor`, `ACFTempoHistogramProcessor`,
  `CombFilterTempoHistogramProcessor`, `DBNTempoHistogramProcessor`,
  `TempoEstimationProcessor` -- all offline-only (`OnlineProcessor` stays a
  permanent exclusion). `TCNTempoHistogramProcessor` stays EXCLUDED (only
  consumes `TCNBeatProcessor` output, which cannot exist -- `BEATS_TCN` is
  not shipped by a real madmom install).
- `madmom_infer/features/downbeats.py`: `SyncronizeFeaturesProcessor`
  (pure numpy, proven bit-identical), `RNNBarProcessor` (ported verbatim,
  but cannot be instantiated end-to-end from raw audio until wave 4d ports
  `CLPChromaProcessor` -- its GRU-ensemble forward pass is instead proven
  bit-exact via a golden intermediate-feature fixture captured from real
  madmom, not a full audio-in run).
- `madmom_infer/ml/nn/__init__.py`: fixed a genuine numpy-2.x-vs-1.23.5
  DTYPE divergence in `average_predictions` (not upstream's fault) --
  averaging a list of 0-DIMENSIONAL float32 arrays (an ensemble fed a
  single-frame input, as `RNNBarProcessor`'s GRU ensembles can be)
  silently stays float32 on numpy >= 2.0 (NEP 50) but real madmom's own
  `sum(pred) / len(pred)` upcasts to float64 on numpy < 2.0 (its
  value-based casting treats 0-d "scalar-kind" arrays differently from
  N-d ones) -- fixed with an explicit branch reproducing the old
  (real-madmom-recorded) dtype on every numpy version; N-d predictions
  (every other model family, including the already-shipped
  `DOWNBEATS_BLSTM` ensemble) are unaffected, already matched on both.
- `madmom_infer/models.py`: `beats_lstm()`/`BEATS_LSTM` (8 files),
  `beats_blstm()`/`BEATS_BLSTM` (8 files), `downbeats_bgru_rhythmic()`/
  `downbeats_bgru_harmonic()`/`downbeats_bgru()`/`DOWNBEATS_BGRU` (12
  files, `[rhythmic, harmonic]` list-of-lists matching upstream's own
  shape) -- 28 sha256-pinned registry entries, cross-checked against both
  a fresh `raw.githubusercontent.com/CPJKU/madmom_models` download and the
  local `../madmom-upstream` submodule checkout (identical).
- 66 new tests (`tests/test_comb_filters.py`: 17; `tests/test_beats.py`:
  15; `tests/test_tempo.py`: 15; `tests/test_downbeats_rnn.py`: +13 GRU/
  sync-features tests; `tests/test_fixtures_exist.py`: +6 more) and
  `tools/generate_beat_tempo_fixtures.py` (the fixture-generation script,
  reference-venv-only). Faithfulness proof: `tests/test_beats.py::
  test_full_pipeline_is_exact_under_original_blas` reproduces real
  madmom's `RNNBeatProcessor`(online=False/True) + `DBNBeatTrackingProcessor`
  activations AND decoded beat times with **zero differing elements**;
  `tests/test_comb_filters.py`/`tests/test_tempo.py` are bit-identical
  even IN-PROCESS (no BLAS involved); `tests/test_downbeats_rnn.py::
  test_downbeats_bgru_ensembles_are_exact_under_original_blas` proves the
  GRU forward pass itself bit-identical. Full offline suite now 174
  passed, 1 skipped, 25 deselected (was 123/1/20); network suite 25
  passed, 1 skipped, 174 deselected, all green.

**Wave 4b of the complete-port campaign (`feat/complete-port` branch):
onset detection.** Adds the complete spectral-flux/phase-deviation/
complex-domain onset detection DSP function family, `SpectralOnsetProcessor`,
`RNNOnsetProcessor` (online + offline RNN ensembles), `CNNOnsetProcessor`,
and `OnsetPeakPickingProcessor` end-to-end (spectrogram frontend -> onset
activation function -> decoded onset times).

### Added
- `madmom_infer/audio/stft.py`: `phase`, `Phase`, `local_group_delay`/`lgd`,
  `LocalGroupDelay`/`LGD`, `ShortTimeFourierTransform.phase()` -- the
  phase-vocoder-style analysis chain the onset phase-deviation family reads
  off `spectrogram.stft.phase()`/`...phase().lgd()`. Faithfully reproduces a
  real upstream bug: `LocalGroupDelay.__new__`'s "reuse an existing `Phase`"
  branch is never actually reachable (an undefined-name reference resolves
  to the module's own `stft()` function instead of the `phase` argument),
  so it always rebuilds -- ported bug-for-bug, not silently fixed.
- `madmom_infer/audio/filters.py`: `hz2mel`, `mel2hz`, `mel_frequencies`,
  `MelFilterbank` -- pulled forward from wave 4g (`CNNOnsetProcessor`'s
  80-band mel input needs it now; 4g's MFCC work will reuse this).
- `madmom_infer/audio/spectrogram.py`: `SuperFluxProcessor`,
  `Spectrogram.diff()`/`.filter()`/`.log()` convenience methods,
  `SpectrogramProcessor.__init__(self, **kwargs): pass` (matches upstream).
- `madmom_infer/audio/signal.py`: `smooth()` (needed by `peak_picking`;
  corrects a Phase-1 audit-table overstatement -- it was listed as already
  PORTED but was not actually present), and a `**kwargs` catch-all added to
  `FramedSignalProcessor.__init__` (matches upstream's own signature,
  needed for `SpectralOnsetProcessor`'s kwargs-forwarding design).
- `madmom_infer/utils.py` (new module): `segment_axis` (narrow carve-out --
  only the case `StrideLayer` needs) and `combine_events` (full port) --
  two real, non-speculative dependencies found while porting this wave,
  not a general `madmom.utils` port.
- `madmom_infer/ml/nn/layers.py`: `StrideLayer` -- the one new layer class
  `onsets_cnn.pkl` needs beyond wave 4a's CNN set (confirmed by
  `pickletools`, not guessed).
- `madmom_infer/ml/nn/unpickle.py`: 2 new `ALLOWED_GLOBALS` entries
  (`madmom.ml.nn.layers.StrideLayer`, `numpy.core.multiarray.scalar`).
- `madmom_infer/models.py`: `onsets_rnn()`/`ONSETS_RNN` (8 files),
  `onsets_brnn()`/`ONSETS_BRNN` (8 files), `onsets_cnn()`/`ONSETS_CNN`
  (1 file) -- 17 sha256-pinned registry entries, cross-checked against both
  a fresh `raw.githubusercontent.com/CPJKU/madmom_models` download and the
  local `../madmom-upstream` submodule checkout (identical).
- `madmom_infer/features/onsets.py` (new module): the complete DSP function
  family (`wrap_to_pi`, `correlation_diff`, `high_frequency_content`,
  `spectral_diff`, `spectral_flux`, `superflux`, `complex_flux`,
  `modified_kullback_leibler`, `phase_deviation`, `weighted_phase_deviation`,
  `normalized_weighted_phase_deviation`, `complex_domain`,
  `rectified_complex_domain`), `SpectralOnsetProcessor`, `RNNOnsetProcessor`,
  `CNNOnsetProcessor`, `peak_picking`, `OnsetPeakPickingProcessor` (offline
  only -- `OnlineProcessor` stays a permanent exclusion). `correlation_diff`
  is a faithful port of a function that crashes under Python 3 in REAL
  madmom too (confirmed against the reference venv), pinned by a
  `pytest.raises(TypeError)` test rather than a golden output. Fixed one
  genuine numpy-2.x-vs-1.23.5 dtype-promotion divergence (not upstream's
  fault): `normalized_weighted_phase_deviation` silently upcast `float32`
  to `float64` under numpy >= 2.0's NEP 50 scalar-promotion rules; an
  explicit `.astype(np.float32)` reproduces real madmom's actual dtype on
  every numpy version.
- 31 new tests (`tests/test_onsets.py`: 20 offline + 6 `pytest.mark.network`;
  `tests/test_fixtures_exist.py`: 5 more) and
  `tools/generate_onset_fixtures.py` (the fixture-generation script,
  reference-venv-only). Faithfulness proof: `tests/test_onsets.py::
  test_full_pipeline_is_exact_under_original_blas` reproduces real madmom's
  `RNNOnsetProcessor`(online=False/True) + `CNNOnsetProcessor` +
  `OnsetPeakPickingProcessor` activations AND decoded onset times with
  **zero differing elements**. In-process ULP drift measured at up to 17
  ULP for the pure-DSP functions (asserted at a 64-ULP margin) and up to 62
  ULP for the RNN/BRNN/CNN activations (asserted at a 256-ULP margin) --
  decoded onset times are EXACT despite that drift. Full offline suite now
  123 passed, 1 skipped, 20 deselected (was 98/1/14).

**Wave 4a of the complete-port campaign (`feat/complete-port` branch): CNN
runtime + key detection.** Adds `CNNKeyRecognitionProcessor` end-to-end
(audio in, 24-class major/minor key probabilities + decoded label out) and
the 5 CNN-era layer classes it -- and every other CNN-based madmom model --
needs.

### Added
- `madmom_infer/ml/nn/layers.py`: `ConvolutionalLayer`, `MaxPoolLayer`,
  `BatchNormLayer`, `PadLayer`, `AverageLayer` -- ported from
  `madmom.ml.nn.layers`, verified against `key_cnn.pkl`'s actual
  `pickletools`-walked global references (exactly these 5 classes plus the
  already-ported `elu`/`linear` activations -- no `ReshapeLayer`/
  `TransposeLayer`/`StrideLayer` needed for this model). Only the
  `scipy.ndimage.convolve` backend of `ConvolutionalLayer`'s `convolve()`
  helper is implemented (no `opencv-python` dependency; the reference venv
  that recorded this wave's fixtures has no `cv2` installed either, so this
  is the real, not a speculative, code path).
- `madmom_infer/ml/nn/unpickle.py`: 5 new `ALLOWED_GLOBALS` entries mapping
  `madmom.ml.nn.layers.{ConvolutionalLayer,MaxPoolLayer,BatchNormLayer,
  PadLayer,AverageLayer}` onto the classes above.
- `madmom_infer/models.py`: `key_cnn()` / `KEY_CNN` registry entry
  (`key/2018/key_cnn.pkl`, sha256-pinned, cross-checked against both a
  fresh `raw.githubusercontent.com/CPJKU/madmom_models` download and the
  local `../madmom-upstream` submodule checkout -- identical bytes).
- `madmom_infer/features/key.py` (new module): `CNNKeyRecognitionProcessor`,
  `key_prediction_to_label`, `add_axis`, `KEY_LABELS` -- the full
  SignalProcessor -> FramedSignalProcessor(8192, fps=5) ->
  ShortTimeFourierTransformProcessor -> FilteredSpectrogramProcessor(24
  bands, 65-2100Hz) -> LogarithmicSpectrogramProcessor -> CNN ensemble ->
  softmax pipeline.
- 9 new tests (`tests/test_key.py`, 6 offline + 3 `pytest.mark.network`): 5
  per-layer-type golden-fixture tests (fully offline, real trained weights
  embedded in the fixture itself, no network/unpickling needed),
  unpickled-model structural-digest match, end-to-end activation match
  (measured worst case 4 ULP, `float32` view-as-`int32` bit-pattern
  distance, asserted at a 16-ULP margin), exact decoded-label match, and a
  cross-BLAS exactness proof against the reference venv (zero differing
  elements). Plus 4 new `tests/test_fixtures_exist.py` checks (2 new test
  functions + 2 new parametrized fixture-file cases) and
  `tools/generate_key_fixtures.py` (the fixture-generation script,
  reference-venv-only, no network needed since `key_cnn.pkl` is already
  vendored as that venv's package data). 13 new tests total; full offline
  suite now 98 passed, 1 skipped, 14 deselected (was 88/1/11).

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
