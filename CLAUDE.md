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
  - **4a status: DONE (2026-07-12).** Ported exactly the 5 layer classes
    the 4.0 audit predicted (`ConvolutionalLayer`, `MaxPoolLayer`,
    `BatchNormLayer`, `PadLayer`, `AverageLayer`) -- `key_cnn.pkl`'s
    real `pickletools`-walked globals confirmed no `ReshapeLayer`/
    `TransposeLayer`/`StrideLayer` are needed for THIS model (those stay
    TO-PORT for 4b/4e's onset/notes CNNs, which do reference them), and
    `elu`/`linear` were already ported in Phase 2. Added 5
    `unpickle.py` allowlist entries (minimal, explicit, one per class,
    no `find_class` widening) and `models.py`'s `key_cnn()` registry
    entry (`key/2018/key_cnn.pkl`, sha256
    `c58ba553be1048877662a663a2670c0051b3c2c66d109b6042ba722ed0bfc7a6`,
    cross-checked identical against a live `raw.githubusercontent.com/
    CPJKU/madmom_models` download AND the local `../madmom-upstream`
    submodule checkout -- network was available, both checked, not just
    one). New `madmom_infer/features/key.py`:
    `CNNKeyRecognitionProcessor` end-to-end (audio in, 24-class
    major/minor key probabilities + decoded label out), composing
    `FilteredSpectrogramProcessor` -> `LogarithmicSpectrogramProcessor`
    instead of upstream's fused `LogarithmicFilteredSpectrogramProcessor`
    convenience class (numerically identical, same pattern
    `RNNDownBeatProcessor` already established). **Faithfulness proof:
    PASSED** -- `tests/test_key.py::
    test_full_pipeline_is_exact_under_original_blas` (this port's own
    `CNNKeyRecognitionProcessor`, run under the reference venv against
    the local `key_cnn.pkl`) reproduces real madmom's activations AND
    decoded key labels with **zero differing elements**, for all 3
    usable 44.1kHz test-wav cases. In-process (differing-BLAS-build) ULP
    drift measured at up to 4 ULP (`float32` view-as-`int32` bit-pattern
    distance) end-to-end -- tests assert a 16-ULP margin (4x observed,
    matching this repo's existing margin convention); the decoded key
    label is EXACT in every case despite that drift. New
    `tools/generate_key_fixtures.py` records per-layer-type golden
    (input, output, real weights) fixtures for all 5 new layer classes
    (fully self-contained -- offline tests reconstruct this port's own
    layer instances directly from the fixture, no unpickling or network
    needed) plus the unpickled structural digest and end-to-end
    activations/labels. 13 new tests (`tests/test_key.py`: 6 offline + 3
    network; `tests/test_fixtures_exist.py`: 4 more). Full offline suite:
    98 passed, 1 skipped, 14 deselected (was 88/1/11 after 4.0); network
    suite: 14 passed, 1 skipped, 98 deselected, all green.
  - **4b status: DONE (2026-07-12).** Ported the complete spectral-flux DSP
    onset family, both NN-based onset activation processors, and
    peak-picking, matching the wave-plan bullet exactly plus the audit
    table's fuller enumeration:
    - `audio/stft.py`: `phase`, `Phase`, `local_group_delay`/`lgd`,
      `LocalGroupDelay`/`LGD`, `ShortTimeFourierTransform.phase()`. Found
      and faithfully reproduced a real upstream bug: `LocalGroupDelay.
      __new__`'s `isinstance(stft, Phase)` check references an undefined
      name that Python resolves to the module-level `stft()` FUNCTION (not
      the `phase` argument), so the "reuse an existing Phase" branch is
      never actually taken -- it always rebuilds. Confirmed by reading the
      exact upstream line, not guessed.
    - `audio/filters.py`: `hz2mel`, `mel2hz`, `mel_frequencies`,
      `MelFilterbank` -- **pulled forward from 4g** (the 4.0 audit itself
      had already flagged, in the same table row, that `MelFilterbank`
      feeds `CNNOnsetProcessor`'s 80-band mel input; porting it in 4g would
      have blocked 4b's own CNN onset target, so it moved here). Audit
      table row updated to PORTED (4b); 4g's MFCC work reuses this.
    - `audio/spectrogram.py`: `SuperFluxProcessor`,
      `Spectrogram.diff()`/`.filter()`/`.log()` convenience methods (the
      onset functions call `spectrogram.diff(...)` directly, matching
      upstream's own `np.ndarray`-subclass API), `SpectrogramProcessor.
      __init__(self, **kwargs): pass` (matches upstream exactly -- needed
      so `SuperFluxProcessor` can construct it with forwarded kwargs; the
      base `Processor`/`object.__init__` has no catch-all).
    - `audio/signal.py`: `smooth()` (needed by `peak_picking`) --
      **correction to a Phase-1 audit-table overstatement**: the 4.0 (and
      earlier) audit table listed `smooth` (and several other names --
      `attenuate`, `rescale`, `resample`, `root_mean_square`,
      `sound_pressure_level`, `energy`, `trim`, `load_audio_file`,
      `load_wave_file`, `write_wave_file`, `Stream`, `LoadAudioFileError`)
      as already PORTED; `smooth` demonstrably was not (grepped, absent).
      Added `smooth` for real this wave; the audit table's row for the
      rest is downgraded to TO-VERIFY (flagged, not silently left wrong --
      full re-audit of that claim is out of scope for 4b). Also added a
      `**kwargs` catch-all to `FramedSignalProcessor.__init__` (matching
      upstream's own signature exactly -- this port had dropped it),
      needed for `SpectralOnsetProcessor`'s literal blind-kwargs-forwarding
      design to work against this project's stricter processor signatures.
    - `madmom_infer/utils.py` (**new module**): `segment_axis` (narrow
      carve-out -- only the `axis=0`/`end='cut'` case `StrideLayer` ever
      uses, implemented via `numpy.lib.stride_tricks.sliding_window_view`,
      not upstream's full generality) and `combine_events` (full port, all
      3 modes). Corrects the 4.0 audit's `utils/*` EXCLUDE row: these two
      functions turned out to be real, non-speculative dependencies, not
      speculative ones -- `utils/*` otherwise stays excluded.
    - `ml/nn/layers.py`: `StrideLayer` -- confirmed by `pickletools`-walking
      `onsets_cnn.pkl` directly to be the ONLY new layer class it needs
      beyond 4a's already-ported CNN set (`ConvolutionalLayer`,
      `MaxPoolLayer`, `BatchNormLayer`) plus Phase-2's `FeedForwardLayer` --
      `ReshapeLayer`/`TransposeLayer` are confirmed NOT needed here (they
      stay TO-PORT for 4e's `notes_cnn.pkl`, per the original audit
      prediction, now verified rather than assumed).
    - `ml/nn/unpickle.py`: 2 new `ALLOWED_GLOBALS` entries --
      `madmom.ml.nn.layers.StrideLayer` and `numpy.core.multiarray.scalar`
      (found by the same `onsets_cnn.pkl` `pickletools` walk: its
      `BatchNormLayer.beta`/`.gamma` are pickled as bare numpy 0-d scalars,
      not 1-element arrays, needing this extra numpy reconstruction hook).
    - `madmom_infer/models.py`: `onsets_rnn()`/`ONSETS_RNN` (8 files),
      `onsets_brnn()`/`ONSETS_BRNN` (8 files), `onsets_cnn()`/`ONSETS_CNN`
      (1 file) -- all 17 sha256s computed from the local `../madmom-upstream`
      submodule checkout AND cross-checked byte-for-byte against fresh
      `raw.githubusercontent.com/CPJKU/madmom_models` downloads (network
      was available but slow/flaky in this sandbox -- needed `curl --retry`,
      all 17 eventually succeeded and matched). `ONSETS_BRNN_PP`
      (`onsets/2014/*`) is real `package_data`-shipped but has no registry
      entry -- only `bin/SuperFluxNN` (excluded) loads it, no processor
      this project ports needs it.
    - New `madmom_infer/features/onsets.py`: the complete DSP function
      family (`wrap_to_pi`, `correlation_diff`, `high_frequency_content`,
      `spectral_diff`, `spectral_flux`, `superflux`, `complex_flux`,
      `modified_kullback_leibler`, `_phase_deviation`, `phase_deviation`,
      `weighted_phase_deviation`, `normalized_weighted_phase_deviation`,
      `_complex_domain`, `complex_domain`, `rectified_complex_domain`),
      `SpectralOnsetProcessor`, `RNNOnsetProcessor` (`online=True`/`False`,
      both fully supported and offline-compatible), `CNNOnsetProcessor`,
      `peak_picking`, `OnsetPeakPickingProcessor` (offline-only, plain
      `Processor`, `OnlineProcessor`'s streaming machinery dropped per this
      project's stated permanent exclusion). Found and faithfully
      reproduced two more real upstream quirks: `correlation_diff` crashes
      under Python 3 in REAL madmom too (`len(c) / 2` used as a slice index
      -- confirmed by running real madmom's own function against the
      reference venv, not merely inspecting source), and
      `SpectralOnsetProcessor.__init__` only appends `onset_method` to its
      processor chain when it had to look it up from a string -- an
      already-callable `onset_method` argument is silently NOT added to
      the pipeline (looks like an oversight, ported as-is). Also found and
      FIXED (not upstream's fault, a genuine numpy-2.x-vs-1.23.5
      divergence, same class as docs/DESIGN.md C.1):
      `normalized_weighted_phase_deviation`'s `epsilon` addition silently
      upcast `float32` to `float64` under numpy >= 2.0's NEP 50 strict
      scalar-promotion rules (since `EPSILON = np.spacing(1)` is a genuine
      numpy `float64` scalar, not a plain Python float) -- an explicit
      `.astype(np.float32)` reproduces real madmom's actual dtype on every
      numpy version, confirmed by cross-BLAS test passing with zero
      differing elements INCLUDING dtype.
    - New `tools/generate_onset_fixtures.py`: per-DSP-function golden
      OUTPUT fixtures (inputs are NOT serialized -- deterministic given the
      shared wav + this project's own already-golden-fixture-proven Phase-1
      DSP chain, so offline tests just rebuild the input and compare only
      the new function's output), `StrideLayer`'s self-contained
      (input, output, `block_size`) fixture, `onsets_rnn_1`/`onsets_brnn_1`/
      `onsets_cnn` structural digests, and `RNNOnsetProcessor`/
      `CNNOnsetProcessor` end-to-end activations + `OnsetPeakPickingProcessor`
      decoded onset times for all 3 44.1kHz test-wav cases. Same
      "shared-instance-in-order" caching-gotcha discipline as
      `test_downbeats_rnn.py` (both `RNNOnsetProcessor`/`CNNOnsetProcessor`
      build one `ShortTimeFourierTransformProcessor`/
      `FilteredSpectrogramProcessor` per frame-size branch and reuse them
      across calls).
    - **Faithfulness proof: PASSED.**
      `tests/test_onsets.py::test_full_pipeline_is_exact_under_original_blas`
      reproduces real madmom's `RNNOnsetProcessor`(online=False/True) +
      `CNNOnsetProcessor` + `OnsetPeakPickingProcessor` activations AND
      decoded onset times with **zero differing elements**, for all 3
      44.1kHz test-wav cases, all 3 model families. In-process
      (differing-BLAS-build) ULP drift measured at up to 17 ULP for the
      pure-DSP functions (tests assert a 64-ULP margin, ~4x observed) and
      up to 62 ULP for the RNN/BRNN/CNN activations (tests assert a
      256-ULP margin, ~4x observed, same order of magnitude as
      `test_downbeats_rnn.py`'s 512 for its own bigger 8-network BLSTM
      ensemble) -- decoded onset TIMES are EXACT in every case despite
      that drift.
    - 31 new tests total (`tests/test_onsets.py`: 20 offline + 6 network;
      `tests/test_fixtures_exist.py`: 5 more). Full offline suite: 123
      passed, 1 skipped, 20 deselected (was 98/1/14 after 4a); network
      suite: 20 passed, 1 skipped, 123 deselected, all green.

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
| `audio/signal.py` | `Signal`, `SignalProcessor`, `FramedSignal`, `FramedSignalProcessor`, `remix`, `normalize`, `adjust_gain`, `signal_frame` | PORTED | -- | Phase 1, complete |
| `audio/signal.py` | `smooth` | PORTED (4b) | -- | needed by `features/onsets.py`'s `peak_picking`; this row previously (Phase 1) claimed it as already PORTED -- it was not actually present in the module until this wave, correcting that overstatement here |
| `audio/signal.py` | `Stream`, `LoadAudioFileError`, `attenuate`, `rescale`, `resample`, `root_mean_square`, `sound_pressure_level`, `energy`, `trim`, `load_audio_file`, `load_wave_file` (public), `write_wave_file` | TO-VERIFY | -- | this row's Phase-1 entry claimed these as PORTED; empirically NOT found in `audio/signal.py` while porting 4b's `smooth` (only a private `_load_wave_file` helper exists) -- flagged rather than silently left overstated, but a full re-audit of Phase 1's own completeness is out of scope for 4b; port on demand if/when a TO-PORT processor is found to need one (`resample` in particular is a known, deliberate Phase-1 gap -- ffmpeg-backed, no project dependency, see that module's header) |
| `audio/filters.py` | `Filterbank`, `LogarithmicFilterbank`, `log_frequencies`, `frequencies2bins`, `bins2frequencies`, freq-conversion helpers (`hz2mel` etc.) | PORTED | -- | Phase 1 |
| `audio/filters.py` | `MelFilterbank` | PORTED (4b) | -- | originally slotted for 4g (`cepstrogram.py` MFCC), pulled forward -- also feeds `CNNOnsetProcessor`'s 80-band mel input, which is in 4b's own scope; 4g's MFCC work reuses this instead of re-porting |
| `audio/filters.py` | `BarkFilterbank`, `RectangularFilter`, `RectangularFilterbank` | TO-PORT (4f) | -- | feeds `MultiBandSpectrogramProcessor`, used by `PatternTrackingProcessor` |
| `audio/filters.py` | `PitchClassProfileFilterbank`, `HarmonicPitchClassProfileFilterbank`, `SimpleChromaFilterbank`, `SemitoneBandpassFilterbank` | TO-PORT (4d, scope addition -- see corrections above) | -- | feed `audio/chroma.py`'s classic (non-DNN) chroma path |
| `audio/filters.py` | `HarmonicFilterbank` | TO-PORT (4g) | -- | used by `SemitoneBandpassSpectrogram`/harmonic feature paths; low priority, no processor in the named waves depends on it alone |
| `audio/stft.py` | `ShortTimeFourierTransform`, `ShortTimeFourierTransformProcessor`, `stft`, `fft_frequencies` | PORTED | -- | Phase 1 |
| `audio/stft.py` | `Phase`, `LocalGroupDelay`/`LGD`, `phase`, `local_group_delay`, `lgd` | PORTED (4b) | -- | feeds onset phase-deviation family; `LocalGroupDelay` reproduces a real upstream bug on purpose (`__new__` checks `isinstance(stft, Phase)` where `stft` is an undefined name resolving to the module's own `stft()` function, so it always rebuilds rather than reusing an existing `Phase`) -- see `audio/stft.py`'s module header |
| `audio/stft.py` | `rfft_builder` | EXCLUDE | -- | `pyfftw` acceleration hook, not a project dependency (see `audio/stft.py`'s Phase-1 header) |
| `audio/spectrogram.py` | `Spectrogram`, `SpectrogramProcessor`, `FilteredSpectrogram(Processor)`, `LogarithmicSpectrogram(Processor)`, `SpectrogramDifference(Processor)` | PORTED | -- | Phase 1 |
| `audio/spectrogram.py` | `SuperFluxProcessor` | PORTED (4b) | -- | onset family |
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
| `ml/nn/layers.py` | `ConvolutionalLayer`, `MaxPoolLayer`, `BatchNormLayer`, `PadLayer`, `AverageLayer` | PORTED (4a) | -- | confirmed pickletools-walked as exactly what `key_cnn.pkl` (`AverageLayer`,`BatchNormLayer`,`ConvolutionalLayer`,`MaxPoolLayer`,`PadLayer`,`elu`,`linear`) references; `onsets_cnn.pkl`, `notes_cnn*.pkl`, `chords_cnnfeat.pkl` also need this same set (reused by 4b/4d/4e, not re-ported) |
| `ml/nn/layers.py` | `GRULayer`, `GRUCell` | TO-PORT (tentative 4c, scope addition -- see corrections above) | -- | `downbeats_bgru_*.pkl` (12 files) reference these; no wave currently plans them |
| `ml/nn/layers.py` | `StrideLayer` | PORTED (4b) | -- | `onsets_cnn.pkl` references it (confirmed by `pickletools`); needs `utils.segment_axis` (see `utils/*` row below) |
| `ml/nn/layers.py` | `ReshapeLayer`, `TransposeLayer` | TO-PORT (4e, alongside the CNN infra that needs them) | -- | `notes_cnn.pkl` needs Reshape+Transpose; confirmed by 4b's own `pickletools` walk of `onsets_cnn.pkl` that it does NOT need either of these (only `StrideLayer`, above) |
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
| `features/onsets.py` | `SpectralOnsetProcessor`, `spectral_diff`, `spectral_flux`, `superflux`, `complex_flux`, `complex_domain`, `rectified_complex_domain`, `high_frequency_content`, `modified_kullback_leibler`, `phase_deviation`, `weighted_phase_deviation`, `normalized_weighted_phase_deviation`, `correlation_diff`, `wrap_to_pi`, `peak_picking`, `OnsetPeakPickingProcessor` | PORTED (4b) | -- | pure DSP, no NN weights; `correlation_diff` is a faithful port of a function real madmom itself crashes on under Python 3 (confirmed empirically against the reference venv) -- ported bug-for-bug, pinned by a `pytest.raises(TypeError)` test, not a golden output; `OnsetPeakPickingProcessor` is offline-only (no `OnlineProcessor`, a stated permanent exclusion) |
| `features/onsets.py` | `RNNOnsetProcessor` | PORTED (4b) | `ONSETS_RNN`, `ONSETS_BRNN` | pickle refs: `FeedForwardLayer`/`RecurrentLayer`/`BidirectionalLayer` -- all already PORTED (Phase 2), no new layer classes; `online=True` (`ONSETS_RNN`) IS supported (unlike `OnsetPeakPickingProcessor`'s online mode) -- it only selects different pretrained weights/frame sizes, not actual streaming; `ONSETS_BRNN_PP` has no registry entry (only loaded by the excluded `bin/SuperFluxNN` CLI script, no processor this project ports needs it) |
| `features/onsets.py` | `CNNOnsetProcessor` | PORTED (4b, reuses 4a's conv layers) | `ONSETS_CNN` | pickle refs: `ConvolutionalLayer`,`MaxPoolLayer`,`BatchNormLayer`,`FeedForwardLayer`,`StrideLayer` (all PORTED, `StrideLayer` new in 4b) + `MelFilterbank` input (pulled forward from 4g into 4b, see `audio/filters.py` row) |
| `features/key.py` | `CNNKeyRecognitionProcessor`, `key_prediction_to_label`, `add_axis` | PORTED (4a) | `KEY_CNN` = `key/2018/key_cnn.pkl` | pickle refs confirmed above; cross-BLAS-proven exact (`tests/test_key.py`) |
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
| `utils/__init__.py` | `segment_axis`, `combine_events` | PORTED (4b, narrow carve-out -- `madmom_infer/utils.py`, NOT a general `utils/*` port) | -- | correction to the row above: 4b found two real, non-speculative dependencies -- `StrideLayer` (`ml/nn/layers.py`) calls `segment_axis` (this port implements only its `axis=0`/`end='cut'` case, the only one `StrideLayer` ever uses, NOT upstream's full generality), `OnsetPeakPickingProcessor` (`features/onsets.py`) calls `combine_events` (ported in full, all 3 `combine` modes, cheap) |

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
