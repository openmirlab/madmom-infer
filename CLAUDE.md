# madmom-infer -- CLAUDE.md

<!-- shape:dev-workflow start -->
## Dev workflow

This project is driven by the **shape / nav** skill workflow. The
planning board lives in `docs/blueprints/`.

| You want to... | Verb |
|---|---|
| Decide what to work on next / refresh the board | `/shape:align` -> `docs/blueprints/plan.md` |
| See the board rendered visually | `/shape:mockup` -> an on-demand board snapshot |
| Scope a feature against the actual code | `/nav:plan` -> `docs/blueprints/plans/` |
| Implement a small decided change | `/nav:do` |
| Drive the in-progress board to done | `/shape:build` |
| Behaviour-preserving structural move | `/nav:refactor` |
| Re-sync file-top headers after restructuring | `/nav:sync` |
| Regenerate / render the repo map | `/nav:map` -> `docs/codebase-map/index.html` |
| Audit architecture | `/nav:audit` |

**Standing pointers:** plan board = `docs/blueprints/plan.md` (agent AND
human -- a visual view renders on demand via `/shape:mockup`, never a
standing `overview.html`) · durable why = `docs/blueprints/decisions.md`
· grounded plans = `docs/blueprints/plans/` · pre-implementation
architecture plan (mostly historical) = `docs/DESIGN.md`.

**Communication:** converse with the user in Traditional Chinese
(Taiwanese phrasing), plain and direct; keep code, identifiers, and
commit messages in English.
<!-- shape:dev-workflow end -->

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
- **Phase 4 — complete-port campaign** (started 2026-07-12, **DONE
  2026-07-13**, branch `feat/complete-port`, not yet merged -- see the "4g
  closure verdict" section below for the closing statement): port every
  remaining inference-relevant madmom
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
      reproduced two more real upstream quirks: `correlation_diff` crashed
      under Python 3 in REAL madmom too (`len(c) / 2` used as a slice index
      -- confirmed by running real madmom's own function against the
      reference venv, not merely inspecting source; **fixed in place
      2026-07-13** once this project's bug-for-bug-fidelity policy was
      reversed, see `docs/blueprints/decisions.md`'s "Fix inherited defects
      after migration" and the 4g audit-table row below), and
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
  - **4c status: DONE (2026-07-13).** Ported everything the audit table's
    `features/beats.py`/`features/tempo.py`/`audio/comb_filters.pyx` rows
    marked TO-PORT(4c), plus the GRU scope addition the 4.0 audit
    corrections flagged:
    - `madmom_infer/audio/comb_filters.py` (**new module**):
      `feed_forward_comb_filter`, `feed_backward_comb_filter` (+ 1D/2D
      helpers), `comb_filter`, `CombFilterbankProcessor`. **Faithfulness
      proof: bit-identical, not ULP-close** -- `tests/test_comb_filters.py`
      asserts `np.array_equal` directly, both in-process (this venv's
      numpy 2.4.6) AND cross-BLAS (reference venv), because neither
      function touches BLAS at all (`feed_forward` is one vectorized
      elementwise op; `feed_backward` is a scalar Python loop matching real
      madmom's own Cython loop's exact operation order) -- there is no
      summation-order non-associativity to average away. Found and
      reproduced a real upstream precision quirk, confirmed empirically:
      `feed_backward_comb_filter`'s Cython helpers declare `alpha` as a C
      `float` (32-bit) parameter, silently rounding any float64 `alpha`
      before the loop -- `feed_forward_comb_filter` has no such truncation
      (untyped `def`). Also found and FIXED a genuine numpy-2.x-vs-1.23.5
      divergence in `comb_filter`'s own per-tau dispatch (extracting
      `alpha[i]` from a numpy array is a "strong" scalar under NEP 50,
      upcasting a float32-array multiply to float64 on numpy >= 2.0 but not
      numpy < 2.0) -- fixed with an explicit `float(alpha[i])` cast, same
      class of fix as 4b's `normalized_weighted_phase_deviation`.
    - `madmom_infer/ml/nn/layers.py`: `GRUCell`, `GRULayer` -- confirmed by
      `pickletools`-walking `downbeats_bgru_{rhythmic,harmonic}_0.pkl`
      directly to be the only new layer classes the 12-file `DOWNBEATS_BGRU`
      ensemble needs (everything else -- `NeuralNetwork`, `sigmoid`/`tanh`,
      `BidirectionalLayer`, `FeedForwardLayer`, `Gate` -- already ported).
      **Found these 12 files are an OLDER pickle format** than every other
      target `.pkl` in this project: loading one with real madmom emits its
      own "please update your GRU models" `RuntimeWarning`, and their
      `pickletools` walk references 2 generic old-style-class-reconstruction
      globals (`copy_reg._reconstructor`, `__builtin__.object`) no other
      target pickle needs. Initially assumed `GRULayer.__setstate__`'s
      legacy `hid_init` -> `init` rename branch was dead code and dropped
      it; empirically WRONG (confirmed by actually loading the real files
      under the reference venv -- the rename branch fires on all 12) --
      restored it, verbatim.
    - `madmom_infer/ml/nn/unpickle.py`: 4 new `ALLOWED_GLOBALS` entries
      (`GRUCell`, `GRULayer`, `copy_reg._reconstructor`,
      `__builtin__.object`), found by the same `pickletools` walk.
    - New `madmom_infer/features/beats.py`: `RNNBeatProcessor` (online =
      `BEATS_LSTM` unidirectional / offline = `BEATS_BLSTM` bidirectional,
      same offline-compatibility shape as `RNNOnsetProcessor`),
      `DBNBeatTrackingProcessor` (beat-only, OFFLINE-ONLY -- drops
      `OnlineProcessor`'s `process_online`/`reset`/visualisation state, same
      precedent as `OnsetPeakPickingProcessor`; reuses `beats_hmm.py`'s
      existing `BeatStateSpace`/`BeatTransitionModel`/
      `RNNBeatTrackingObservationModel`, Phase 2), `MultiModelSelectionProcessor`.
      **Found and fixed a genuine Phase-2 latent bug this wave surfaced**:
      `RNNBeatTrackingObservationModel.log_densities`
      (`features/beats_hmm.py`) called `np.asarray(observations, ndmin=1)`
      -- not valid on ANY numpy version (`asarray` has no `ndmin` keyword,
      confirmed) -- a Phase-2-era numpy-2.x-compat comment wrongly claimed
      this was the fix and wrongly claimed `RNNDownBeatTrackingObservationModel`
      inherits this method (it defines its own, which is why
      `DBNDownBeatTrackingProcessor` never hit the bug); fixed as
      `np.array(observations, ndmin=1)`, both claims corrected in that
      module's header. **Found, not silently dropped -- an apparent
      audit-table gap**: `BeatTrackingProcessor`/`BeatDetectionProcessor`/
      `detect_beats` are real upstream classes the audit table's EXCLUDE
      row incorrectly grouped with `TCNBeatProcessor` as "TCN-specific"
      (`detect_beats` is actually `BeatTrackingProcessor`'s own helper,
      unrelated to TCN) -- flagged in the audit table above, deferred (no
      4c target needs them; `CRFBeatDetectionProcessor`, 4f, will need
      `BeatTrackingProcessor` as a base class, so 4f inherits this gap).
    - New `madmom_infer/features/tempo.py`: `smooth_histogram`,
      `interval_histogram_acf`, `interval_histogram_comb`,
      `dominant_interval`, `detect_tempo`, `TempoHistogramProcessor`,
      `ACFTempoHistogramProcessor`, `CombFilterTempoHistogramProcessor`,
      `DBNTempoHistogramProcessor` (reuses `features/beats.py`'s
      `DBNBeatTrackingProcessor`), `TempoEstimationProcessor` -- all
      OFFLINE-ONLY, same precedent as `DBNBeatTrackingProcessor`. Every
      test in `tests/test_tempo.py` (including the cross-BLAS one) runs
      fully offline -- no model download, no unpickling, since tempo
      estimation is pure numpy/scipy given an activation array.
    - `madmom_infer/features/downbeats.py`: `SyncronizeFeaturesProcessor`
      (pure numpy, **bit-identical**, `np.array_equal` both in-process and
      cross-BLAS) and `RNNBarProcessor` (ported verbatim). **`RNNBarProcessor`
      cannot be instantiated end-to-end from raw audio this wave** -- its
      `__init__` needs `audio/chroma.py`'s `CLPChromaProcessor` (4d, not yet
      ported), confirmed by reading `RNNBarProcessor.__init__` directly
      (`downbeats.py:965/980`); raises `ImportError` on construction,
      matching the audit table's own prediction. What IS proven bit-exact:
      the `DOWNBEATS_BGRU` `NeuralNetworkEnsemble` forward pass itself
      (`GRULayer`/`GRUCell` in context), fed real madmom's own captured
      intermediate `perc_synced`/`harm_synced` beat-synchronized features
      as a golden fixture -- `tests/test_downbeats_rnn.py::
      test_downbeats_bgru_ensembles_are_exact_under_original_blas`
      reproduces real madmom's `perc_nn`/`harm_nn` outputs with **zero
      differing elements**. Found one more genuine numpy-2.x-vs-1.23.5
      divergence surfaced by this particular fixture (a degenerate
      single-beat-window case, `mono_44100.wav` being short): `madmom_infer/
      ml/nn/__init__.py`'s `average_predictions`, averaging a list of
      0-DIMENSIONAL float32 ensemble outputs, stayed float32 on numpy >=
      2.0 (NEP 50) but real madmom's own `sum(pred) / len(pred)` upcasts to
      float64 on numpy < 2.0 (0-d "scalar-kind" arrays follow different
      value-based-casting rules than N-d ones) -- fixed with an explicit
      branch reproducing the old (real-madmom-recorded) dtype on every
      numpy version; every OTHER model family in this project (N-d
      predictions, including the already-shipped `DOWNBEATS_BLSTM`
      ensemble) was already unaffected on both numpy versions, confirmed by
      the full suite staying green after the fix.
    - `madmom_infer/models.py`: `beats_lstm()`/`BEATS_LSTM` (8 files),
      `beats_blstm()`/`BEATS_BLSTM` (8 files), `downbeats_bgru_rhythmic()`/
      `downbeats_bgru_harmonic()`/`downbeats_bgru()`/`DOWNBEATS_BGRU` (12
      files) -- 28 sha256s computed from the local `../madmom-upstream`
      submodule checkout AND cross-checked byte-for-byte against fresh
      `raw.githubusercontent.com/CPJKU/madmom_models` downloads (network
      was available, all 28 succeeded and matched, confirmed 2026-07-13).
    - New `tools/generate_beat_tempo_fixtures.py`: comb-filter direct
      function-level fixtures (fed a REAL beat activation function, not
      synthetic noise), `beats_lstm_1`/`beats_blstm_1`/
      `downbeats_bgru_{rhythmic,harmonic}_0` structural digests,
      `RNNBeatProcessor`/`DBNBeatTrackingProcessor` end-to-end activations +
      decoded beat times (all 3 44.1kHz cases, shared-instance-in-order
      discipline), `MultiModelSelectionProcessor`'s self-contained
      selection fixture, per-mode tempo histogram + tempi fixtures
      (self-contained -- records the input activation array too, so
      `tests/test_tempo.py` needs no model/network at all),
      `SyncronizeFeaturesProcessor`'s self-contained fixture, and
      `RNNBarProcessor`'s GRU-ensemble intermediate-feature fixture (real
      beat times from real madmom's own `RNNBeatProcessor` ->
      `DBNBeatTrackingProcessor`, not hand-picked).
    - **Faithfulness proof: PASSED.** `tests/test_beats.py::
      test_full_pipeline_is_exact_under_original_blas` reproduces real
      madmom's `RNNBeatProcessor`(online=False/True) +
      `DBNBeatTrackingProcessor` activations AND decoded beat times with
      **zero differing elements**, for all 3 44.1kHz test-wav cases, both
      model families. In-process (differing-BLAS-build) ULP drift for the
      LSTM/BLSTM activations measured well within the existing 512-ULP
      margin convention (same order of magnitude as
      `test_downbeats_rnn.py`'s own bigger ensemble) -- decoded beat TIMES
      are EXACT in every case despite that drift. Comb filters and tempo
      histograms are bit-identical with NO tolerance at all (see above).
    - 66 new tests total (`tests/test_comb_filters.py`: 17;
      `tests/test_beats.py`: 15; `tests/test_tempo.py`: 15;
      `tests/test_downbeats_rnn.py`: +13; `tests/test_fixtures_exist.py`:
      +6). Full offline suite: 174 passed, 1 skipped, 25 deselected (was
      123/1/20 after 4b); network suite: 25 passed, 1 skipped, 174
      deselected, all green.
  - **4d status: DONE (2026-07-13).** Ported everything the audit table's
    `ml/crf.py`/chroma/chords rows marked TO-PORT(4d), plus the classic
    (non-DNN) chroma scope addition the 4.0 audit corrections flagged, and
    closed the loop 4c left open on `RNNBarProcessor`:
    - `madmom_infer/ml/crf.py` (**new module**): `ConditionalRandomField`
      (pure-numpy matrix-formulation Viterbi decode, forward-inference
      only, verbatim port). Added a `.load()` classmethod (not in upstream,
      which inherits `Processor.load`'s bare `pickle.load`) delegating to
      `unpickle.load_model` -- same restricted-unpickling convention as
      `NeuralNetwork.load`/`NeuralNetworkEnsemble.load`. **Pickle
      introspection finding (pickletools-walked all 4 target `.pkl` files
      directly, not guessed)**: `chroma/2016/chroma_dnn.pkl` references
      only already-ported globals (`NeuralNetwork`, `FeedForwardLayer`,
      `relu`, `sigmoid`); `chords/2016/chords_cnnfeat.pkl` references only
      wave 4a's already-ported CNN layer set (`ConvolutionalLayer`,
      `BatchNormLayer`, `MaxPoolLayer`, `linear`, `relu`); `chords/2016/
      chords_dccrf.pkl` and `chords/2016/chords_cnncrf.pkl` each need
      exactly ONE new global, `madmom.ml.crf.ConditionalRandomField` --
      confirming the audit table's own "pickle has no NN globals -- CRF-
      only" prediction for both CRF-only pickles. Both CRF pickles restore
      via `NEWOBJ` + direct `__dict__` update under this class's own
      `__init__` attribute names (`pi`/`tau`/`c`/`A`/`W`) -- no
      `__getstate__`/`__setstate__` needed, same "attribute names, not
      constructor-perfect `__init__`s" shape as `ml/nn/layers.py`'s
      pickled layer classes.
    - `madmom_infer/ml/nn/unpickle.py`: 1 new `ALLOWED_GLOBALS` entry
      (`madmom.ml.crf.ConditionalRandomField`), found by the same
      `pickletools` walk.
    - `madmom_infer/audio/filters.py`: `hz2midi`, `midi2hz`,
      `semitone_frequencies` (verbatim ports), `PitchClassProfileFilterbank`/
      `HarmonicPitchClassProfileFilterbank` (composition ports, built on
      this project's own `Filterbank` base rather than upstream's ndarray-
      view `__new__`), `SemitoneBandpassFilterbank` (own composition class,
      not a `Filterbank` subclass -- it's a time-domain IIR filterbank,
      matching upstream's own design). **Found and ported faithfully, not
      silently completed**: `SimpleChromaFilterbank`'s upstream `__new__`
      unconditionally `raise NotImplementedError`s before any of its own
      (dead, TODO-commented) filterbank-construction code runs -- confirmed
      by reading `filters.py:1340-1341` directly. This port reproduces that
      exact not-actually-implemented state rather than finishing code
      upstream itself never enabled; `HarmonicFilterbank` stays TO-PORT(4g)
      as previously audited (no target this wave needs it).
    - `madmom_infer/audio/signal.py`: `resample()` -- **a real, load-bearing
      policy correction to this project's "no ffmpeg dependency" stance
      (Phase 1 through 4c), not a silent reversal**. `SemitoneBandpassFilterbank`
      filters each of its ~78 semitone bands at ONE of 3 FIXED sample rates
      (882/4410/22050 Hz), all three unconditionally different from this
      project's 44100 Hz input convention -- resampling is unavoidable on
      every single call, not an optional convenience (unlike the narrower
      `utils.segment_axis` carve-out precedent). Confirmed the `ffmpeg`
      system binary is present in this sandbox (`/usr/bin/ffmpeg`) and
      implemented `resample()` as a narrow ffmpeg-subprocess call -- only
      the exact shape `SemitoneBandpassFilterbank`'s caller needs (an
      already-loaded `Signal`, unchanged `dtype`/`num_channels`), not
      upstream's full `_ffmpeg_call` generality -- shelling out with the
      same command shape real madmom's own `_ffmpeg_call`/`decode_to_pipe`
      build for a `Signal` input. **Faithfulness proof: bit-identical, not
      ULP-close** -- `resample()`'s output matches real madmom's own
      `resample()` output via `np.array_equal` for all 3 fixed target rates
      tested (882/4410/22050 Hz) on `mono_44100.wav`, because both sides
      invoke the literal SAME system `ffmpeg` binary with the literal same
      arguments (not a reimplementation of ffmpeg's resampling filter).
    - `madmom_infer/audio/spectrogram.py`: `SemitoneBandpassSpectrogram` --
      own composition class (NOT a `FilteredSpectrogram` subclass: no STFT
      stage at all, `scipy.signal.filtfilt`-applied time-domain IIR
      filtering instead of `np.dot`-against-a-matrix). **Faithfulness
      finding, measured not assumed**: this class does NOT reproduce real
      madmom bit-for-bit when the two sides run under DIFFERENT scipy
      versions (this project's dev venv: scipy 1.17.1; the reference venv
      that recorded fixtures: scipy 1.15.3) -- measured up to ~1e-5
      absolute difference (on a data range roughly 0-36) across the 3
      usable test-wav cases, root-caused to `scipy.signal.filtfilt`'s
      recursive (IIR) nature amplifying tiny per-scipy-version `ellip()`
      filter-coefficient differences over ~1.5s of audio, NOT a bug in this
      port (confirmed: `resample()` itself, the other new scipy-touching
      piece, IS bit-identical across the same two environments, see above
      -- isolating the divergence to `filtfilt`/`ellip` specifically).
      `tests/test_chroma.py` documents and asserts this measured tolerance
      (`atol=1e-4`, ~10x observed) rather than claiming an exactness that
      doesn't hold across scipy versions -- this is the ONE non-exact
      numerical claim this wave makes for a pure-DSP (no NN weights)
      module, and it is stated plainly, not buried.
    - `madmom_infer/audio/chroma.py` (**new module**): `PitchClassProfile`/
      `HarmonicPitchClassProfile` (composition subclasses of `audio/
      spectrogram.py`'s `Spectrogram`, matching this project's own
      composition-not-ndarray-subclass convention rather than upstream's
      `FilteredSpectrogram`-via-`__new__`/`__array_finalize__` hierarchy --
      `fref=None`'s "auto-estimate via `Spectrogram.tuning_frequency()`"
      branch raises `NotImplementedError` rather than silently mis-behaving,
      since `tuning_frequency()` itself is a documented, still-not-ported
      gap), `DeepChromaProcessor` (composes the same "`FilteredSpectrogramProcessor`
      -> `LogarithmicSpectrogramProcessor`" two-stage split every other
      end-to-end processor in this project uses instead of upstream's fused
      `LogarithmicFilteredSpectrogramProcessor`, plus one composition
      wrinkle: an `np.asarray` stage inserted before re-wrapping the
      filtered-log-spectrogram output as a fresh `Signal`, since this
      project's `LogarithmicSpectrogram` isn't an `np.ndarray` subclass the
      way upstream's is), `CLPChroma`/`CLPChromaProcessor` (own composition
      class, needs `SemitoneBandpassSpectrogram` above). **Found and fixed
      a genuine latent bug in wave 4c's own `SyncronizeFeaturesProcessor`**
      (`features/downbeats.py`), surfaced only now that `RNNBarProcessor`
      can actually be exercised end-to-end for the first time: it called
      `features.T` directly (`features` being, in real use, one of this
      project's own composition-style spectrogram objects, e.g.
      `SpectrogramDifference`/`CLPChroma`) -- works on upstream's
      `np.ndarray`-subclass spectrograms (`.T` comes free), raises
      `AttributeError` on this project's composition ones (no `.T`
      attribute defined). Fixed with an explicit `np.asarray(features).T`;
      wave 4c's own test of this function never caught it because it fed a
      raw, already-captured ndarray fixture, never a live composition
      object -- exactly the kind of gap only true end-to-end exercise
      surfaces, confirmed empirically (reproduced the `AttributeError`
      before the fix, confirmed it's gone after).
    - `madmom_infer/features/chords.py` (**new module**):
      `majmin_targets_to_chord_labels` (verbatim port; `SEGMENT_DTYPE`
      inlined directly rather than imported from a ported `io.*` package,
      which stays a permanent EXCLUDE), `DeepChromaChordRecognitionProcessor`,
      `CNNChordFeatureProcessor` (added an `nn_file=` override, not in
      upstream, purely for testability -- matches the `nn_files=`/`models=`
      override convention `CNNKeyRecognitionProcessor`/`DeepChromaProcessor`
      already establish), `CRFChordRecognitionProcessor`. **Confirmed by
      reading `madmom-upstream/madmom/features/chords.py` directly, not
      assumed**: NEITHER chord-recognition path touches `CLPChroma` at all
      (`DeepChromaChordRecognitionProcessor` uses `DeepChromaProcessor`'s
      ordinary filtered-log-spectrogram frontend; `CNNChordFeatureProcessor`
      uses the same frontend directly, no chroma stage) -- so full audio-in
      chord recognition is achievable and EXACT-testable completely
      independent of `SemitoneBandpassSpectrogram`'s scipy-version
      precision caveat above.
    - `madmom_infer/models.py`: `chroma_dnn()`/`CHROMA_DNN` (1 file),
      `chords_dccrf()`/`CHORDS_DCCRF` (1 file), `chords_cnn_feat()`/
      `CHORDS_CNN_FEAT` (1 file), `chords_cfcrf()`/`CHORDS_CFCRF` (1 file,
      backed by `chords_cnncrf.pkl` -- upstream's own naming, `CF` =
      "CNN Feature", preserved not "fixed") -- 4 sha256s computed from the
      local `../madmom-upstream` submodule checkout AND cross-checked
      byte-for-byte against fresh `raw.githubusercontent.com/CPJKU/
      madmom_models` downloads (network was available, all 4 succeeded and
      matched, confirmed 2026-07-13).
    - **`RNNBarProcessor` (wave 4c, `features/downbeats.py`) is now
      instantiable AND provably correct end-to-end from raw audio**,
      closing the loop 4c's own status entry explicitly left open (its
      `__init__` needed `CLPChromaProcessor`, now ported above). New
      `tools/generate_chroma_chord_fixtures.py` records a full audio-in
      fixture (real madmom's own `RNNBeatProcessor` -> `DBNBeatTrackingProcessor`
      -> `RNNBarProcessor`, FRESH instances per case -- see that tool's
      header for why this deliberately deviates from the "shared-instance-
      in-order" discipline other waves' fixture tools use: a shared
      instance across differing-dtype wavs was found, empirically, to
      silently produce an EMPTY `perc_synced` array for the `float32_44100`
      case, a real instance-reuse caching artifact of this port's own
      composition-style stateful processors, not a fixture-vs-port
      algorithmic mismatch). **Faithfulness proof: decoded beat times
      EXACT** in every case, both in-process and cross-BLAS; decoded
      downbeat activation matches within ~4e-8 absolute (both in-process
      and cross-BLAS) -- small enough to be explained entirely by
      `CLPChroma`'s already-documented scipy-version noise above, not a
      new divergence; the `GRULayer`/`GRUCell` ensemble forward pass itself
      was already proven bit-exact independent of this in wave 4c.
    - **Faithfulness proof (chord recognition): PASSED, EXACT.**
      `tests/test_crf.py`/`tests/test_chords.py`'s cross-BLAS tests
      reproduce real madmom's CRF-decoded state sequences AND merged chord-
      segment boundaries/labels with **zero differing elements**, for both
      `DeepChromaChordRecognitionProcessor` and `CNNChordFeatureProcessor`
      + `CRFChordRecognitionProcessor`, all 3 usable 44.1kHz test-wav cases.
      `tests/test_chroma.py`'s `DeepChromaProcessor` cross-BLAS test is
      likewise **zero differing elements**; its in-process ULP drift
      measured up to 24 ULP (asserted at a 128-ULP margin, ~5x observed).
      Classic chroma (`PitchClassProfile`/`HarmonicPitchClassProfile`)
      in-process ULP drift measured up to 5 ULP (asserted at a 16-ULP
      margin, ~3x observed) -- pure linear filterbank ops on an already-
      golden-fixture-proven `Spectrogram`, same order of magnitude as prior
      waves' comparable stages.
    - 35 new tests total (`tests/test_crf.py`: 4; `tests/test_chroma.py`:
      22; `tests/test_chords.py`: 9; `tests/test_downbeats_rnn.py`: +2;
      `tests/test_fixtures_exist.py`: +7). Full offline suite: 205 passed,
      1 skipped, 40 deselected (was 174/1/25 after 4c); network suite: 40
      passed, 1 skipped, 205 deselected, all green.
  - **4e status: DONE (2026-07-13).** Ported everything the audit table's
    `features/notes.py`/`features/notes_hmm.py` rows marked TO-PORT(4e):
    - `madmom_infer/features/notes_hmm.py` (**new module**): `ADSRStateSpace`,
      `ADSRTransitionModel`, `ADSRObservationModel` -- near-line-for-line port
      on the existing Phase-1 `ml/hmm.py` `TransitionModel`/`ObservationModel`
      base classes, same shape as `features/beats_hmm.py`. Nothing here hit a
      numpy-2.x incompatibility (`ADSRObservationModel.log_densities` is
      plain `np.ones`/`np.log` on an already-2D array, unlike
      `beats_hmm.py`'s `RNNBeatTrackingObservationModel.log_densities`).
    - `madmom_infer/ml/nn/layers.py`: `ReshapeLayer`, `TransposeLayer` --
      confirmed by `pickletools`-walking all 4 target note-CNN pickles
      (`notes/2019/notes_cnn.pkl` = `NOTES_CNN`, `notes/2018/
      notes_cnn_{1,2}.pkl` = `NOTES_CNN_MIREX`, walked for completeness
      though unused by any ported processor) to be exactly the 2 new layer
      classes needed, confirming the 4.0/4b audit's own prediction.
    - **Real, load-bearing surprise -- found by actually `pickletools`-
      walking AND loading `notes_cnn.pkl` with real madmom, not guessed**:
      it does NOT pickle a bare `NeuralNetwork` the way every other target
      `.pkl` in this project does (`key_cnn.pkl`, `onsets_cnn.pkl`,
      `chords_cnnfeat.pkl`, all `downbeats_blstm_*`/`beats_*`/
      `downbeats_bgru_*`). It pickles the model's ENTIRE multi-task
      `madmom.processors.SequentialProcessor`/`ParallelProcessor` OBJECT
      GRAPH directly: `SequentialProcessor([BatchNormLayer,
      ConvolutionalLayer x3, ParallelProcessor([3x SequentialProcessor(
      ConvolutionalLayer, TransposeLayer, ReshapeLayer, FeedForwardLayer)]),
      numpy.dstack])` -- the 3 parallel branches are the note/onset/offset
      heads, `numpy.dstack` is the final multi-task merge, all baked
      straight into the pickle rather than built by
      `CNNPianoNoteProcessor.__init__` the way `CNNKeyRecognitionProcessor`
      builds its pipeline around a bare `NeuralNetwork`. This turned out to
      need ZERO new code in `madmom_infer/ml/nn/__init__.py`:
      `NeuralNetworkEnsemble.load`/`NeuralNetwork.load` were already fully
      generic (`unpickle.load_model` just returns whatever top-level object
      type the pickle actually contains, matching upstream's own
      `Processor.load`'s equally generic behavior verbatim), and
      `average_predictions` already degrades to the identity function for a
      length-1 ensemble list -- only `ml/nn/unpickle.py`'s allowlist needed
      new entries: `madmom.processors.{SequentialProcessor,
      ParallelProcessor}` (mapped to this project's own classes, which
      already support NEWOBJ+dict-restore unpickling with no changes, same
      "attribute names, not constructor-perfect `__init__`s" shape as the
      layer classes), `numpy.dstack` (two module-path spellings across the
      2019 vs. 2018 pickles: `('numpy', 'dstack')` and `('numpy.lib.
      shape_base', 'dstack')`, both the one real function), and 2
      Python-2-pickle-compat primitives real madmom's own bare `pickle.load`
      resolves transparently via `pickle._compat_pickle.NAME_MAPPING`
      (consulted automatically by the stdlib `Unpickler.find_class` for
      protocol < 4) but this project's allowlist-only `SafeUnpickler.
      find_class` does not consult at all: `_codecs.encode` (byte-payload
      reconstruction, same "safe, mechanical" category as numpy's own
      `_reconstruct`/`scalar`) and `itertools.imap` -> Python 3's builtin
      `map` (an older madmom's `ParallelProcessor.__init__` used `self.map =
      it.imap` before simplifying to `self.map = map`; inert in this port
      either way, since `madmom_infer.processors.ParallelProcessor.process`
      never reads `self.map`). Same shape of gap as 4c's
      `copy_reg._reconstructor`/`__builtin__.object` entries for the
      older-format `downbeats_bgru_*.pkl` files -- confirmed empirically
      (unpickling succeeds end-to-end, structural digest matches real
      madmom's own unpickling exactly), not assumed.
    - `madmom_infer/ml/nn/unpickle.py`: 8 new `ALLOWED_GLOBALS` entries (the
      6 above, i.e. `ReshapeLayer`/`TransposeLayer`/`SequentialProcessor`/
      `ParallelProcessor`/`numpy.dstack`/`itertools.imap`, plus
      `numpy.lib.shape_base.dstack` and `_codecs.encode`), found by the same
      `pickletools` walk.
    - New `madmom_infer/features/notes.py`: `RNNPianoNoteProcessor` (single
      `NeuralNetwork` from `NOTES_BRNN`, pickle refs confirmed
      `BidirectionalLayer`/`FeedForwardLayer`/`RecurrentLayer`, all already
      ported -- no new classes at all), `NoteOnsetPeakPickingProcessor`
      (subclasses `features/onsets.py`'s already-offline-only
      `OnsetPeakPickingProcessor`, reuses its `peak_picking` function),
      `NotePeakPickingProcessor` (upstream's own deprecated-since-0.17
      alias, ported anyway -- the audit table lists it as a real public
      class, not dead code this project gets to skip), `_cnn_pad`,
      `CNNPianoNoteProcessor` (`NeuralNetworkEnsemble.load(NOTES_CNN)` --
      see the surprise above for why this needed no new ensemble-handling
      code), `ADSRNoteTrackingProcessor` (per-pitch independent
      `HiddenMarkovModel.viterbi()` decode, `ml/hmm.py`'s Phase-1 machinery
      unmodified).
    - `madmom_infer/models.py`: `notes_brnn()`/`NOTES_BRNN` (1 file),
      `notes_cnn()`/`NOTES_CNN` (1 file) -- 2 sha256s computed from the
      local `../madmom-upstream` submodule checkout AND cross-checked
      byte-for-byte against fresh `raw.githubusercontent.com/CPJKU/
      madmom_models` downloads (network was available, both succeeded and
      matched, confirmed 2026-07-13). `NOTES_CNN_MIREX` (`notes/2018/
      notes_cnn_[12].pkl`) is real, `package_data`-shipped, and was
      `pickletools`-walked for completeness -- but, like 4b's
      `ONSETS_BRNN_PP`, no processor this project ports ever loads it
      (confirmed by reading `CNNPianoNoteProcessor.__init__` directly: it
      hardcodes `NOTES_CNN`, never `NOTES_CNN_MIREX`), so it has no
      registry entry.
    - New `tools/generate_notes_fixtures.py`: `notes_brnn.pkl`/
      `notes_cnn.pkl` structural digests (the latter a recursive digest of
      the nested `SequentialProcessor`/`ParallelProcessor` graph above),
      self-contained `ReshapeLayer`/`TransposeLayer` golden (input, output)
      fixtures (no trainable weights needed -- these layers have none),
      `RNNPianoNoteProcessor`/`CNNPianoNoteProcessor` end-to-end activations
      + decoded notes for all 3 usable 44.1kHz test-wav cases, and two
      SYNTHETIC (hand-crafted, deterministic, no real audio) fixtures for
      `ADSRNoteTrackingProcessor`/`NoteOnsetPeakPickingProcessor`'s decode
      logic. **Found and fixed a real bug in this wave's OWN first draft of
      the fixture generator, not the port**: an initial version reused one
      shared `RNNPianoNoteProcessor`/`CNNPianoNoteProcessor` instance across
      all 3 differing-dtype test wavs (same "shared-instance-in-order"
      pattern other waves' fixture tools use successfully) -- this silently
      made REAL MADMOM ITSELF produce a materially wrong `float32_44100`
      activation array (max abs diff ~0.097 against a fresh-instance
      recording of the exact same wav+weights, not BLAS-noise-scale), a
      real upstream `FilteredSpectrogramProcessor`/
      `ShortTimeFourierTransformProcessor` instance-reuse caching artifact
      (same category already documented in those modules' headers and in
      4d's `RNNBarProcessor` fixture) -- confirmed by comparing two
      independently-recorded "real madmom" outputs for the same input
      against each other, not by comparing against this port. Fixed by
      switching to fresh instances per case (matching 4d's own precedent);
      `tests/test_notes.py` uses the same fresh-per-case discipline
      throughout, including its cross-BLAS subprocess script. Also found
      (before committing the fixture, not after): the real-audio test wavs
      decode to EMPTY output from BOTH `ADSRNoteTrackingProcessor` and
      `NoteOnsetPeakPickingProcessor` on every one of the 3 cases -- a
      technically-valid but weak golden fixture (masks real bugs, as the
      caching artifact above demonstrated: the empty-decode test passed
      even while the underlying activations were badly wrong) -- so 2
      synthetic, hand-crafted activation-array fixtures (including one
      deliberately INCOMPLETE note that must be discarded under
      `complete=True`) were added specifically to exercise the decode
      logic's branches, verified against real madmom before committing.
    - **Faithfulness proof: PASSED.** `tests/test_notes.py::
      test_full_pipeline_is_exact_under_original_blas` reproduces real
      madmom's `RNNPianoNoteProcessor`/`CNNPianoNoteProcessor` activations
      AND decoded notes (peak-picked onset events, ADSR-HMM-decoded note
      segments) with **zero differing elements**, for all 3 44.1kHz
      test-wav cases, both model families -- independently confirmed for
      the synthetic ADSR fixture too
      (`test_adsr_synthetic_decode_is_exact_under_original_blas`). In-process
      (differing-BLAS-build) drift: CNN activations measured up to 247 ULP
      (asserted at a 1024-ULP margin, ~4x observed, matching this repo's
      convention); RNN activations -- a raw, near-zero-centered
      linear-layer output (NOT a bounded-[0,1] probability like every other
      model family's final activation in this project), where an ULP-view
      metric is measurably unstable that close to zero (a tiny absolute
      BLAS-noise-scale difference translates into millions of "ULPs" purely
      because the float32 exponent is small) -- measured up to ~7.15e-7
      absolute (asserted at `atol=1e-5`, ~14x observed, same "documented
      absolute tolerance instead of ULP" precedent as 4d's
      `SemitoneBandpassSpectrogram` finding, stated plainly rather than
      forcing an ULP metric where it doesn't apply).
    - 23 new tests total (`tests/test_notes.py`: 12 offline + 11 network;
      `tests/test_fixtures_exist.py`: +6). Full offline suite: 223 passed,
      1 skipped, 51 deselected (was 205/1/40 after 4d); network suite: 51
      passed, 1 skipped, 223 deselected, all green.
  - **4f status: DONE (2026-07-13).** Ported everything the audit table's
    `beats_crf.pyx`/`ml/gmm.py`/pattern-tracking rows marked TO-PORT(4f),
    plus the audit-table gap 4c itself flagged and deferred
    (`BeatTrackingProcessor`/`BeatDetectionProcessor`/`detect_beats`):
    - `madmom_infer/features/beats_crf.py` (**new module**):
      `initial_distribution`, `transition_distribution`,
      `normalisation_factors`, `best_sequence` (verbatim ports -- not
      Cython-typed in the original `.pyx`, already plain numpy/scipy) and
      `viterbi` (a real numpy translation of the typed-Cython Viterbi loop,
      same playbook as Phase 1's `hmm.pyx` port). **Two precision mistakes
      found and fixed only by fuzzing against real madmom directly, NOT by
      reading the `.pyx` source** -- confirmed empirically at every step,
      not reasoned about in isolation: (1) `viterbi.pyx`'s `cdef double
      new_prob` does NOT mean the candidate addition happens in double
      precision -- C's/Cython's usual arithmetic conversions determine an
      expression's precision from its OPERANDS (both `float`/32-bit
      memoryview reads here), not its assignment target, so the actual
      arithmetic is float32 throughout and the `double` declaration only
      widens an already-rounded result; a first attempt assumed double
      precision, matching the decoded PATH in 120/120 fuzz trials against
      the reference venv but leaving the scalar `log_prob` off by 1-2 ULP
      in ~1/3 of them. (2) Switching to float32 arithmetic alone still left
      `log_prob` wrong ~35% of the time, root-caused to a SECOND, unrelated
      mistake: `v_c[i] += activations[i] + norm_factor[i]` is a compound
      assignment that groups as `v_c[i] + (activations[i] +
      norm_factor[i])`, NOT the left-to-right `(v_c[i] + activations[i]) +
      norm_factor[i]` a naive three-term translation produces --
      floating-point addition is not associative, so the grouping is
      load-bearing. Fixing BOTH reproduced real madmom's decoded path AND
      `log_prob` bit-for-bit in 200/200+ randomized fuzz trials (plus every
      fixture case), not just the ULP-close claim an intermediate (wrong)
      attempt could support.
    - `madmom_infer/features/beats.py`: `detect_beats`,
      `BeatTrackingProcessor`, `BeatDetectionProcessor`,
      `CRFBeatDetectionProcessor` -- verbatim ports, closing 4c's own
      audit-table gap (`CRFBeatDetectionProcessor` subclasses
      `BeatTrackingProcessor` and needs `beats_crf.py`, neither of which
      existed in 4c). `CRFBeatDetectionProcessor`'s `multiprocessing.Pool`
      dispatch is dropped in favor of a plain sequential `map`, matching
      `processors.py`'s stated permanent exclusion of multiprocessing
      plumbing.
    - `madmom_infer/ml/gmm.py` (**new module**): `logsumexp`, `pinvh`,
      `log_multivariate_normal_density` (all 4 covariance-type variants),
      `GMM` (forward-inference only -- `score`/`score_samples`, no
      `fit()`, matching this project's permanent inference-only scope).
      `GMM.__setstate__` (verbatim port of upstream's legacy
      `weights_`/`means_`/`covars_` rename branch) is what makes
      unpickling `PATTERNS_BALLROOM` work at all -- **both target `.pkl`
      files are OLD-FORMAT pickles**, confirmed empirically (loading either
      with real madmom emits the "please update your GMM models"
      `UserWarning`), same finding-shape as 4c's `downbeats_bgru_*.pkl`/
      `GRULayer`. Both target files' GMMs use `covariance_type='full'`
      (confirmed by loading and inspecting directly) -- all 4 covariance
      variants ported anyway for API completeness.
    - `madmom_infer/features/beats_hmm.py`: `MultiPatternStateSpace`,
      `MultiPatternTransitionModel`, `GMMPatternTrackingObservationModel`
      -- near-line-for-line ports, the pattern-tracking HMM machinery.
    - `madmom_infer/audio/filters.py`: `bark_frequencies`,
      `bark_double_frequencies`, `BarkFilterbank`, `RectangularFilterbank`
      (+ internal `_rectangular_filters` helper). **Found by reading
      upstream directly, not assumed**: only `RectangularFilterbank` is
      actually load-bearing for `MultiBandSpectrogram`
      (`audio/spectrogram.py:1310`) -- `BarkFilterbank` is unreachable from
      any target this project ships; ported anyway (real, public, cheap,
      same "port the surface even if unreachable" precedent as
      `NOTES_CNN_MIREX`/`ONSETS_BRNN_PP`). `BarkFilterbank` inverts the
      usual `unique_bins`/`unique_filters` relationship
      (`unique_bins=not unique_filters`) -- confirmed matching upstream
      exactly, not a typo carried over.
    - `madmom_infer/audio/spectrogram.py`: `MultiBandSpectrogram`,
      `MultiBandSpectrogramProcessor` -- feeds `PatternTrackingProcessor`.
    - `madmom_infer/features/downbeats.py`: `PatternTrackingProcessor`
      (loads `PATTERNS_BALLROOM` via this project's own restricted
      `SafeUnpickler`, NOT upstream's bare `pickle.load`),
      `DBNBarTrackingProcessor` (no GMM at all -- different `beats_per_bar`
      values are treated as different "patterns" of the same
      `MultiPatternStateSpace`/`MultiPatternTransitionModel` machinery,
      reusing the already-ported `RNNBeatTrackingObservationModel`).
      **Bug found and fixed during this wave's OWN verification, not
      upstream's**: a first draft wired `DBNBarTrackingProcessor.om` to
      `RNNDownBeatTrackingObservationModel` (2D beat+downbeat observations)
      instead of the correct `RNNBeatTrackingObservationModel` (1D,
      matching upstream `downbeats.py:1110` exactly and this processor's
      own `data[:, 1]`-only usage) -- caught immediately by an end-to-end
      cross-check against real madmom (a loud `AxisError`, not a silent
      wrong answer).
    - `madmom_infer/models.py`: `patterns_ballroom()`/`PATTERNS_BALLROOM`
      (2 files, NOT neural-network weights -- each a plain dict of fitted
      `ml.gmm.GMM` instances) -- both sha256s computed from the local
      `../madmom-upstream` submodule checkout AND cross-checked
      byte-for-byte against fresh `raw.githubusercontent.com/CPJKU/
      madmom_models` downloads (network was available, both succeeded and
      matched, confirmed 2026-07-13).
    - `madmom_infer/ml/nn/unpickle.py`: 1 new `ALLOWED_GLOBALS` entry
      (`madmom.ml.gmm.GMM`), found by `pickletools`-walking both target
      `.pkl` files directly (each pickles a plain dict whose `'gmms'` list
      elements are `GMM` instances, restored via the standard
      `pickle.Unpickler`'s `BUILD` opcode -> `GMM.__setstate__`, which
      `SafeUnpickler` leaves untouched -- only `find_class` is overridden).
    - New `tools/generate_crf_pattern_fixtures.py`: `beats_crf`
      function-level fixtures (fed a real beat activation function),
      `BeatTrackingProcessor`/`BeatDetectionProcessor`/
      `CRFBeatDetectionProcessor` end-to-end decoded beat times (all 3
      44.1kHz test-wav cases, shared-`RNNBeatProcessor`-instance-in-order
      discipline), `GMM` score/posterior fixtures against the real
      `PATTERNS_BALLROOM` GMMs, a `PATTERNS_BALLROOM` structural digest,
      and `PatternTrackingProcessor` end-to-end (audio -> multi-band
      features -> decoded (down-)beats, all 3 cases).
    - **Faithfulness proof: PASSED, bit-identical throughout.**
      `tests/test_beats_crf.py`'s cross-BLAS test AND an 80/150-trial
      in-process/subprocess fuzz test both reproduce real madmom's
      CRF-decoded path AND scalar `log_prob` with **zero differing
      elements** -- `viterbi()` touches no BLAS at all (pure
      elementwise/reduction numpy), so bit-identity, not ULP-closeness, is
      the expected and verified claim, same precedent as 4c's comb
      filters. `tests/test_gmm.py`'s cross-BLAS test reproduces real
      madmom's `GMM.score`/`score_samples` with **zero differing
      elements** for every GMM in both pattern files (despite
      `score_samples` calling `scipy.linalg.cholesky`/`solve_triangular`,
      genuinely BLAS/LAPACK-backed -- a non-free claim, verified rather
      than assumed). `tests/test_beats.py`'s and `tests/test_patterns.py`'s
      cross-BLAS tests reproduce real madmom's `BeatTrackingProcessor`/
      `BeatDetectionProcessor`/`CRFBeatDetectionProcessor` decoded beat
      times AND `PatternTrackingProcessor`'s decoded (down-)beat
      positions/beat numbers with **zero differing elements**, for all 3
      44.1kHz test-wav cases.
    - 61 new tests total (`tests/test_beats_crf.py`: 14 offline + 1 fuzz,
      skipped outside the reference venv; `tests/test_gmm.py`: 8 offline +
      1 network; `tests/test_patterns.py`: 8; `tests/test_beats.py`: +7 (6
      offline + 1 network); `tests/test_fixtures_exist.py`: +5). Full
      offline suite: 264 passed, 2 skipped, 53 deselected (was 223/1/51
      after 4e); network suite: 53 passed, 1 skipped, 265 deselected, all
      green.
  - **4g status: DONE (2026-07-13) -- final wave, campaign closed.** Ported
    the last 4 audit-table targets and resolved the 4b TO-VERIFY flag, then
    closed out the whole Phase-4 campaign:
    - New `madmom_infer/audio/cepstrogram.py`: `Cepstrogram`,
      `CepstrogramProcessor`, `MFCC`, `MFCCProcessor` -- composition classes
      (not `np.ndarray` subclasses, matching this project's convention),
      reusing 4b's already-ported `MelFilterbank` (the reason that pull-
      forward existed). **Major, real, confirmed upstream bug found**:
      `MFCC.__new__`'s "was this spectrogram already filtered?" check
      unconditionally raises `AttributeError` for a PLAIN `Spectrogram` (or
      a raw wav path/array, which builds one internally) -- confirmed
      directly against the reference venv: the base `Spectrogram` class
      never defines a `.filterbank` attribute at all, so
      `MFCC(plain_spectrogram)`/`MFCC(wav_path)` always crashed in real
      madmom. The ONLY input that worked was an ALREADY-`FilteredSpectrogram`
      instance, whose real `.filterbank` attribute trips the "redo
      calculation" warn-and-recompute branch (which discards that filter
      and rebuilds from `.stft` -- no further attribute checks after that
      point). This wave (4g, 2026-07-13) initially reproduced the bug
      bug-for-bug -- a plain (undefended) attribute access letting the
      identical `AttributeError` propagate, pinned by `pytest.raises` --
      per this project's then-current bug-for-bug-fidelity policy; also
      verbatim-reproduced at the time: `MFCCProcessor.process()` never
      forwarded its own stored `self.transform` to the `MFCC(...)` call it
      makes (a custom `transform=` was inert, matching upstream's own
      apparent oversight). **Superseded, same day (2026-07-13)**: the
      bug-for-bug-fidelity policy was reversed (see
      `docs/blueprints/decisions.md`'s "Fix inherited defects after
      migration") and both defects were fixed in place instead --
      `MFCC` now inspects filtering/scaling attributes defensively
      (`getattr(..., None)`), so raw audio, a plain `Spectrogram`, and an
      already-filtered `Spectrogram` all follow the documented Mel -> log
      -> DCT path correctly (the warn-and-recompute behavior for genuinely
      already-filtered input is unchanged); `MFCCProcessor.process()` now
      forwards its stored `self.transform`. Tests assert correct MFCC
      output and transform forwarding, replacing the old `pytest.raises`
      expectations.
      **Faithfulness proof**: `Cepstrogram` is bit-identical to real madmom
      both in-process and cross-BLAS (`np.array_equal`, pure `scipy.
      fftpack.dct`, no BLAS/build sensitivity at all); `MFCC` is
      bit-identical CROSS-BLAS (confirmed directly against the reference
      venv) but shows small in-process (differing-numpy/scipy-build) drift
      -- up to ~3.8e-6 absolute on roughly [-25, 25]-range output --
      root-caused to the compounding `np.dot`(filterbank) -> `np.log10` ->
      `dct` chain, same class of finding as 4d's `SemitoneBandpassSpectrogram`
      and 4e's raw RNN activations (an unstable near-zero ULP metric doesn't
      apply here either -- most of MFCC's output floats near zero); the
      in-process test asserts a documented `atol=1e-5` (~2.6x observed)
      instead.
    - New `madmom_infer/audio/hpss.py`:
      `HarmonicPercussiveSourceSeparation` (alias `HPSS`). **Real, confirmed
      upstream bug found**: `process()` is unconditionally broken for EVERY
      input in real madmom 0.17.dev0 -- confirmed directly against the
      reference venv: a `Spectrogram` input raises `AttributeError`
      (`Spectrogram` has no `.spec` attribute, despite `process()`'s own
      code reading `data.spec`); any other input raises `UnboundLocalError`
      (`spectrogram` referenced before assignment, since it's only assigned
      inside the `if isinstance(data, Spectrogram)` branch). This wave (4g,
      2026-07-13) initially reproduced both failures exactly, pinned by
      `pytest.raises`, per this project's then-current bug-for-bug-fidelity
      policy. **Superseded, same day (2026-07-13)**: the policy was
      reversed (see `docs/blueprints/decisions.md`'s "Fix inherited defects
      after migration") and `process()` was fixed in place instead -- it now
      normalizes any 2-D spectrogram-like input via `np.asarray`, composes
      the already-correct `slices()`/`masks()` helpers, and raises a clear
      `ValueError` for non-2-D input, satisfying the `Processor` contract
      for every valid call instead of failing for all of them. Tests assert
      correct harmonic/percussive output, replacing the old `pytest.raises`
      expectations. `slices()`/`masks()` (the two helper methods, never
      themselves broken) remain bit-identical to real madmom, both
      in-process and cross-BLAS (pure `scipy.ndimage.median_filter` +
      elementwise mask arithmetic, no BLAS at all).
    - `madmom_infer/audio/filters.py`: `HarmonicFilterbank` -- verbatim port
      of upstream's own unconditional `raise NotImplementedError`, confirmed
      by reading `filters.py:1369-1379` directly (no filterbank-construction
      code follows the raise at all -- not even `SimpleChromaFilterbank`'s
      own dead, TODO-commented code). Audit table's lowest-priority row: no
      processor in this project needs it.
    - `madmom_infer/audio/signal.py`: `attenuate`, `rescale`, `trim`,
      `energy`, `root_mean_square`, `sound_pressure_level` -- 6 verbatim
      ports resolving the wave-4b audit-table TO-VERIFY flag (see that
      row's own history: 4b found `smooth` missing and, rather than
      trusting the Phase-1 audit's overstated PORTED claim for the whole
      rest of the row, downgraded it to TO-VERIFY pending a real re-audit).
      Re-auditing `../madmom-upstream/madmom/audio/signal.py`'s actual
      public surface (`grep -n '^def \|^class '`) found these 6 are the
      only genuine gaps; the rest of that flagged list (`Stream`,
      `LoadAudioFileError`, `load_wave_file`, `write_wave_file`,
      `load_audio_file`) is a DELIBERATE non-port: `Stream` is already
      covered by this project's permanent online/live-audio exclusion, and
      the other four are themselves nothing but upstream's own
      deprecated-since-0.16 shims delegating to `madmom.io.audio.*`
      (confirmed by reading `signal.py:442-493` directly) -- `io/*` is this
      project's own separate, already-documented permanent EXCLUDE.
      **Faithfulness proof**: all 6 bit-identical to real madmom, both
      in-process and cross-BLAS (pure numpy, no BLAS at all). Found and
      fixed a real bug in this wave's OWN first draft of `rescale()`: a
      bare `signal.astype(dtype)` call assumed ndarray semantics this
      project's own composition `Signal` class doesn't have (no `.astype`
      method) -- fixed with `np.asarray(signal).astype(dtype)`, which works
      uniformly for both a `Signal` and a plain ndarray.
    - New `tools/generate_leftovers_fixtures.py`: output-only golden
      fixtures for all of the above (same "reuse this project's own
      already-golden-fixture-proven Phase-1 DSP chain to reconstruct
      inputs" economy as `tools/generate_onset_fixtures.py`) --
      `tests/fixtures/cepstrogram.npz`, `tests/fixtures/hpss.npz`,
      `tests/fixtures/signal_leftovers.npz`.
    - **Closure audit**: walked the entire audit table below top to bottom;
      every row now reads PORTED or EXCLUDE, zero TO-PORT/TO-VERIFY rows
      remain. Cross-checked by introspection: imported all 26 PORTED-row
      modules and verified every named class/function actually exists and
      is importable in `madmom_infer` (script run, zero failures -- see
      the "4g closure verdict" section below for the full statement).
    - 38 new tests total (`tests/test_cepstrogram.py`: 12;
      `tests/test_hpss.py`: 9; `tests/test_signal_leftovers.py`: 13;
      `tests/test_filters.py`: +1; `tests/test_fixtures_exist.py`: +3).
      Full offline suite: 302 passed, 2 skipped, 53 deselected (was
      264/2/53 after 4f); network suite: 53 passed, 1 skipped, 303
      deselected, all green (no new network-marked tests -- none of 4g's
      targets need model downloads).
  - **Phase 4 complete-port campaign: DONE (2026-07-13).** See the "4g
    closure verdict" section (right after the audit table below) for the
    full, dated closing statement.

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
| `audio/signal.py` | `resample` | PORTED (4d) -- **policy correction**: the "no ffmpeg dependency" Phase-1 exclusion (below) does not survive `SemitoneBandpassFilterbank`'s unconditional, load-bearing need for it; narrow ffmpeg-subprocess port, bit-identical to real madmom's own `resample()` (both invoke the literal same system `ffmpeg` binary), see 4d status | -- | feeds `audio/spectrogram.py`'s `SemitoneBandpassSpectrogram` |
| `audio/signal.py` | `attenuate`, `rescale`, `trim`, `energy`, `root_mean_square`, `sound_pressure_level` | PORTED (4g) | -- | resolves the 4b TO-VERIFY flag; 6 verbatim ports, bit-identical to real madmom both in-process and cross-BLAS (pure numpy, no BLAS) -- confirmed by re-grepping `../madmom-upstream/madmom/audio/signal.py`'s actual `^def \|^class ` surface, not re-trusting the old Phase-1 claim |
| `audio/signal.py` | `Stream`, `LoadAudioFileError`, `load_audio_file`, `load_wave_file` (public), `write_wave_file` | EXCLUDE (4g, resolves the rest of the 4b TO-VERIFY flag) | -- | `Stream` is madmom's online/live-audio (PyAudio) class, already covered by this project's permanent online-processing exclusion; the other four are themselves nothing but upstream's own deprecated-since-0.16 shims that `warnings.warn()` and delegate to `madmom.io.audio.*` (confirmed by reading `signal.py:442-493` directly) -- `io/*` is this project's own separate, already-documented permanent EXCLUDE (see the `io/*`/`utils/*` row below), so porting these 4 would mean porting `io.audio` under a different name; none referenced by any `../madmom-upstream/madmom/{audio,features,ml}/*` file this project ports from |
| `audio/filters.py` | `Filterbank`, `LogarithmicFilterbank`, `log_frequencies`, `frequencies2bins`, `bins2frequencies`, freq-conversion helpers (`hz2mel` etc.) | PORTED | -- | Phase 1 |
| `audio/filters.py` | `MelFilterbank` | PORTED (4b) | -- | originally slotted for 4g (`cepstrogram.py` MFCC), pulled forward -- also feeds `CNNOnsetProcessor`'s 80-band mel input, which is in 4b's own scope; 4g's MFCC work reuses this instead of re-porting |
| `audio/filters.py` | `BarkFilterbank`, `RectangularFilter`, `RectangularFilterbank` | PORTED (4f) | -- | **correction**: only `RectangularFilterbank` is actually load-bearing for `MultiBandSpectrogramProcessor`/`PatternTrackingProcessor` (confirmed by reading `audio/spectrogram.py:1310` directly) -- `BarkFilterbank`/`RectangularFilter` are unreachable from any target this project ships, ported anyway for API completeness (same precedent as `NOTES_CNN_MIREX`) |
| `audio/filters.py` | `PitchClassProfileFilterbank`, `HarmonicPitchClassProfileFilterbank`, `SemitoneBandpassFilterbank` | PORTED (4d, scope addition -- see corrections above) | -- | feed `audio/chroma.py`'s classic (non-DNN) and CLP chroma paths |
| `audio/filters.py` | `SimpleChromaFilterbank` | PORTED (4d) -- ported INCLUDING its unconditional `raise NotImplementedError` | -- | confirmed by reading upstream directly: not actually implemented in real madmom either (dead code below the raise); this port reproduces that state rather than finishing what upstream itself never enabled |
| `audio/filters.py` | `HarmonicFilterbank` | PORTED (4g) -- ported INCLUDING its unconditional `raise NotImplementedError` | -- | confirmed by reading upstream directly: not actually implemented in real madmom either (no filterbank-construction code follows the raise at all), same not-actually-implemented shape as `SimpleChromaFilterbank` (4d); no processor in this project needs it, ported anyway for API-surface completeness |
| `audio/stft.py` | `ShortTimeFourierTransform`, `ShortTimeFourierTransformProcessor`, `stft`, `fft_frequencies` | PORTED | -- | Phase 1 |
| `audio/stft.py` | `Phase`, `LocalGroupDelay`/`LGD`, `phase`, `local_group_delay`, `lgd` | PORTED (4b) | -- | feeds onset phase-deviation family; `LocalGroupDelay` reproduces a real upstream bug on purpose (`__new__` checks `isinstance(stft, Phase)` where `stft` is an undefined name resolving to the module's own `stft()` function, so it always rebuilds rather than reusing an existing `Phase`) -- see `audio/stft.py`'s module header |
| `audio/stft.py` | `rfft_builder` | EXCLUDE | -- | `pyfftw` acceleration hook, not a project dependency (see `audio/stft.py`'s Phase-1 header) |
| `audio/spectrogram.py` | `Spectrogram`, `SpectrogramProcessor`, `FilteredSpectrogram(Processor)`, `LogarithmicSpectrogram(Processor)`, `SpectrogramDifference(Processor)` | PORTED | -- | Phase 1 |
| `audio/spectrogram.py` | `SuperFluxProcessor` | PORTED (4b) | -- | onset family |
| `audio/spectrogram.py` | `MultiBandSpectrogram`, `MultiBandSpectrogramProcessor` | PORTED (4f) -- cross-BLAS-proven exact (as part of `PatternTrackingProcessor`'s own end-to-end test) | -- | `PatternTrackingProcessor` input |
| `audio/spectrogram.py` | `SemitoneBandpassSpectrogram` | PORTED (4d) -- own composition class, NOT a `FilteredSpectrogram` subclass; measured NOT bit-identical to real madmom across differing scipy versions (up to ~1e-5 absolute, `scipy.signal.filtfilt`/`ellip` version sensitivity, see 4d status) | -- | `CLPChromaProcessor` input; needs `audio/signal.py`'s new ffmpeg-subprocess `resample()` |
| `audio/cepstrogram.py` | `Cepstrogram`, `CepstrogramProcessor`, `MFCC`, `MFCCProcessor` | PORTED (4g) | -- | `Cepstrogram` bit-identical to real madmom (in-process and cross-BLAS); `MFCC` bit-identical cross-BLAS, `atol=1e-5`-documented in-process drift (see 4g status) -- **fixed in place 2026-07-13** (previously reproduced a confirmed upstream bug bug-for-bug: real madmom's `MFCC` could only be constructed from an already-`FilteredSpectrogram` instance, every other input unconditionally raising `AttributeError`; see `docs/blueprints/decisions.md`'s "Fix inherited defects after migration"): `MFCC` now inspects filtering/scaling attributes defensively so raw audio, a plain `Spectrogram`, and an already-filtered `Spectrogram` all follow the Mel -> log -> DCT path correctly, and `MFCCProcessor` now forwards its stored `transform` |
| `audio/chroma.py` | `DeepChromaProcessor` | PORTED (4d) -- cross-BLAS-proven exact | `CHROMA_DNN` = `chroma/2016/chroma_dnn.pkl` | pickle refs confirmed: `NeuralNetwork`, `FeedForwardLayer`, `relu`/`sigmoid` -- no new layer classes needed beyond 4a's set |
| `audio/chroma.py` | `CLPChroma`, `CLPChromaProcessor` | PORTED (4d) -- see `SemitoneBandpassSpectrogram` row re: measured (not bit-identical) cross-scipy-version precision | -- | pure DSP, no NN weights; needs `SemitoneBandpassSpectrogram` |
| `audio/chroma.py` | `PitchClassProfile`, `HarmonicPitchClassProfile` | PORTED (4d, scope addition) | -- | classic chroma, not DNN-based; composition subclasses of `Spectrogram`, not upstream's ndarray-view hierarchy |
| `audio/comb_filters.pyx` | `feed_forward_comb_filter`, `feed_backward_comb_filter`, `comb_filter`, `CombFilterbankProcessor` | PORTED (4c) | -- | numpy port (same playbook as `hmm.pyx`); feeds `TempoEstimationProcessor`'s comb-filter histogram mode; bit-identical, not just ULP-close -- see 4c status below |
| `audio/hpss.py` | `HPSS`/`HarmonicPercussiveSourceSeparation` | PORTED (4g) | -- | standalone preprocessing utility, not consumed by any other processor in this project; `slices()`/`masks()` bit-identical to real madmom (in-process and cross-BLAS) -- **fixed in place 2026-07-13** (previously reproduced a confirmed upstream bug bug-for-bug: real madmom's `process()` unconditionally raised `AttributeError` or `UnboundLocalError` for EVERY call; see `docs/blueprints/decisions.md`'s "Fix inherited defects after migration"): `process()` now normalizes any 2-D spectrogram-like input via `np.asarray`, composes `slices()`/`masks()`, and raises a clear `ValueError` for non-2-D input, satisfying the `Processor` contract for every valid call |
| `ml/hmm.py` | `TransitionModel`, `ObservationModel`, `DiscreteObservationModel`, `HiddenMarkovModel`/`HMM` | PORTED | -- | Phase 1 |
| `ml/crf.py` | `ConditionalRandomField` | PORTED (4d) -- cross-BLAS-proven exact | -- | chord decoding (`CRFChordRecognitionProcessor`, `DeepChromaChordRecognitionProcessor`); added a `.load()` classmethod (not in upstream) delegating to the restricted unpickler, matching `NeuralNetwork.load` |
| `ml/gmm.py` | `GMM`, `log_multivariate_normal_density`, `logsumexp`, `pinvh` | PORTED (4f) -- cross-BLAS-proven exact | -- | backs `GMMPatternTrackingObservationModel`; forward-inference only, no `fit()` (permanent scope); `GMM.__setstate__`'s legacy rename branch is load-bearing -- both target `PATTERNS_BALLROOM` files are old-format pickles |
| `ml/nn/__init__.py` | `NeuralNetwork`, `NeuralNetworkEnsemble`, `average_predictions` | PORTED | -- | Phase 2 |
| `ml/nn/layers.py` | `Layer`, `FeedForwardLayer`, `RecurrentLayer`, `BidirectionalLayer`, `Gate`, `Cell`, `LSTMLayer` | PORTED | -- | Phase 2 |
| `ml/nn/layers.py` | `ConvolutionalLayer`, `MaxPoolLayer`, `BatchNormLayer`, `PadLayer`, `AverageLayer` | PORTED (4a) | -- | confirmed pickletools-walked as exactly what `key_cnn.pkl` (`AverageLayer`,`BatchNormLayer`,`ConvolutionalLayer`,`MaxPoolLayer`,`PadLayer`,`elu`,`linear`) references; `onsets_cnn.pkl`, `notes_cnn*.pkl`, `chords_cnnfeat.pkl` also need this same set (reused by 4b/4d/4e, not re-ported) |
| `ml/nn/layers.py` | `GRULayer`, `GRUCell` | PORTED (4c, scope addition -- see corrections above) | -- | `downbeats_bgru_*.pkl` (12 files) reference these; also needed 2 generic old-style-class-reconstruction unpickle allowlist entries the other target pickles don't (`copy_reg._reconstructor`, `__builtin__.object`) -- these 12 files are an OLDER pickle format than every other target `.pkl` in this project, confirmed by real madmom's own "please update your GRU models" `RuntimeWarning` firing on load |
| `ml/nn/layers.py` | `StrideLayer` | PORTED (4b) | -- | `onsets_cnn.pkl` references it (confirmed by `pickletools`); needs `utils.segment_axis` (see `utils/*` row below) |
| `ml/nn/layers.py` | `ReshapeLayer`, `TransposeLayer` | PORTED (4e) | -- | `notes_cnn.pkl` needs Reshape+Transpose, confirmed by `pickletools`; confirmed by 4b's own `pickletools` walk of `onsets_cnn.pkl` that it does NOT need either of these (only `StrideLayer`, above) |
| `madmom.processors` | `SequentialProcessor`, `ParallelProcessor` (as unpickle targets) | PORTED (4e, scope addition) | -- | `notes_cnn.pkl` pickles a whole processor graph directly, not a bare `NeuralNetwork` -- see 4e status for the full finding; this project's own `madmom_infer/processors.py` classes already worked as unpickle targets with no changes |
| `ml/nn/layers.py` | `TCNBlock`, `TCNLayer` | EXCLUDE | -- | no shipped model references them (`BEATS_TCN` not in `package_data`; confirmed by attempted load, file absent from installed tree) |
| `ml/nn/layers.py` | `MultiTaskLayer`, `ParallelLayer`, `SequentialLayer` | EXCLUDE | -- | only used by TCN multi-task models, which aren't shipped |
| `ml/nn/activations.py` | `linear`, `tanh`, `sigmoid`, `relu`, `elu`, `softmax` | PORTED | -- | Phase 2 |
| `features/beats_hmm.py` | `BeatStateSpace`, `BarStateSpace`, `BeatTransitionModel`, `BarTransitionModel`, `RNNBeatTrackingObservationModel`, `RNNDownBeatTrackingObservationModel`, `exponential_transition` | PORTED | -- | Phase 2 |
| `features/beats_hmm.py` | `MultiPatternStateSpace`, `MultiPatternTransitionModel`, `GMMPatternTrackingObservationModel` | PORTED (4f) | -- | pattern-tracking HMM machinery; near-line-for-line ports |
| `features/downbeats.py` | `RNNDownBeatProcessor`, `DBNDownBeatTrackingProcessor` | PORTED | `DOWNBEATS_BLSTM` | Phase 2, cross-BLAS-proven exact |
| `features/downbeats.py` | `RNNBarProcessor`, `SyncronizeFeaturesProcessor` | PORTED (4c, scope addition; INSTANTIABLE + full-audio-in-proven-exact as of 4d) | `DOWNBEATS_BGRU` | needed `GRULayer`/`GRUCell` (4c) + `CLPChromaProcessor` (4d, unblocked `RNNBarProcessor.__init__`); 4d found and fixed a genuine `SyncronizeFeaturesProcessor` latent bug (`features.T` on a non-ndarray composition object), see 4d status |
| `features/downbeats.py` | `DBNBarTrackingProcessor`, `PatternTrackingProcessor` | PORTED (4f) -- cross-BLAS-proven exact (`PatternTrackingProcessor`) | `PATTERNS_BALLROOM` (no NN globals -- GMM-only) | upstream's actual class name is `PatternTrackingProcessor`, not `GMMPatternTrackingProcessor` as the original wave-plan bullet named it -- same processor, corrected here; `PatternTrackingProcessor` loads pattern files via this project's own restricted `SafeUnpickler`, not upstream's bare `pickle.load`; `DBNBarTrackingProcessor` needs no GMM at all, reuses `RNNBeatTrackingObservationModel` |
| `features/downbeats.py` | `LoadBeatsProcessor` | EXCLUDE | -- | file/STDIN batch-loading plumbing for `bin/` CLI scripts, not an inference algorithm |
| `features/beats.py` | `RNNBeatProcessor`, `DBNBeatTrackingProcessor`, `MultiModelSelectionProcessor` | PORTED (4c) | `BEATS_LSTM`, `BEATS_BLSTM` | pickle refs confirm no new layer classes beyond Phase-2's LSTM/BLSTM set; `DBNBeatTrackingProcessor` is offline-only (drops `OnlineProcessor`'s `process_online`, same precedent as `OnsetPeakPickingProcessor`) |
| `features/beats.py` | `BeatTrackingProcessor`, `BeatDetectionProcessor`, `detect_beats` | PORTED (4f) -- cross-BLAS-proven exact | -- | real upstream classes/function (look-aside/look-ahead tempo-driven beat alignment), NOT TCN-specific despite being grouped in the same EXCLUDE row as `TCNBeatProcessor` below in an earlier version of this table -- `detect_beats` is `BeatTrackingProcessor`'s own helper, unrelated to TCN (confirmed by reading `beats.py:301-465` directly); flagged as an audit-table gap in 4c, closed here |
| `features/beats.py` | `CRFBeatDetectionProcessor` | PORTED (4f) -- cross-BLAS-proven exact | -- | needs `features/beats_crf.py` (this wave's own numpy `beats_crf.pyx` port) and `BeatTrackingProcessor` (row above, also this wave) |
| `features/beats.py` | `TCNBeatProcessor`, TCN-specific parts of `detect_beats`, `threshold_activations` | EXCLUDE | `BEATS_TCN` (not shipped) | see corrections above; `threshold_activations` itself is already ported (`features/downbeats.py`) and reused, not duplicated; **correction (4c): `detect_beats` itself is NOT TCN-specific**, see row above -- only ever excluded here because no 4c target needed it, not because it's actually TCN-only |
| `features/tempo.py` | `TempoEstimationProcessor`, `TempoHistogramProcessor`, `ACFTempoHistogramProcessor`, `CombFilterTempoHistogramProcessor`, `DBNTempoHistogramProcessor`, `detect_tempo`, `dominant_interval`, `interval_histogram_acf`, `interval_histogram_comb`, `smooth_histogram` | PORTED (4c) | -- | comb variant needed `audio/comb_filters.py` (this wave); all offline-only (`OnlineProcessor` stays a permanent exclusion, same precedent as `DBNBeatTrackingProcessor`/`OnsetPeakPickingProcessor`) |
| `features/tempo.py` | `TCNTempoHistogramProcessor` | EXCLUDE | -- | only consumes `TCNBeatProcessor` output, which can't exist (no shipped model) |
| `features/onsets.py` | `SpectralOnsetProcessor`, `spectral_diff`, `spectral_flux`, `superflux`, `complex_flux`, `complex_domain`, `rectified_complex_domain`, `high_frequency_content`, `modified_kullback_leibler`, `phase_deviation`, `weighted_phase_deviation`, `normalized_weighted_phase_deviation`, `correlation_diff`, `wrap_to_pi`, `peak_picking`, `OnsetPeakPickingProcessor` | PORTED (4b) | -- | pure DSP, no NN weights; `correlation_diff` **fixed in place 2026-07-13** (previously ported bug-for-bug: real madmom itself crashes on this function under Python 3, confirmed empirically against the reference venv, pinned by a `pytest.raises(TypeError)` test rather than a golden output; see `docs/blueprints/decisions.md`'s "Fix inherited defects after migration") -- now uses explicit integer division for the correlation midpoint and is covered by golden-output tests instead; `OnsetPeakPickingProcessor` is offline-only (no `OnlineProcessor`, a stated permanent exclusion) |
| `features/onsets.py` | `RNNOnsetProcessor` | PORTED (4b) | `ONSETS_RNN`, `ONSETS_BRNN` | pickle refs: `FeedForwardLayer`/`RecurrentLayer`/`BidirectionalLayer` -- all already PORTED (Phase 2), no new layer classes; `online=True` (`ONSETS_RNN`) IS supported (unlike `OnsetPeakPickingProcessor`'s online mode) -- it only selects different pretrained weights/frame sizes, not actual streaming; `ONSETS_BRNN_PP` has no registry entry (only loaded by the excluded `bin/SuperFluxNN` CLI script, no processor this project ports needs it) |
| `features/onsets.py` | `CNNOnsetProcessor` | PORTED (4b, reuses 4a's conv layers) | `ONSETS_CNN` | pickle refs: `ConvolutionalLayer`,`MaxPoolLayer`,`BatchNormLayer`,`FeedForwardLayer`,`StrideLayer` (all PORTED, `StrideLayer` new in 4b) + `MelFilterbank` input (pulled forward from 4g into 4b, see `audio/filters.py` row) |
| `features/key.py` | `CNNKeyRecognitionProcessor`, `key_prediction_to_label`, `add_axis` | PORTED (4a) | `KEY_CNN` = `key/2018/key_cnn.pkl` | pickle refs confirmed above; cross-BLAS-proven exact (`tests/test_key.py`) |
| `features/chords.py` | `DeepChromaChordRecognitionProcessor` | PORTED (4d) -- cross-BLAS-proven exact (decoded segments) | `CHORDS_DCCRF` | pickle confirmed **no** NN globals -- CRF-only (`ml/crf.py`), confirms 4d's CRF-first framing; does NOT touch `CLPChroma` (confirmed by reading upstream directly) |
| `features/chords.py` | `CNNChordFeatureProcessor` | PORTED (4d) | `CHORDS_CNN_FEAT` | pickle refs confirmed: `ConvolutionalLayer`,`BatchNormLayer`,`MaxPoolLayer` (4a's set, no new classes) |
| `features/chords.py` | `CRFChordRecognitionProcessor` | PORTED (4d) -- cross-BLAS-proven exact (decoded segments) | `CHORDS_CFCRF` | pickle confirmed no NN globals -- CRF-only |
| `features/chords.py` | `majmin_targets_to_chord_labels` | PORTED (4d) | -- | label-decoding helper alongside the chord processors |
| `features/notes_hmm.py` | `ADSRObservationModel`, `ADSRStateSpace`, `ADSRTransitionModel` | PORTED (4e) | -- | HMM state spaces on existing `ml/hmm.py` machinery |
| `features/notes.py` | `RNNPianoNoteProcessor` | PORTED (4e) | `NOTES_BRNN` = `notes/2013/notes_brnn.pkl` | pickle refs confirmed: `BidirectionalLayer`,`FeedForwardLayer`,`RecurrentLayer` -- already PORTED, no new classes |
| `features/notes.py` | `CNNPianoNoteProcessor` | PORTED (4e, reuses 4a's conv layers) | `NOTES_CNN` = `notes/2019/notes_cnn.pkl` | pickle refs confirmed: `ConvolutionalLayer`,`BatchNormLayer`,`ReshapeLayer`,`TransposeLayer` -- **also a real surprise**: the pickle is a whole `SequentialProcessor`/`ParallelProcessor` graph, not a bare `NeuralNetwork`, see 4e status; `NOTES_CNN_MIREX` (`notes/2018/notes_cnn_[12].pkl`) is real+shipped but unused by any ported processor (confirmed by reading `CNNPianoNoteProcessor.__init__` directly), no registry entry, same precedent as 4b's `ONSETS_BRNN_PP` |
| `features/notes.py` | `ADSRNoteTrackingProcessor`, `NotePeakPickingProcessor`, `NoteOnsetPeakPickingProcessor` | PORTED (4e) | -- | decode/peak-picking, no NN weights |
| **`evaluation/*`** | (entire subpackage) | EXCLUDE | -- | out of scope per this repo's stated scope (see top of this file) |
| **`bin/*`** | (CLI scripts, installed as `console_scripts`-style `scripts=` entries by upstream `setup.py`) | EXCLUDE | -- | this package is a library, processors are the API (Permanent exclusions) |
| **`io/*`, `utils/*`** | `io.audio`, `io.midi`, `utils.midi`, `utils.stats` | EXCLUDE (out of this audit's stated scope: `features/`, `audio/`, `ml/` only) | -- | I/O/annotation-file helpers, not inference algorithms; flagged here rather than silently dropped, revisit only if a TO-PORT processor is found to need one (none currently do) |
| `utils/__init__.py` | `segment_axis`, `combine_events` | PORTED (4b, narrow carve-out -- `madmom_infer/utils.py`, NOT a general `utils/*` port) | -- | correction to the row above: 4b found two real, non-speculative dependencies -- `StrideLayer` (`ml/nn/layers.py`) calls `segment_axis` (this port implements only its `axis=0`/`end='cut'` case, the only one `StrideLayer` ever uses, NOT upstream's full generality), `OnsetPeakPickingProcessor` (`features/onsets.py`) calls `combine_events` (ported in full, all 3 `combine` modes, cheap) |

### 4g closure verdict (2026-07-13)

**COMPLETE: every inference-relevant public class/function in upstream
`features/`, `audio/`, and `ml/` is ported or documented-excluded.** Walked
the audit table above top to bottom, row by row: zero rows read TO-PORT or
TO-VERIFY; every row reads either PORTED (with the wave that ported it) or
EXCLUDE (with a one-line reason). This is the closing statement for the
whole Phase 4 complete-port campaign (waves 4.0, 4a-4g), not just this
wave's own targets.

**Cross-checked against reality, not just against this file's own prose**:
a sample-the-whole-surface introspection script imported all 26 modules the
table's PORTED rows point at (`madmom_infer.audio.{signal,filters,stft,
spectrogram,cepstrogram,chroma,comb_filters,hpss}`, `madmom_infer.ml.{hmm,
crf,gmm,nn,nn.layers,nn.activations}`, `madmom_infer.features.{beats_hmm,
downbeats,beats,beats_crf,tempo,onsets,key,chords,notes_hmm,notes}`,
`madmom_infer.processors`, `madmom_infer.utils`) and verified every single
named class/function in this table's PORTED rows actually exists as an
attribute of its stated module and is importable -- zero failures, run
2026-07-13 against this wave's own dev venv.

**Permanent EXCLUDE list, final (one-line reasons, cross-checked this
wave)**:
- `evaluation/*` (entire subpackage) -- out of this project's stated scope
  from day one (see the top of this file); not an inference algorithm.
- `bin/*` (CLI scripts) -- this package is a library, processors are the
  API; re-confirmed this wave that no audit-table PORTED row secretly
  depends on any `bin/*` code.
- `io/*`, most of `utils/*` -- I/O/annotation-file helpers, not inference
  algorithms; the two real, load-bearing exceptions (`utils.segment_axis`/
  `utils.combine_events`, narrow carve-outs, wave 4b) are already PORTED,
  not excluded (see the row above).
- `OnlineProcessor`/streaming machinery (`Stream`, `FramedSignalProcessor`'s
  `'stream'`-origin PyAudio integration, `OnsetPeakPickingProcessor`'s/
  `DBNBeatTrackingProcessor`'s/`DBNTempoHistogramProcessor`'s online modes)
  -- documented as a permanent exclusion since Phase 1/2
  (`madmom_infer/audio/signal.py`'s module header) and re-confirmed
  consistently at every wave that touched an `OnlineProcessor` subclass
  (4b's `OnsetPeakPickingProcessor`, 4c's `DBNBeatTrackingProcessor`/
  `DBNTempoHistogramProcessor`) -- this project ships the offline-batch
  decode path only, never the live/streaming one.
- `TCNBlock`/`TCNLayer`/`TCNBeatProcessor`/`TCNTempoHistogramProcessor`/
  `MultiTaskLayer`/`ParallelLayer`/`SequentialLayer` -- **re-confirmed this
  wave, not merely re-asserted**: `BEATS_TCN`'s `.pkl` files are absent
  from `../madmom-upstream`'s `setup.py` `package_data` (only `models/
  beats/201[56]/*`, i.e. the 2015 BLSTM + 2016 LSTM families, are actually
  installed by a real madmom package) -- no shipped model can ever reach
  TCN code, so it stays permanently unreachable, not merely low-priority.
- `rfft_builder` (`audio/stft.py`) -- `pyfftw` acceleration hook, not a
  project dependency.
- `LoadBeatsProcessor` (`features/downbeats.py`) -- file/STDIN batch-loading
  plumbing for `bin/` CLI scripts, not an inference algorithm (same
  category as the `bin/*` exclusion, just defined in a features/ module).
- `Stream`, `LoadAudioFileError`, `load_audio_file`, `load_wave_file`
  (public), `write_wave_file` (`audio/signal.py`) -- resolved this wave
  (see the row above): `Stream` falls under the online-streaming exclusion
  above; the other four are themselves nothing but upstream's own
  deprecated-since-0.16 shims delegating to the already-excluded `io.audio`
  module (confirmed by reading `signal.py:442-493` directly).

**What was NOT re-verified this wave** (stated plainly, not buried): the
introspection check above confirms importability and attribute existence,
not per-class behavioral correctness -- that's what each wave's own
golden-fixture tests already established at the time it shipped (see each
wave's status entry above for its own faithfulness proof), and this
closure audit did not re-run every prior wave's cross-BLAS test individually
by hand (the full-suite run below covers that mechanically). No new
upstream source-reading was done this wave for modules outside 4g's own 4
targets -- the closure claim rests on the audit table's accumulated,
wave-by-wave, cited findings, not a fresh re-read of the entire upstream
tree.

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

**Current policy (in force since 2026-07-13)**: once migration is
validated and stable, a confirmed inherited defect is judged against this
project's own public contract, reasonable API usage, and result
correctness -- not against matching an upstream failure -- and is fixed in
place rather than preserved forever. See `docs/blueprints/decisions.md`'s
"Fix inherited defects after migration" for the full rationale and the
first repair set (`correlation_diff`, `MFCC`, `MFCCProcessor`,
`HPSS.process()`), all fixed in place 2026-07-13. Preserve migration
assets users and model files depend on (import paths, public names,
processor composition, serialized-model compatibility, numerically
validated inference behavior); don't add a second "clean" namespace beside
a fixed legacy symbol. Tests for a repaired symbol should assert useful,
correct behavior, not an inherited exception.

**This supersedes the original policy** (in force 2026-07-12 through
2026-07-13, while Phase 4 was still landing waves): when a wave found a
confirmed upstream bug, the default was bug-for-bug reproduction (pinned by
a test expecting the same failure), never a silent fix. That is what
produced the four symbols above in their originally-broken shape; it no
longer applies to new findings -- read the current `decisions.md` entry
before treating any inherited failure as permanent.

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
