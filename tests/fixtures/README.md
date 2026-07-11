# Golden fixtures

Everything in this directory was recorded from a **real, compiled madmom
install** by `tools/generate_fixtures.py` (repo root). These fixtures are the
acceptance standard every madmom_infer port module must match bit-for-bit
(numpy backend) -- see `../../CLAUDE.md`'s "Dual-backend + golden-fixture
testing philosophy" and `../../docs/DESIGN.md` section C.4.

## Provenance

Recorded against:

- **madmom** `0.17.dev0`, commit `27f032e8947204902c675e5e341a3faf5dc86dae`
  (https://github.com/CPJKU/madmom), installed in
  `/home/worzpro/Desktop/dev/openmirlab/all-in-one-fix/.venv`
- **numpy** `1.23.5`
- **scipy** `1.15.3`
- **Python** `3.10.18`

Exact machine-readable provenance (regenerated every run) lives in
[`manifest.json`](./manifest.json) alongside the parameters used for every
fixture category.

## Regenerating

```sh
/home/worzpro/Desktop/dev/openmirlab/all-in-one-fix/.venv/bin/python \
    tools/generate_fixtures.py

# or, equivalently:
uv run --project /home/worzpro/Desktop/dev/openmirlab/all-in-one-fix \
    python tools/generate_fixtures.py
```

The script is standalone (imports only `numpy`/`scipy`/`madmom`, never
`madmom_infer`) and fully deterministic: every random input is seeded, and
re-running it reproduces byte-identical `.wav`/`.npz` files (verified: two
consecutive runs hash-compared identical, including `manifest.json`).

For a from-scratch environment (no dependency on the `all-in-one-fix`
checkout existing), see `docs/DESIGN.md` section C.4 for the pinned
`numpy==1.23.5 scipy==1.15.3 cython 'git+https://github.com/CPJKU/madmom'`
recipe.

## Test wavs (`wavs/`)

Four short (1.5s), deterministic, seeded (`seed=1234`) synthetic clips
covering the exact dtype/channel/sample-rate combinations all-in-one-infer
feeds madmom with:

| file | sample rate | channels | dtype |
|---|---|---|---|
| `mono_44100.wav` | 44100 | 1 | int16 |
| `stereo_44100.wav` | 44100 | 2 (distinct L/R content) | int16 |
| `stereo_48000.wav` | 48000 | 2 (distinct L/R content) | int16 |
| `float32_44100.wav` | 44100 | 1 | float32, range [-1, 1] |

Stereo L/R channels use deliberately different frequency content so the
mono downmix (`Signal(path, num_channels=1)`) exercises a variety of
fractional-average cases, not just symmetric ones.

## Fixture files

- **`signal.npz`** -- for each test wav: `Signal(path)` raw output
  (`{case}_raw` + `{case}_raw_sample_rate`), `Signal(path, num_channels=1)`
  mono-downmix output (`{case}_mono` + `_sample_rate`), and
  `Signal(ndarray, sample_rate=...)` from-array output
  (`{case}_fromarray` + `_sample_rate`).
- **`framing.npz`** -- `FramedSignalProcessor(frame_size=2048, fps=100)`
  applied to each *raw* test wav's `Signal` (i.e. including the raw
  multi-channel stereo cases, to catch channel-shape framing bugs, not just
  the mono downmix). Records `num_frames`, `frame_size`, `hop_size`, frames 0
  and 1 in full, the *last* frame in full (to catch off-by-one
  origin/padding bugs), and a SHA-256 hash of all frames stacked (cheap way
  to fingerprint the entire framed signal without storing it).
- **`stft.npz`** -- `ShortTimeFourierTransformProcessor()` (defaults) applied
  to the **mono-compatible** signal for each test wav (stereo files are
  downmixed first -- see "Surprises" below for why). Records the complex
  STFT of frames 0, 1, and the last frame, plus an all-frames SHA-256 hash.
  Also records a deliberate **window-caching gotcha** demonstration:
  `window_caching_reused_output` (a float32 signal's STFT computed by a
  `ShortTimeFourierTransformProcessor` instance that was *first* called on an
  int16 signal) vs. `window_caching_fresh_output` (the same float32 signal,
  fresh instance) -- these differ (`window_caching_max_abs_diff` records by
  how much), because the processor caches its int16-scaled window across
  calls. See "Surprises" below.
- **`filterbank.npz`** -- the filterbank **matrix itself**
  (`filterbank_matrix_44100`, `filterbank_matrix_48000` -- one per sample
  rate actually exercised, since the matrix depends on `sample_rate` and FFT
  size) from `FilteredSpectrogramProcessor(num_bands=12, fmin=30,
  fmax=17000, norm_filters=True)`, plus filtered-spectrogram outputs
  (frame 0/1/last + all-frames hash) for one 44.1kHz and one 48kHz chain.
- **`logspec.npz`** -- `LogarithmicSpectrogramProcessor(mul=1, add=1)`
  outputs (frame 0/1/last + all-frames hash) for the same two chains as
  `filterbank.npz`.
- **`full_chain.npz`** -- the top-level integration fixture: exactly
  all-in-one-infer's `build_spec_processor()` --
  `SequentialProcessor([frames, stft, filt, spec])` -- run end-to-end on the
  mono-compatible signal for each test wav. Frame 0/1/last + all-frames hash
  + `num_frames`, per case.
- **`hmm_toy.npz`** -- a hand-built 10-state HMM: a dense transition matrix
  (`dense_transition_matrix`), the `TransitionModel.from_dense(...)`-derived
  CSR arrays (`tm_states`/`tm_pointers`/`tm_probabilities`), a
  `DiscreteObservationModel` (`observation_probabilities`, 10 states x 5
  observation types), a fixed observation sequence
  (`observation_sequence`, length 20), and the resulting
  `viterbi()` path + log-probability (`viterbi_path`, `viterbi_log_prob`) and
  `forward()` output (`forward_output`).
- **`beats_hmm.npz`** -- the *real* state space: `BarStateSpace` /
  `BarTransitionModel` / `RNNDownBeatTrackingObservationModel` with
  all-in-one-infer's exact params (`min_bpm=55, max_bpm=215, num_tempi=60,
  transition_lambda=100, observation_lambda=16, fps=100`), for
  `beats_per_bar` in `{3, 4}` **separately** (keys prefixed `bpb3_`/`bpb4_`).
  Records state-space metadata (`num_states`, `num_beats`, `first_states`,
  `last_states`), and the transition model's full CSR arrays (`tm_states`,
  `tm_pointers`, `tm_probabilities`) -- **this is the ground truth for the
  port**, per the task brief -- plus the observation model's `om_pointers`.
- **`dbn_downbeat.npz`** -- a full
  `DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100).process()`
  decode of a synthetic, seeded (`seed=1236`), ~30-second beat-activation
  array (`activations`, float32, shape `(3000, 2)`) into beat/downbeat times
  (`beat_times`, shape `(N, 2)`: seconds + beat-in-bar position).

## Surprises worth knowing before porting this code

- **`num_bands` in `FilteredSpectrogramProcessor` means "bands *per
  octave*", not a total band count.** `num_bands=12, fmin=30, fmax=17000`
  produces an **81**-band filterbank (`filterbank_matrix_44100.shape ==
  (1024, 81)`), not a 12-band one. Easy to misport if you read `num_bands`
  as a literal total.
- **`ShortTimeFourierTransformProcessor` requires a mono (2D:
  `(num_frames, frame_size)`) `FramedSignal`.** Feeding it a raw
  multi-channel `FramedSignal` (3D: `(num_frames, frame_size, channels)`)
  raises `ValueError: frames must be a 2D array or iterable, got ...`. This
  is recorded verbatim in `manifest.json`'s `known_error_cases.
  stereo_full_chain` (confirmed empirically by this generation script, not
  assumed) rather than as an `.npz` fixture. It's *why* `stft.npz`,
  `filterbank.npz`, `logspec.npz`, and `full_chain.npz` all use the
  **mono-downmixed** signal for stereo test wavs, while `framing.npz` (which
  doesn't hit the STFT stage) still exercises the raw multi-channel shape.
  This matches real usage: all-in-one-infer always feeds madmom a mono
  int16 stem, never a raw stereo signal.
- **A reused `ShortTimeFourierTransformProcessor` instance silently caches
  the wrong window scale across differing input dtypes.** The processor
  scales its Hanning window by `1/np.iinfo(dtype).max` the first time it's
  called on an integer-dtype signal (`madmom/audio/stft.py:339-349`); on a
  later call with a *different* dtype (e.g. float32) through the *same
  instance*, it does not recompute that scale -- it silently reuses the
  stale one, with no error or warning, and produces numerically wrong STFT
  output. See `stft.npz`'s `window_caching_reused_output` vs.
  `window_caching_fresh_output` (and `window_caching_max_abs_diff`, which is
  large -- not a rounding-level difference). This never bites
  all-in-one-infer today (`build_spec_processor()` reuses one `stft`
  instance, but always across mono int16 stems -- same dtype every call),
  but it's a real, silent correctness trap for any future caller that mixes
  dtypes through one processor instance, or for a port that "helpfully"
  tries to memoize a scaled window without checking dtype identity first.
- **Mono downmix truncates toward zero, it does not round.**
  `Signal(path, num_channels=1)` on a stereo int16 file averages channels in
  float and casts back with `.astype(int16)`, which truncates toward zero
  (`-0.5 -> 0`, not `-1`; `-3.5 -> -3`, not `-4`). Verified directly:
  `np.array([-0.5, -0.5, -3.5]).astype(np.int16) == [0, 0, -3]`. A port that
  uses `np.floor`/`np.round` here instead of `.astype(int)`'s
  truncate-toward-zero semantics will disagree with madmom on exactly the
  boundary cases the `stereo_44100`/`stereo_48000` test wavs were designed
  (distinct L/R frequency content) to hit.
- **One HMM bar-length hypothesis can fail to decode entirely.** With a
  synthetic activation pattern that doesn't match a given bar length well,
  `HiddenMarkovModel.viterbi()` can return `-inf` log-probability (madmom
  raises a `RuntimeWarning` and returns an empty path for that HMM only).
  `DBNDownBeatTrackingProcessor.process()` handles this by taking
  `argmax` over each bar-length hypothesis's log-probability
  (`madmom/features/downbeats.py:280-282`) and simply not choosing a
  hypothesis that failed -- it's not an error path a caller needs to
  special-case, but a port's `viterbi()` needs to handle the `-inf` /
  empty-path case, not just the ordinary path. `dbn_downbeat.npz`'s
  synthetic activation array was deliberately tuned to keep both the
  `beats_per_bar=3` and `beats_per_bar=4` hypotheses within a valid
  (non-`-inf`) log-probability range, so this edge case does *not* appear
  in that specific fixture -- it surfaced only during fixture *design*, and
  is recorded here so the port's own test suite knows to add a dedicated
  test for it rather than assuming it can't happen.

## Phase 2 fixtures (recorded by `tools/generate_phase2_fixtures.py`)

Same provenance (real madmom 0.17.dev0, numpy 1.23.5, scipy 1.15.3,
`all-in-one-fix/.venv`) plus real madmom `DOWNBEATS_BLSTM` pretrained
weights (CC BY-NC-SA 4.0, see NOTICE -- read transiently from that venv's
already-installed madmom wheel, never bundled here). Reuses the SAME test
wavs above; only the 44.1kHz-native ones (`mono_44100`, `stereo_44100`,
`float32_44100`) are usable -- `RNNDownBeatProcessor` hard-codes 44.1kHz and
this project has no resampling, so `stereo_48000.wav` is skipped.

- **`nn_structural_digest.json`** -- for each of the 8
  `downbeats_blstm_[1-8].pkl` ensemble networks: every layer's type, weight/
  bias/recurrent-weight/peephole-weight array shape+dtype+sha256, and
  activation-function name, recursed through `BidirectionalLayer`'s
  `fwd_layer`/`bwd_layer` and `LSTMLayer`'s `input_gate`/`forget_gate`/
  `cell`/`output_gate`. This is the ground truth `tests/test_ml_nn.py`
  compares this port's `SafeUnpickler`-based loading against.
- **`rnn_downbeat.npz`** -- for each of the 3 usable cases:
  `RNNDownBeatProcessor()(wav)` activations (`{case}_activations`, shape
  `(num_frames, 2)`: beat, downbeat) and
  `DBNDownBeatTrackingProcessor(beats_per_bar=[3,4], fps=100)(activations)`
  decoded beat times (`{case}_beat_times`).

### A fourth caching gotcha (same shape as the two above, found one level up)

`RNNDownBeatProcessor.__init__` builds one `ShortTimeFourierTransformProcessor`
and one `FilteredSpectrogramProcessor` PER FRAME-SIZE BRANCH (1024/2048/4096).
The generation script (like real madmom's own typical usage) reuses ONE
`RNNDownBeatProcessor` instance across all 3 cases, IN ORDER
(`mono_44100` -> `stereo_44100` -> `float32_44100`) -- which means the
SAME window-caching (`stft.npz`'s surprise, above) and filterbank-caching
(`filterbank.npz`'s docstring) gotchas apply here too, now triggered by
`RNNDownBeatProcessor` reuse instead of a bare processor reuse. Confirmed
empirically: a FRESH `RNNDownBeatProcessor()` per wav gives a wildly
different `float32_44100` activation (max abs diff ~0.14, nowhere near
ULP-scale) than the shared-instance-in-order version this fixture actually
records. **Any code (test or otherwise) reproducing these numbers MUST
replicate the same instance-reuse/call-order** -- see
`tests/test_downbeats_rnn.py`'s module header for the full write-up and its
tests' explicit in-order loops (not independent per-case fixtures).
