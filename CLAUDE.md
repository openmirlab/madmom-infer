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
    layer classes no shipped model needs (no speculative GRU/TCN ports).
  - Weights discipline is unchanged: every new model family goes through
    `models.py`'s sha256-pinned runtime download, never bundled; the
    CC-BY-NC-SA weights license is one models-repo-wide fact and applies
    to key/chords/onsets/notes exactly as it does to the downbeat models.

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
original reference venv (`all-in-one-fix/.venv`, numpy 1.23.5 -- the same
technique `test_spectrogram.py` established in Phase 1) and asserts this
project's own `RNNDownBeatProcessor` -> `DBNDownBeatTrackingProcessor`
output is bit-identical to real madmom's, not just within a tolerance.
Regenerate the Phase-2 fixtures it and `test_ml_nn.py` depend on with:

```bash
/home/worzpro/Desktop/dev/openmirlab/all-in-one-fix/.venv/bin/python \
    tools/generate_phase2_fixtures.py
```

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
