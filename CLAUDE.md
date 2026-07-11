# madmom-infer -- CLAUDE.md

## Scope and roadmap

madmom-infer is a from-scratch reimplementation of CPJKU/madmom's
inference-relevant algorithms (not training-only code, not
`madmom.evaluation.*`). Full scope and the 3-phase roadmap (Phase 1: DSP
pipeline + numpy Viterbi decoder, complete; Phase 2: forward-pass-only NN
runtime + restricted model unpickling + runtime weights download +
`RNNDownBeatProcessor` end-to-end, complete -- the pretrained-weights
question is resolved as "never bundle, always download at runtime", see
`madmom_infer/models.py`; Phase 3: remaining odds and ends, incl. onset/
tempo/chord/key/note feature extraction beyond `RNNDownBeatProcessor`) are
documented in [README.md](./README.md) -- read that first, don't re-derive
the phasing here.

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

## Dual-backend + golden-fixture testing philosophy

This project follows the same pattern established by the sibling
all-in-one-infer package's pure-Python NATTEN replacement:

- A **numpy backend** is the default and required implementation, treated as
  the reference. It must be verified **bit-identical to original madmom**
  using golden fixtures -- recorded input/output pairs captured from running
  the real (compiled) madmom, checked against this port's output in tests.
  Do not consider a Phase 1/2/3 module "done" until it has a golden-fixture
  test, not just a hand-written unit test.
- An **optional torch backend** exists for GPU-accelerated batch processing,
  gated behind the `torch` extra. It is expected to help most at the
  spectrogram/STFT stage (trivially batches across frames) and to help little
  or not at all for the Viterbi decoder (inherently sequential). Don't
  oversell torch-backend speedups for sequential algorithms in docs or
  commit messages.
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
