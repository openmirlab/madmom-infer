# madmom-infer -- CLAUDE.md

## Scope and roadmap

madmom-infer is a from-scratch reimplementation of CPJKU/madmom's
inference-relevant algorithms (not training-only code, not
`madmom.evaluation.*`). Full scope and the 3-phase roadmap (Phase 1: DSP
pipeline + numpy Viterbi decoder; Phase 2: onset/tempo/chord/key/note
extraction + NN runtime, gated on pretrained-weights licensing; Phase 3:
remaining odds and ends) are documented in [README.md](./README.md) -- read
that first, don't re-derive the phasing here.

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
  permanent policy, not a phase-gate detail to relax later.
