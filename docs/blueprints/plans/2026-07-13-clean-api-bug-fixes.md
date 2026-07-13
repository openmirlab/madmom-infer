# Clean API and inherited bug fixes — plan

> Generated: 2026-07-13 · Spec source: user discussion and
> `docs/blueprints/decisions.md` · Stage 1: fresh grounding against the
> audio, onset, processor, and test domains

## Context

The four confirmed defects live in three existing public modules.
`correlation_diff` in `madmom_infer/features/onsets.py` uses Python 2 integer
division semantics and crashes on valid Python 3 input. `MFCC` in
`madmom_infer/audio/cepstrogram.py` assumes every `Spectrogram` has filtering
attributes, while `MFCCProcessor` stores but does not forward its `transform`.
`HPSS.process()` in `madmom_infer/audio/hpss.py` cannot process any input even
though it implements the package's standard `Processor` interface.

These symbols are not called by the current beat, downbeat, onset-model, key,
chord, tempo, or note inference pipelines. They can therefore be repaired in
place without changing those pipelines. Their public names and signatures are
still part of the useful madmom-shaped API, so they remain; only the inherited
failure behavior is removed. There will be no legacy mode or legacy namespace.

The package currently has no high-level feature facade: users must manually
compose audio classes, activation models, and decoders. The clean API will be
a thin task layer over the existing processors, not a second implementation.
It covers the general MIR vocabulary: onsets, beats, downbeats, tempo, key,
chords, notes, chroma, MFCC, and HPSS. Research-specific pattern/bar tracking
and raw activations remain in the advanced API.

## Resolved questions

| Question | User's answer |
|---|---|
| Add high-level `mfcc()` and `hpss()` entry points? | Yes; retain class-based APIs for advanced composition. |
| Accepted inputs | File paths, audio arrays, and corresponding `Spectrogram` objects. Include a `sample_rate` keyword; require it for bare arrays, infer it from paths or metadata-bearing objects when omitted, and reject conflicting metadata. |
| High-level return values | `mfcc()` returns a 2-D NumPy array; `hpss()` returns `(harmonic, percussive)` NumPy arrays. |
| Legacy behavior | Do not retain broken legacy behavior, flags, or namespaces. Keep established public names/signatures where they are the main madmom-style API and fix them in place. |
| Task scope | Include the ten general MIR tasks; keep pattern/bar tracking and raw activations in the advanced processor API. |
| Repeated analysis | Provide `Analyzer` with lazy model reuse and shared beat/chroma intermediates, plus one-shot task functions. |
| Algorithm choices | One canonical pipeline per clean task; CNN/RNN/DBN/CRF choices remain in the advanced API. |
| Non-44.1 kHz audio | Resample inside the clean input boundary; arrays require an explicit source rate. |

## Approach

1. Repair the four existing APIs in place.
   - Change `correlation_diff`'s correlation midpoint to explicit integer
     division and retain its current arguments and output shape.
   - Make `MFCC` inspect optional filtering/scaling attributes defensively so
     raw audio and plain `Spectrogram` inputs follow the documented Mel -> log
     -> DCT path. Preserve the existing warning/recompute behavior for inputs
     that really were filtered or scaled already.
   - Forward `MFCCProcessor.transform` into `MFCC`.
   - Make `HPSS.process()` normalize a `Spectrogram` or 2-D array with
     `np.asarray`, then compose the already-correct `slices()` and `masks()`
     methods. Reject non-2-D input with a clear `ValueError`.
   - Rewrite bug-for-bug module/docstring comments as current behavior and
     rationale; do not leave comments that claim failures are intentional.

2. Replace inherited-failure tests with product-contract tests.
   - For `correlation_diff`, cover normal output, `pos=True`, frame/bin offsets,
     and existing invalid-argument handling using a small independent reference
     calculation.
   - For `MFCC`, prove path, plain `Spectrogram`, and pre-filtered inputs work;
     prove a custom processor transform is called and affects output; retain
     numerical fixture coverage for the algorithm itself.
   - For HPSS, prove both `Spectrogram` and 2-D array inputs return two arrays
     of the original shape, reconstruct the input within numerical tolerance,
     and match direct `slices()`/`masks()` composition for binary and soft masks.

3. Add one clean high-level facade in `madmom_infer/api.py`.
   - Add one-shot functions for all ten tasks plus a reusable, lazy `Analyzer`.
   - Centralize input normalization in one private helper: bare NumPy arrays
     require `sample_rate`; paths use the file's native rate when omitted;
     metadata-bearing `Signal`/`Spectrogram` inputs use their own rate and reject
     an explicitly conflicting value.
   - Resample clean-API audio to each canonical pipeline's required rate.
   - Reuse beat activations for beats+tempo and chroma for chroma+chords within
     an `Analyzer` call; lazily retain loaded model processors across calls.
   - Re-export the task vocabulary from `madmom_infer/__init__.py`; keep torch
     optional by importing only NumPy/SciPy-backed modules.

4. Test the facade as the user-facing contract, including every canonical
   end-to-end task pipeline.
   - Cover paths, arrays with a sample rate, and existing `Spectrogram` inputs.
   - Assert a bare array without `sample_rate` and conflicting rate metadata
     fail with clear messages.
   - Assert facade outputs equal direct class-based calls, proving the clean API
     is only a front door and not a divergent implementation.
   - Add a fresh-interpreter import smoke test showing `import madmom_infer`
     does not import or require torch.

5. Update documentation and remove stale compatibility framing.
   - Replace README workarounds with direct clean-API examples that always show
     `sample_rate` explicitly, plus one short advanced class-based example.
   - Update affected module headers and test descriptions so upstream remains
     provenance, not the expected failure oracle.
   - Keep historical fixture-generation notes only where they explain numerical
     validation; remove instructions that teach users to route around the bugs.

## Critical files

| File | Why it matters | Touched in step |
|---|---|---|
| `madmom_infer/features/onsets.py:111` | Owns `correlation_diff` and its Python 3 indexing defect. | 1 |
| `madmom_infer/audio/cepstrogram.py:165` | Owns MFCC construction and processor option forwarding. | 1 |
| `madmom_infer/audio/hpss.py:121` | Owns the broken standard `Processor` entry point. | 1 |
| `tests/test_onsets.py:242` | Currently pins the inherited exception. | 2 |
| `tests/test_cepstrogram.py:97` | Currently pins MFCC failures and ignored transforms. | 2 |
| `tests/test_hpss.py:64` | Currently pins both unconditional HPSS failures. | 2 |
| `madmom_infer/api.py` | New single owner for clean input normalization and convenience calls. | 3 |
| `madmom_infer/__init__.py:16` | Small public facade export surface; must remain torch-free. | 3 |
| `tests/test_api.py` | New end-user contract tests for paths, arrays, metadata, and imports. | 4 |
| `README.md:424` | Currently documents workarounds and unusable behavior. | 5 |

## Single-source-of-truth owners

| Decision | Owner |
|---|---|
| Path/array/metadata-bearing input normalization and sample-rate validation | Private helper in `madmom_infer/api.py` |
| MFCC algorithm and options | `madmom_infer.audio.cepstrogram.MFCC` |
| HPSS slicing, masking, and separation | `madmom_infer.audio.hpss.HPSS` |
| Python-level correlation-difference algorithm | `madmom_infer.features.onsets.correlation_diff` |

Step 3 explicitly adopts the facade's normalization helper in both high-level
functions. Step 4 verifies both functions exercise it, preventing a nominal
owner with duplicated per-function validation.

## Verification

1. Step 1 -> run the focused onset, cepstrogram, and HPSS tests; inspect that no
   test still expects the four inherited failures.
2. Step 2 -> run
   `python -m pytest -q tests/test_onsets.py tests/test_cepstrogram.py tests/test_hpss.py`.
3. Step 3 -> inspect public signatures and run a fresh Python process importing
   `madmom_infer`, `mfcc`, and `hpss` without torch installed.
4. Step 4 -> run `python -m pytest -q tests/test_api.py` and compare every facade
   result with its direct class-based equivalent.
5. Step 5 -> run `rg` for stale phrases such as `bug-for-bug`, `NOT usable`, and
   `raises AttributeError` in the affected docs and module headers, keeping only
   historical references that are explicitly useful provenance.

End-to-end: run the complete offline suite with `python -m pytest -q`, then run
the existing representative beat/downbeat, key, onset-model, chord, tempo, and
note inference smoke tests to prove the repaired standalone APIs did not alter
the main model pipelines.

## Out of scope

- Changing model weights, decoders, or the main pretrained inference pipelines.
- Adding torch implementations of MFCC or HPSS.
- Preserving an opt-in bug-compatible mode or adding a legacy namespace.
- Treating exact upstream failure behavior as an acceptance criterion.
