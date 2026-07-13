# madmom-infer — plan

> 2026-07-13 · status index (one layer, by status). Only "what to do +
> which doc". Design in `thoughts/`, why in `decisions.md`.
>
> `plan.md` is the single maintained board. A visual view, if ever
> wanted, renders on demand via `/shape:mockup` — never a standing
> `overview.html` file (shape ADR-063).

## 🚧 In progress

None — the Phase 4 complete-port campaign just closed (merged to main,
`54c8bff`).

## ▶ Next —— 接下來

- **PyPI release of v0.3.0** — the Phase 4 campaign bumped the version
  but release pre-flight (tag/PyPI-existence checks, CI test gate, a
  wheel-from-sdist install smoke test) hasn't run. Outward-facing —
  needs explicit sign-off before publishing.
- **torch CNN inference backend** — Phase 4 shipped three CNN-based
  models (key detection, CNN onset, CNN notes) that have straightforward
  `torch.nn.Conv1d`/`MaxPool`/`BatchNorm` equivalents, unlike the
  LSTM/GRU path (see Future below). Worth scoping as its own wave now
  that the models exist to port against.

## ⏸ Future —— deferred

> Common blocker: madmom's LSTM uses peephole connections
> `torch.nn.LSTM` doesn't implement.

- **Phase 3b: torch forward pass for LSTM/GRU ensembles** — needs a
  custom cell, not a drop-in `nn.LSTM` swap. Blocked on that design
  question, not effort.
- **torch port of Viterbi/DBN/CRF/GMM decoding** — deliberately never:
  sequential, discrete-state recursion, no GPU/batching benefit to
  speak of. Stated project policy, not a backlog item.

## ✅ Shipped

Clean task-level API (`Analyzer` plus ten one-shot MIR tasks, shared input
normalization/resampling and lazy model reuse) and repair of the four confirmed
inherited defects (`correlation_diff`, `MFCC`, `MFCCProcessor.transform`,
`HPSS.process()`), Phase 4 complete-port campaign (key detection, onset detection, tempo
estimation, chord recognition, chroma, piano note transcription,
MFCC/cepstrogram, HPSS, CRF beat detection, GMM pattern tracking — v0.3.0,
302 offline tests, 21 cross-BLAS exactness proofs, zero TO-PORT rows
remaining), Phase 3a differentiable torch spectrogram frontend, Phase 2
NN runtime + `RNNDownBeatProcessor` end-to-end, Phase 1 DSP pipeline +
numpy Viterbi/DBN decoder. (detail in git log)
