# madmom-infer — plan

> 2026-07-13 · status index (one layer, by status). Only "what to do +
> which doc". Design in `thoughts/`, why in `decisions.md`.
>
> **Deviation**: no `overview.html` — solo-maintained repo, zero human
> readers for a rendered board. `plan.md` is the sole maintained board;
> generate a human view on demand if one is ever actually wanted, don't
> pre-scaffold it.

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
- **Clean API surface for the bug-for-bug symbols** — `decisions.md`'s
  bug-for-bug entry; new, differently-named convenience functions
  (`madmom_infer.mfcc(path)` etc.) that don't claim madmom parity,
  sitting alongside the faithful originals. Discussed, not scheduled.

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

Phase 4 complete-port campaign (key detection, onset detection, tempo
estimation, chord recognition, chroma, piano note transcription,
MFCC/cepstrogram, HPSS, CRF beat detection, GMM pattern tracking — v0.3.0,
302 offline tests, 21 cross-BLAS exactness proofs, zero TO-PORT rows
remaining), Phase 3a differentiable torch spectrogram frontend, Phase 2
NN runtime + `RNNDownBeatProcessor` end-to-end, Phase 1 DSP pipeline +
numpy Viterbi/DBN decoder. (detail in git log)
