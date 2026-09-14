# madmom-infer — plan

> 2026-09-14 · status index (one layer, by status). Only "what to do +
> which doc". Design in `thoughts/`, why in `decisions.md`.
>
> `plan.md` is the single maintained board. A visual view, if ever
> wanted, renders on demand via `/shape:mockup` — never a standing
> `overview.html` file (shape ADR-063).

## 🚧 In progress

None — the torch backend (Phase 3b) just merged to main.

## ▶ Next —— 接下來

- **PyPI release** — v0.3.0's release pre-flight (tag/PyPI-existence
  checks, CI test gate, a wheel-from-sdist install smoke test) never ran,
  and the torch backend now sits in `[Unreleased]`. Outward-facing —
  needs explicit sign-off before publishing.

## ⏸ Future —— deferred

- **Soft (differentiable) decoding** — forward-backward / smoothed-max
  versions of the DBN (then CRF, ADSR HMM) so tempo/meter structure can
  produce gradients, not just frame activations. Explicitly deferred
  2026-09-14 until a training use case needs it; would start with the
  downbeat DBN, validated by recovering the Viterbi path as temperature
  goes to 0. Memory-heavy (thousands of states × frames).
- **Chunked torch pipelines for long audio** — torch pipelines
  materialize whole-file tensors; a 4.5 min file measured fine on CUDA
  (4 GiB peak for downbeats) but long-file torch-on-CPU memory was never
  measured.
- **torch port of hard Viterbi/DBN/CRF/GMM decoding** — deliberately
  never: sequential, discrete-state recursion, no GPU/batching benefit
  to speak of. Stated project policy, not a backlog item.

## ✅ Shipped

- **Opt-in torch backend (Phase 3b)** — `backend="torch", device=` on the
  ten NN processors and `MadmomAnalyzer`; differentiable
  waveform-to-activation pipelines (`madmom_infer.torch.build_pipeline`,
  `to_torch(trainable=True)`); stacked/gate-fused recurrent ensembles.
  Decoded results identical to numpy on real music (CPU and CUDA);
  7-task analysis 13.5 s → 4.4 s on 30 s audio, CNN activations
  hundreds-to-thousands × faster on GPU. Why in `decisions.md`.

- **`tempo_from_downbeat_activations`** — opt-in on `MadmomAnalyzer` letting
  `tempo` read the downbeat ensemble's beat column instead of running a
  second eight-net BLSTM ensemble (−31% on a `downbeats`+`onsets`+`tempo`
  analysis). Deliberately not the default; why in `decisions.md`.

Clean task-level API (`MadmomAnalyzer` plus ten one-shot MIR tasks, shared input
normalization/resampling and lazy model reuse) and repair of the four confirmed
inherited defects (`correlation_diff`, `MFCC`, `MFCCProcessor.transform`,
`HPSS.process()`), Phase 4 complete-port campaign (key detection, onset detection, tempo
estimation, chord recognition, chroma, piano note transcription,
MFCC/cepstrogram, HPSS, CRF beat detection, GMM pattern tracking — v0.3.0,
302 offline tests, 21 cross-BLAS exactness proofs, zero TO-PORT rows
remaining), Phase 3a differentiable torch spectrogram frontend, Phase 2
NN runtime + `RNNDownBeatProcessor` end-to-end, Phase 1 DSP pipeline +
numpy Viterbi/DBN decoder. (detail in git log)
