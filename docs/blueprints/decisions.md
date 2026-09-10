# madmom-infer — decisions

> Durable *why* layer. One curated file, feature-sections, each: the call ·
> how it shows up in the system · what was rejected. Normally graduated
> from `thoughts/` by `/shape:reconcile`; this first section was migrated
> directly from the repo's pre-existing `docs/DECISIONS.md` (2026-07-13,
> written before this `blueprints/` tree existed) since it already fit the
> shape. New entries follow the normal thought → graduate path.

## Fix inherited defects after migration

**The call**: migration is complete, so current `madmom-infer` behavior is
judged against this project's own public contract, reasonable API usage,
and result correctness. Upstream madmom remains useful as provenance for
algorithms, model formats, and the origin of inherited behavior, but
matching an upstream failure is not a reason to preserve it. A behavior is
a product bug when valid input violates the documented contract, a public
processor cannot perform its advertised operation, an accepted option is
silently ignored, or the implementation produces an incorrect result.

**How it shows up**: preserve migration assets that users and model files
depend on — import paths, public names where practical, processor
composition, serialized-model compatibility, and numerically validated
inference behavior. Fix confirmed defects in place instead of adding a
parallel "clean" namespace. The first confirmed repair set is
`correlation_diff` (valid input currently crashes because of Python 2
division semantics), `MFCC` (raw audio and a plain `Spectrogram` currently
fail despite being valid constructor inputs), `MFCCProcessor` (its stored
`transform` option is ignored), and `HPSS.process()` (every input currently
fails, violating the `Processor` contract). Tests for these symbols should
assert useful behavior and numerical correctness, not inherited
exceptions.

**What was rejected**: bug-for-bug fidelity as an ongoing product
principle, because it turns known defects into permanent API commitments
after the migration goal has already been achieved. A second clean API
beside deliberately broken legacy symbols was also rejected: it would
duplicate concepts, leave traps in the primary namespace, and make users
choose between two APIs without a product need. Upstream comparisons can
still be used as diagnostic evidence, but they are no longer the release
criterion for changed or newly fixed behavior.

## Let `tempo` ride on the downbeat ensemble, but only when asked

**The call**: `MadmomAnalyzer` gains `tempo_from_downbeat_activations`, default
`False`. When set (and both `tempo` and `downbeats` are selected), the `tempo` task
takes its beat activation from column 0 of `RNNDownBeatProcessor`'s `(n, 2)` output
instead of running `RNNBeatProcessor`. The saving is a whole eight-net BLSTM
ensemble: 11.8 s → 8.2 s for a `downbeats` + `onsets` + `tempo` analysis of a 64 s
file, i.e. about a third of the wall time, for zero change to the `downbeats` result.

**How it shows up**: `_analyze` memoizes the downbeat activations per call the same
way it already memoized beat activations and chroma, and `_build_processor("tempo")`
returns `None` under the flag so the second ensemble is never even loaded. The
constructor raises `ValueError` rather than silently no-opping when the flag is set
without both tasks — a caller who asks for this is making a deliberate accuracy
trade and should hear about a mis-configuration immediately.

**What was rejected**: (a) doing it unconditionally. The two activations are not
interchangeable. The joint beat/downbeat net emphasises a different periodicity;
under `DBNBeatTrackingProcessor` its column-0 curve tracks at *double* tempo (161
beats vs 80 on the same 64 s file), and measured across five music files from 13 s to
4 minutes the leading tempo candidate agreed on four and diverged on a
percussion-free vocal stem, with candidate *strengths* shifting on every file. A
library whose release gate is ULP-level parity must not move a default onto an
approximation. (b) Extending the flag to `beats` — that is precisely the case the
double-tempo evidence rules out; the flag covers `tempo` only, whose autocorrelation
is octave-tolerant. (c) Dropping the `tempo` task from the caller instead — same
saving, but it discards the tempo candidates and the octave-suspect warning
altogether, and an approximate diagnostic beats an absent one.

