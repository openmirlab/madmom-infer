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
