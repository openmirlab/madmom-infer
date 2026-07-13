# madmom-infer — decisions

> Durable *why* layer. One curated file, feature-sections, each: the call ·
> how it shows up in the system · what was rejected. Normally graduated
> from `thoughts/` by `/shape:reconcile`; this first section was migrated
> directly from the repo's pre-existing `docs/DECISIONS.md` (2026-07-13,
> written before this `blueprints/` tree existed) since it already fit the
> shape. New entries follow the normal thought → graduate path.

## Bug-for-bug fidelity for confirmed upstream bugs

**The call**: when a port wave finds a real, confirmed bug in upstream
madmom, reproduce it exactly (pinned by a test expecting the same
failure) — never silently fix it. The product's whole claim is "behaves
exactly like real madmom, just installable," proven by golden-fixture
byte comparison; a silent fix breaks that claim in a way users can't
detect, and "fixed" is usually an undocumented API redesign in disguise
(deciding the *right* behavior is a real design question, not a
side-effect of porting).

**How it shows up**: three symbols ship broken-on-purpose, each with a
docstring citing the reference-venv proof and a pinning test —
`correlation_diff` (`features/onsets.py`, crashes under Python 3 in real
madmom too — a Py2 integer-division assumption — pinned via
`pytest.raises(TypeError)`, no golden output exists to record);
`MFCC` (`audio/cepstrogram.py`, only accepts an already-`FilteredSpectrogram`
input — every other input, including a raw wav path, raises
`AttributeError` in real madmom too); `HPSS.process()`
(`audio/hpss.py`, unconditionally broken for every input in every
version of madmom — the underlying `slices()`/`masks()` building blocks
are correct and verified bit-identical, and are the documented supported
path around the broken convenience method).

**What was rejected**: deleting `correlation_diff` outright (an
EXCLUDE-style removal, same class as the TCN exclusions) — rejected
because real madmom's `from madmom.features.onsets import
correlation_diff` succeeds at import time; deleting the symbol would turn
an import-time success into an import-time failure, a *worse*
compatibility break than the call-time crash already being reproduced.
A future "clean API surface" layer — new, differently-named convenience
functions that don't claim madmom parity, sitting alongside the
bug-for-bug originals — was discussed and is a live option, but not
scheduled (see `plan.md`).
