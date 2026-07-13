# madmom-infer: decision log

Dated entries recording *why* a specific implementation choice was made,
written when the choice is non-obvious enough that a future contributor
(or agent) would otherwise have to re-derive it from the diff. Distinct
from `docs/DESIGN.md`, which is the pre-implementation architecture plan
(now mostly historical -- Phase 4 shipped well past what it scoped) --
this file is a decision log for choices made *during and after*
implementation, and it is expected to keep growing.

---

## 2026-07-13 -- Bug-for-bug fidelity: reproducing confirmed upstream bugs instead of fixing them

### The principle

madmom-infer's entire value proposition is "behaves exactly like real
madmom, just installable." That claim is proven with golden-fixture tests
-- every output is checked byte-for-byte (or to a documented ULP bound)
against a real, compiled madmom install. When the Phase 4 complete-port
campaign found real, confirmed bugs in upstream madmom itself, the
default was **reproduce the bug, pin it with a test, document it loudly
-- never silently fix it**.

Why not just fix bugs we find? Two reasons:

1. **A silent fix breaks the "exact replacement" claim in a way users
   can't detect.** If some outputs match real madmom and others are
   quietly "improved," a user migrating from real madmom has no way to
   tell which is which without re-deriving our diff themselves.
2. **"Fixed" is usually an API design decision in disguise**, not a
   mechanical patch -- see the `MFCC` case below. Deciding what the
   *right* behavior should have been is a real design question, not
   something to resolve as a side effect of porting.

The org constitution's article 2 says accuracy-affecting changes ship as
opt-in flags, default off, never silently. This is the same rule applied
to bugs discovered mid-port rather than deliberate algorithm changes.

### The three cases found (Phase 4 waves 4b, 4g)

**1. `correlation_diff` (features/onsets.py, wave 4b)** -- crashes
unconditionally in real madmom under Python 3. The bug: `centre = len(c) /
2` was written for Python 2, where `/` on two ints truncates; under
Python 3 it's true division, so `centre` becomes a `float`, and the next
line uses it as a slice index -> `TypeError`. Confirmed against the
rebuilt reference venv (`madmom-reference/.venv`, Python 3.10.18) that
real, compiled madmom 0.17.dev0 crashes on this too -- not a porting
artifact.

Why nobody ever noticed: the function is dead weight even in upstream's
own words -- its docstring says "not intended to be actually used ...
extremely slow" -- and no other madmom code path calls it. It has been
importable-but-broken for years without anyone filing an issue.

Ported verbatim (`madmom_infer/features/onsets.py`), pinned by
`test_correlation_diff_raises_typeerror` asserting the exact `TypeError`,
not a golden output (there isn't one to record -- it never produced one).

**2. `MFCC` (audio/cepstrogram.py, wave 4g)** -- `MFCC.__new__`'s "was
this spectrogram already filtered?" check unconditionally reads
`data.filterbank`. The base `Spectrogram` class never defines that
attribute (confirmed: `hasattr(plain_spectrogram, 'filterbank')` is
`False` on the reference venv), so `MFCC("song.wav")` and
`MFCC(plain_spectrogram)` both raise `AttributeError` in real madmom.
The *only* input that works is an already-constructed
`FilteredSpectrogram` -- which happens to have the attribute for
unrelated reasons.

Why nobody ever noticed: every real usage sits inside a pipeline where a
`FilteredSpectrogramProcessor` stage runs immediately before `MFCC`, so
the one input shape that works is exactly the one every example teaches.
The "obviously broken" direct-from-audio path was never the one anyone
actually used.

A second, smaller finding in the same class: `MFCCProcessor.__init__`
stores a `transform` parameter that `.process()` never forwards to the
`MFCC(...)` call it makes -- a dead parameter, also ported as-is.

Ported verbatim; the only non-crashing construction path is documented
explicitly in `MFCC.__init__`'s docstring rather than defended against.

**3. `HarmonicPercussiveSourceSeparation.process()` (audio/hpss.py, wave
4g)** -- unconditionally broken for *every* input: a `Spectrogram` input
hits `data.spec`, which doesn't exist (`AttributeError`); any other input
skips the assignment entirely and the next line references an unbound
name (`UnboundLocalError`). There has never been a working call to this
method in any version of madmom.

Why nobody ever noticed: `HPSS` was never wired into any `bin/` program
or other processor -- it shipped as an unused, uncalled convenience
class. The building blocks underneath it (`slices()`, `masks()`) are
correct and independently useful; this project verified those
bit-identical to real madmom and ships them as the supported path.

Ported verbatim; `process()`'s brokenness is pinned by tests expecting
the exact `AttributeError`/`UnboundLocalError`, with the working
`slices()`/`masks()` path documented as the actual way to use this class.

### The `correlation_diff` removal question, and why it stays

It's tempting to just delete `correlation_diff` outright rather than ship
a function nobody can call successfully -- unlike `MFCC`/`HPSS`, which
have at least one working path, this one has *zero*. The case for
removal: it's a clean EXCLUDE (same category as the TCN models: no
reachable code path, upstream's own docs disclaim it), and the org
constitution already has a `bin/`/evaluation-style exclusion mechanism
for exactly this.

**Decision: keep it.** The deciding factor is import-time compatibility,
not call-time behavior. In real madmom, `from madmom.features.onsets
import correlation_diff` **succeeds** -- the module imports cleanly; the
function only crashes if you call it. Migration code that does a broad
`import *` or a compatibility shim (without ever calling this specific
function) works today against real madmom and must keep working against
madmom-infer. Deleting the symbol would turn an import-time success into
an import-time failure -- a *worse* compatibility break than the
call-time crash it already reproduces. The cost of keeping it is small
(35 lines + one pinning test); the benefit of matching import-time
behavior exactly is the whole point of this project.

This also keeps the project internally consistent with two prior
precedents already shipped: `SimpleChromaFilterbank` and
`HarmonicFilterbank` (wave 4d/4g) are both upstream stubs that
unconditionally `raise NotImplementedError` -- ported the same way, for
the same reason. If `correlation_diff` were removed, those two would need
removing too for consistency, which would mean deviating from upstream's
*importable* surface in three places instead of reproducing it faithfully
everywhere.

### If a clean API surface is wanted later

The bug-for-bug symbols aren't a dead end if someone eventually wants
`madmom-infer` to be more ergonomic than madmom itself. The correct shape
for that, discussed but **not decided or scheduled**:

- Leave the existing symbols (`correlation_diff`, `MFCC`, `HPSS.process`)
  exactly as they are -- still bit-for-bit/behavior-for-behavior matched
  to real madmom, still covered by their pinning tests.
- Add new, clearly-different-named convenience entry points (e.g. a
  top-level `madmom_infer.mfcc(path)` helper) that do NOT claim madmom
  parity -- their docstrings would say plainly "this is a madmom-infer
  addition, not upstream behavior," and they would NOT be covered by the
  golden-fixture bit-identity discipline the way the parity surface is.

This preserves the verified-parity guarantee for the existing surface
while giving a future contributor room to build something more usable on
top, without the two ever being confused for each other.
