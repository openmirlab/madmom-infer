# madmom-infer: dual-backend architecture, differentiability audit, modernization plan

This document proposes how to build `madmom-infer` around the three missions defined for
the project:

1. **Modernize** -- installable on Python 3.10-3.13+ with zero compilation, numpy-2.x
   compatible, maintainable.
2. **Infer-only** -- no training code, no `madmom.evaluation.*`, no bundled pretrained
   weights.
3. **Differentiable torch backend** -- go beyond a speed backend: make parts of the
   pipeline genuinely backprop-through-able.

All claims about madmom's own code cite `madmom-upstream/<path>:<line>` against the fresh
clone at `/home/worzpro/Desktop/dev/openmirlab/madmom-upstream` (models/ submodule
deliberately not initialized -- weights are CC BY-NC-SA and out of scope forever, per
`README.md` / `NOTICE`). This design doc assumes the reader has read this repo's
`README.md` and `CLAUDE.md` (phasing, dual-backend philosophy, golden-fixture testing
philosophy) -- it does not re-derive that framing, only builds on it.

## Executive summary

madmom's phase-1 pipeline (`Signal` -> `FramedSignal` -> STFT -> filterbank -> log -> DBN
Viterbi decode) is, with one exception, plain numpy/scipy arithmetic: framing is array
slicing, STFT is a per-frame FFT, filtering is a matrix multiply (`np.dot(spectrogram,
filterbank)`, `madmom/audio/spectrogram.py:382`), and log-compression is
`mul`/`add`/`log10`. The one exception is `ml/hmm.pyx`'s Viterbi decoder, a Cython triple
loop over frames/states/CSR-sparse predecessors -- this is also the only stage that is
*not* differentiable by construction (hard argmax).

We recommend:

- **(A)** Keep the `Processor`/`SequentialProcessor` composition idiom for the numpy
  reference backend and an optional torch-*exact* backend (same class names, same
  `.process()`/`__call__` contract, 1:1 import-path parity with madmom so
  all-in-one-infer's swap is a dependency-name change, not a rewrite). Put the
  differentiable-relaxation code in a *separately named* `madmom_infer.torch.diff`
  namespace using plain `nn.Module`/function idioms (not `Processor`) so it reads as
  distinct code, for a distinct audience (researchers building training loops), rather
  than a backend flag on the same class.
- **(B)** Tiered differentiability: the spectrogram chain is trivially and fully
  differentiable in torch (STFT, filterbank matmul, log are all standard autograd ops).
  The DBN decoder is not, and should not be silently made "soft" -- ship (i) a
  torch-exact `viterbi()` that mirrors the Cython version bit-for-bit (hard, GPU
  gains marginal, sequential recursion), and separately (ii) research-tier
  `soft_viterbi()`/`forward()`-as-proxy relaxations, clearly renamed and re-homed so
  nobody mistakes a soft decode for the real one.
- **(C)** `Signal`/`Filterbank`/`Filter` are `np.ndarray` subclasses in madmom
  (`madmom/audio/signal.py:506`, `madmom/audio/filters.py:413,692`). We recommend
  **composition, not subclassing**, for madmom-infer -- ndarray subclassing is fragile
  (madmom's own `__array_finalize__`/pickle-`__reduce__` machinery,
  `signal.py:634-656`, exists purely to fight that fragility) and, more importantly, has
  no torch equivalent (torch tensors cannot be numpy-subclassed), so composition is the
  only shape that lets the numpy and torch backends look structurally alike. Golden
  fixtures should be generated from the *already-working* madmom 0.17.dev0 install in
  `all-in-one-fix/.venv` (Python 3.10.18, numpy 1.23.5, scipy 1.15.3) -- no new
  environment needs to be built to start this.
- **(D)** First three phase-1 tasks, two of them parallelizable immediately: (1) golden
  -fixture generation harness against the existing madmom env, (2) `Processor`/
  `SequentialProcessor` base classes, (3) `Signal`/`FramedSignalProcessor` port. The
  `ml/hmm.py` Viterbi port is the highest-risk, highest-effort task and can run as its
  own parallel workstream against the spectrogram-chain work, since the two share no
  code.

---

## A. Architecture for the dual/triple backend

### A.1 How madmom's own composition works (the thing we're deciding whether to keep)

`Processor` is a one-method abstract base: `process(self, data, **kwargs)` raises
`NotImplementedError` by default (`madmom/processors.py:90-108`), and `__call__` just
forwards to it (`madmom/processors.py:112-114`):

```python
def __call__(self, *args, **kwargs):
    # this magic method makes a Processor callable
    return self.process(*args, **kwargs)
```

`SequentialProcessor` (`madmom/processors.py:288-388`) is a `MutableSequence` of
`Processor`s whose own `process()` is a plain fold:

```python
def process(self, data, **kwargs):
    for processor in self.processors:
        data = _process((processor, data, kwargs))
    return data
```
(`madmom/processors.py:399-410`, elided the multiprocessing dispatch in `_process`)

This is the entire idiom: no dependency injection, no config objects, no graph compiler
-- a processor is a callable, and a pipeline is a list of callables applied in order.
`all-in-one-infer`'s `build_spec_processor()` (`all-in-one-fix/src/allin1_infer/
spectrogram.py:27-40`) is a direct, unmodified instance of this pattern:
`SequentialProcessor([frames, stft, filt, spec])`.

### A.2 The decision: same-class backend switch vs. separate namespace vs. array-API-agnostic core

Three options were on the table:

1. **Same class names, backend switch** -- like all-in-one-infer's own `NA_BACKEND`
   pattern in `dinat.py:8-44`: import-time `try/except` picks whichever kernel is
   available (`natten` fused kernel vs. pure-torch fallback), both exposed under the
   *same* function names (`na1d_av`, `na1d_qk`, ...), invisibly to the caller.
2. **Separate `madmom_infer.torch` namespace**, mirroring the numpy tree 1:1.
3. **Array-API-agnostic core** (Python array API standard / `array-api-compat`), writing
   the DSP math once against `xp.stft`/`xp.matmul`/etc. and letting numpy or torch flow
   through underneath.

**Recommendation: option 2 (separate namespace), with the `Processor` idiom kept for
"faithful port" code and dropped for "differentiable relaxation" code.** Reasoning:

- The `NA_BACKEND` precedent (option 1) works *because* both of its backends are
  numerically identical drop-in kernels -- the whole point is the caller shouldn't be
  able to tell which one ran. madmom-infer's backends are not interchangeable in that
  way: numpy-reference and torch-exact must be bit-identical to original madmom, but a
  *differentiable* torch mode is explicitly allowed (indeed required, mission 3) to use
  soft relaxations that produce **different numbers by design**. Silently auto-selecting
  between "exact" and "soft" the way `NA_BACKEND` auto-selects between "fused" and
  "pure-torch" would violate the sibling project's hard rule ("accuracy cannot drop")
  and the task's explicit honesty constraint ("must be impossible to miss"). A same-name
  switch is the wrong shape for a distinction this load-bearing.
- Array-API-agnostic core (option 3) is attractive for the pure-elementwise/matmul
  spectrogram stages, but breaks down hard at the decoder: `TransitionModel` is built via
  `scipy.sparse.csr_matrix` (`madmom/ml/hmm.pyx:196-207`) and `viterbi()`'s inner loop is
  a ragged, variable-degree-per-state scan over CSR row-slices
  (`madmom/ml/hmm.pyx:526-556`) -- there is no array-API-standard sparse type, no
  standard segment-max/scatter-max op, and the `signal_frame()` padding logic
  (`madmom/audio/signal.py:860-962`, see B.1) does dtype-preserving in-place assignment
  into a freshly allocated buffer, which doesn't translate cleanly either. An
  array-API core would end up being numpy-shaped code with occasional escape hatches
  into backend-specific branches anyway -- it buys uniformity for 60% of the pipeline at
  the cost of a debugging abstraction layer for the other 40%. Not worth it here.
- Keeping the `Processor` idiom for the numpy-reference and torch-*exact* backends
  directly serves the phase-1 acceptance test: all-in-one-infer's own code (`spectrogram.
  py:27-40`) already builds `SequentialProcessor([frames, stft, filt, spec])` --
  if `madmom_infer.audio.signal.FramedSignalProcessor` etc. keep the same constructor
  signatures and `Processor` contract, swapping the dependency is a two-line import
  change, not a pipeline rewrite. This is explicitly the phase-1 acceptance test (task D).
- The differentiable-relaxation code (`madmom_infer.torch.diff.*`) should **not** use
  `Processor` -- its audience is torch-native researchers building training loops who
  expect `nn.Module`/plain-function/autograd idioms (parameters, `.to(device)`,
  `requires_grad`), not a `process(data, **kwargs)` call convention carried over from a
  2013-era MIR toolkit. Using a different idiom here is itself a signal that this is a
  different kind of code, reinforcing the separation from A's exactness distinction.

### A.3 Proposed module structure

```
madmom_infer/
  processors.py                  # Processor, SequentialProcessor (numpy world; unchanged idiom)
  audio/
    signal.py                    # Signal (composition, see C), FramedSignal(Processor)
    stft.py                      # ShortTimeFourierTransformProcessor
    filters.py                   # Filterbank construction (fixed matrices)
    spectrogram.py                # FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor
  ml/
    hmm.py                       # HiddenMarkovModel.viterbi()/forward(), numpy, log-domain
  features/
    beats_hmm.py                  # BarStateSpace/BarTransitionModel construction
    downbeats.py                  # DBNDownBeatTrackingProcessor (phase-1 top-level target)
  torch/                          # optional, gated by the `torch` extra
    __init__.py                   # raises a clear ImportError if torch isn't installed
    audio/
      signal.py, stft.py, filters.py, spectrogram.py
                                   # torch-EXACT accelerated mirrors: same class names as
                                   # madmom_infer.audio.*, same Processor contract, batched
                                   # across frames (see B.1); output must match the numpy
                                   # reference within float32 tolerance, verified by the
                                   # same golden fixtures used for the numpy backend.
    ml/
      hmm.py                       # torch-EXACT viterbi()/forward(): hard decode, GPU
                                   # gains expected marginal (sequential recursion, see B.2)
    diff/                          # torch-DIFFERENTIABLE relaxations -- deliberately
      spectrogram.py                # not Processor-based; nn.Module / plain functions.
      hmm.py                        # soft_viterbi(), forward()-as-differentiable-proxy.
                                    # Every public name here is prefixed/suffixed so it
                                    # cannot be mistaken for the exact decoder (B.3).
```

`madmom_infer.torch.audio.*` and `madmom_infer.torch.diff.*` are both optional (behind
the `torch` extra already in `pyproject.toml`); `madmom_infer.torch.diff` should also be
its own extra-gated import path (e.g. still under `torch`, but documented as "research,
not for inference") so a user who only wants GPU-accelerated *exact* inference never
has to know the differentiable module exists.

### A.4 Migration table (madmom -> madmom_infer, phase-1 API)

| madmom (original) | madmom_infer (phase 1) |
|---|---|
| `madmom.audio.signal.Signal` | `madmom_infer.audio.signal.Signal` |
| `madmom.audio.signal.FramedSignalProcessor` | `madmom_infer.audio.signal.FramedSignalProcessor` |
| `madmom.audio.stft.ShortTimeFourierTransformProcessor` | `madmom_infer.audio.stft.ShortTimeFourierTransformProcessor` |
| `madmom.audio.filters.*Filterbank` | `madmom_infer.audio.filters.*Filterbank` |
| `madmom.audio.spectrogram.FilteredSpectrogramProcessor` | `madmom_infer.audio.spectrogram.FilteredSpectrogramProcessor` |
| `madmom.audio.spectrogram.LogarithmicSpectrogramProcessor` | `madmom_infer.audio.spectrogram.LogarithmicSpectrogramProcessor` |
| `madmom.processors.Processor` / `SequentialProcessor` | `madmom_infer.processors.Processor` / `SequentialProcessor` |
| `madmom.ml.hmm.HiddenMarkovModel`/`TransitionModel` | `madmom_infer.ml.hmm.HiddenMarkovModel`/`TransitionModel` |
| `madmom.features.beats_hmm.BarStateSpace`/`BarTransitionModel` | `madmom_infer.features.beats_hmm.*` |
| `madmom.features.downbeats.DBNDownBeatTrackingProcessor` | `madmom_infer.features.downbeats.DBNDownBeatTrackingProcessor` |
| n/a (no madmom equivalent) | `madmom_infer.torch.audio.*`, `madmom_infer.torch.ml.hmm` (exact, optional) |
| n/a (no madmom equivalent) | `madmom_infer.torch.diff.*` (differentiable, optional, research-tier) |

The import-path mapping is a pure top-level package rename (`madmom.` -> `madmom_infer.`)
for every phase-1 symbol -- this is by design (A.2/A.3): all-in-one-infer's
`from madmom.audio.signal import FramedSignalProcessor, Signal` becomes
`from madmom_infer.audio.signal import FramedSignalProcessor, Signal` and nothing else in
that file needs to change (`all-in-one-fix/src/allin1_infer/spectrogram.py:19-22`).

---

## B. Differentiability audit per pipeline stage

### B.1 Spectrogram chain -- trivially differentiable, with three specific gotchas

The chain is: `Signal` -> `signal_frame()` slicing -> window multiply -> FFT
-> `np.dot(spectrogram, filterbank)` -> `mul`/`add`/`log`. Every one of these ops has a
direct, standard autograd equivalent (`torch.stft` or manual frame+window+`torch.fft.rfft`,
`torch.matmul`, `torch.log10`). Three details need explicit handling, though:

**(1) Framing/padding semantics.** `signal_frame()` (`madmom/audio/signal.py:860-962`)
computes, per frame index `i`: `ref_sample = int(i * hop_size)`,
`start = ref_sample - frame_size // 2 - origin`, `stop = start + frame_size`
(`signal.py:912-920`), then either a plain slice (`signal.py:930`) or, if the frame
overflows the signal boundary, builds a padded buffer from `np.repeat(signal[:1],
frame_size, axis=0)` and back-fills with a `pad` value (default `0`) or edge-repetition
(`signal.py:932-962`). `FramedSignal.num_frames` is `ceil(len(signal)/hop_size)` for the
default `end='normal'`, or `floor(len(signal)/hop_size + 1)` for `end='extend'`
(`signal.py:1144-1156`). None of this is learned or gradient-relevant -- it is pure
indexing -- so a torch port can implement it as `torch.nn.functional.pad` +
`.unfold(dim, frame_size, hop_size)` (both differentiable index/view ops) with no
special-casing needed, as long as the *same* start/stop/pad-value arithmetic is
reproduced exactly for numeric parity with the numpy reference. `origin` translation
(`'center'/'offline'->0`, `'left'/'past'/'online'->(frame_size-1)/2`,
`'right'/'future'/'stream'->-(frame_size/2)`, `signal.py:1125-1142`) is likewise pure
control flow, not math to differentiate.

**(2) The Hanning window and int16-scaling convention.** madmom's default window is
`np.hanning` (`madmom/audio/stft.py:311,317,380,472`), a fixed buffer -- there is no
reason to ever want gradients *through* the window's shape itself (it's not a learned
parameter in madmom's own pipeline), so it should be treated as a constant buffer in the
torch port, exactly like the filterbank (below). The subtler point: madmom does **not**
rescale the raw signal before FFT. Because `Signal` defaults to `dtype=None`, i.e.
"keep whatever dtype the source has" (`signal.py:503`, `DTYPE = None`), and wav files load
as native `int16` PCM by default (`madmom/io/audio.py:642-652`, using
`scipy.io.wavfile.read(filename, mmap=True)`), `FramedSignal` frames stay `int16`
end-to-end (`signal.py:934-938`, confirmed by the module's own doctest,
`signal.py:1055-1056`). Instead, `ShortTimeFourierTransform.__new__` divides the
*window* by `np.iinfo(frames.signal.dtype).max` when the signal has an integer dtype
(`stft.py:339-349`), documented explicitly at `stft.py:230-238` ("this results in same
valued STFTs independently of the dtype of the signal ... prevents extra memory
consumption since the data-type of the signal does not need to be converted"). The
actual multiply that upcasts int16 x float-window -> float happens inside `stft()`
(`stft.py:105-131`). This matches how all-in-one-infer feeds madmom: its
`quantize_stem_to_madmom_mono_int16` (`all-in-one-fix/src/allin1_infer/stems.py:245-262`)
produces a real int16 PCM array (`(wav.clamp(-1,1) * 32768).round().clamp(-32768,
32767).to(torch.int16)`) and passes it straight into `Signal(...)` with no pre-division
-- consistent with madmom's own convention of scaling the window, not the signal. This is
a constant multiplicative factor (`1/32767` folded into the window buffer), so it does
not interact with gradients in any interesting way for the **exact** backends -- it just
needs to be reproduced bit-for-bit. For the **differentiable** backend, we recommend
*not* carrying this convention forward: accept float32 signal in `[-1, 1]` directly (the
natural tensor a training loop already holds, before any int16 quantization step) and
skip the `iinfo`-based window rescale entirely (fold no extra factor into the window).
This is a deliberate, documented API difference between the exact and differentiable
paths, motivated by the fact that the int16 convention exists only to avoid a memory-map
-defeating float copy (`io/audio.py`'s docstring at line ~600) -- a concern that doesn't
apply to a tensor a caller already holds in memory for backprop.

**(3) The filterbank is a fixed matrix, not something to put gradients through --
usually.** `Filterbank` is `(num_bins, num_bands)`
(`madmom/audio/filters.py:692-701`) and gets applied via `np.dot(spectrogram,
filterbank)` (`madmom/audio/spectrogram.py:382,1321`). madmom builds this matrix once
from fixed center-frequency formulas (mel/log/bark spacing, `MelFilterbank`/
`LogarithmicFilterbank`/`BarkFilterbank`, `filters.py:1035-1240`) and never updates it.
For madmom-infer's phase-1 scope this should stay a constant buffer (a
non-`requires_grad` tensor) in both the exact and differentiable torch backends --
there is no madmom use case that trains the filterbank shape itself. We flag this
explicitly, though, because it is the one place in the chain where "could you want
gradients through this" is a real, non-rhetorical question (e.g. a research use case
that wants to *learn* filter center frequencies) -- if that ever becomes a phase-2+ ask,
`madmom_infer.torch.diff.spectrogram` is the natural home for a
`requires_grad=True` filterbank variant, kept out of the exact backends.

**One more subtlety for a torch port specifically:** `LogarithmicSpectrogram.__new__`
does `log(data, data)` (`madmom/audio/spectrogram.py:534`) -- an explicit in-place numpy
ufunc call (`out=data`). A literal torch translation must use the out-of-place
`torch.log10(data)` (or `.log10()`, not `.log10_()`) wherever the tensor participates in
an autograd graph, since in-place ops on tensors with graph history are restricted/
error-prone in torch. This is a one-line but easy-to-miss porting trap.

### B.2 Viterbi/DBN decoding -- hard argmax, not differentiable by construction

`HiddenMarkovModel.viterbi()` (`madmom/ml/hmm.pyx:481-585`) is a log-domain, Cython
`cdef`-typed triple loop: for each frame, for each state, scan that state's CSR-sparse
predecessor segment (`tm_pointers[state]:tm_pointers[state+1]`, mirroring the
`TransitionModel` docstring's `states[pointers[s]:pointers[s+1]]` convention,
`hmm.pyx:32-58`) and keep a running max + backpointer
(`hmm.pyx:526-559`, the `if transition_prob > current_viterbi[state]` comparison at
line 555). Backtracking (`hmm.pyx:561-585`) walks the stored backpointers from the final
frame's argmax state. This is exactly a per-frame **segment-max/scatter-max keyed by CSR
row** -- vectorizable in numpy (frame loop stays sequential; the state/predecessor double
loop can become a masked-max or `np.maximum.at`-style reduction), but the argmax itself is
a hard, non-differentiable operation regardless of language.

Three tiers of relaxation, assessed for applicability:

- **(i) `forward()` (sum-product) as a differentiable proxy.** madmom already ships this
  algorithm: `forward()` (`hmm.pyx:591-659`) is explicitly documented as *not* log-domain
  ("instead of computing in the log domain, we normalise at each step, which is faster",
  `hmm.pyx:592-595`) -- it sums over the same CSR predecessor segments
  (`hmm.pyx:638-650`) that `viterbi()` maxes over, then renormalizes per frame
  (`hmm.pyx:651-656`). Structurally this is a sum-product / forward-algorithm recursion,
  which is smooth and directly differentiable (every op is `+`/`*`/`/`). It computes
  state *occupancy* probabilities, not a best path, so it's a genuine proxy, not a
  drop-in Viterbi replacement -- appropriate for training losses defined on marginal
  beat-probability rather than a hard decoded sequence.
- **(ii) Entropy-regularized / soft-max DP (Mensch & Blondel 2018-style).** Replace the
  hard `max` in `viterbi()`'s inner comparison (`hmm.pyx:552-556`) with a
  temperature-scaled `logsumexp` over the same CSR predecessor segment:
  `soft_max_gamma(x) = gamma * logsumexp(x / gamma)`, recovering the hard Viterbi path as
  `gamma -> 0`. This is the direct differentiable generalization of the *actual* decode
  (not just a proxy quantity), and would be `soft_viterbi()` in
  `madmom_infer.torch.diff.hmm` (B.3). It is real research-engineering work: the ragged,
  per-state variable-degree CSR segments (`hmm.pyx:32-58`) mean a torch implementation
  needs either a padded-dense representation of predecessors (feasible at ~11k-15k
  states with ~1.5-2 edges/state -- small enough to densify per bar-length HMM) or a
  `torch_scatter`-style segment-reduce; `torch.logsumexp` is the right primitive for the
  per-segment reduction once the segments are laid out.
- **(iii) Expose per-frame log-densities, decode stays hard (pragmatic baseline).**
  Simply give differentiable-mode users the observation log-densities/activations (the
  BLSTM/BGRU output that normally feeds into `viterbi()`) and let them apply their own
  loss *before* decoding -- e.g. training a beat-activation network end-to-end against
  ground-truth beat frames, never touching the DBN's internals. This needs zero new
  decoding math, is immediately available once the spectrogram chain differentiability
  (B.1) exists, and is honestly the highest-value-per-effort tier.

**Recommended tiering:**
- **Phase 1 (ships now, in scope):** torch spectrogram chain, end-to-end
  differentiable (B.1), feeding into whatever downstream activation model the user
  supplies -- this alone unlocks "train a model through the spectrogram chain."
- **Phase 2+ (realistic, moderate effort):** expose per-frame log-densities/activations
  (tier iii) as the documented, supported way to get a "soft beat-tracking loss" without
  touching decode internals; ship `forward()` as `madmom_infer.torch.diff.hmm.forward()`
  (tier i) as a differentiable marginal-probability proxy.
- **Phase 3+ (research territory, high effort/uncertain payoff):** `soft_viterbi()`
  (tier ii), the true entropy-regularized relaxation of the actual decode. This is a
  genuine research contribution, not a mechanical port, and should be scoped and staffed
  as such rather than folded into the phase-1 timeline.

**Prior art worth studying before building tier (ii):** `torchaudio.functional.
forced_align` (present in the installed `torchaudio==2.7.1+cu126` in
`all-in-one-fix/.venv`) is a GPU-batched Viterbi-style forced-alignment kernel for CTC
emissions (`log_probs` `(B,T,C)` + a target label sequence -> best alignment path). Its
transition structure is constrained by the target sequence (not a free HMM transition
matrix like madmom's), so it isn't directly reusable, but it demonstrates the shape of a
real, shipped, GPU-batched Viterbi decoder in the torch ecosystem, and its
CPU/CUDA implementation is a genuine compiled custom op (TorchScript + C++/CUDA kernel),
**not** a naive per-frame Python/torch loop. That's a direct tension with mission 1
("zero compilation"): a bespoke fused CUDA Viterbi kernel would give real GPU speedups
but would reintroduce exactly the kind of compiled unit madmom-infer exists to remove.
We recommend madmom-infer's torch-exact `viterbi()` stay a pure-torch scan (accepting
that GPU gains are marginal, as the scaffold's README/CLAUDE.md already say) rather than
chase `forced_align`-style kernel performance -- and note this trade-off explicitly so a
future contributor doesn't "fix" it by adding a compiled extension.

### B.3 Making the exact/differentiable distinction impossible to miss

Given the sibling project's hard rule ("accuracy cannot drop") and this project's
parallel rule (numpy backend bit-identical to madmom via golden fixtures; torch-exact
backend bit-identical in inference mode), the API must make it structurally hard to
accidentally call a soft relaxation where an exact decode was expected. Three
reinforcing signals, all applied together (not either/or):

1. **Separate module path**: exact code lives in `madmom_infer.ml.hmm` (numpy) and
   `madmom_infer.torch.ml.hmm` (torch-exact); soft/relaxed code lives only in
   `madmom_infer.torch.diff.hmm`. A soft decoder is never importable from a path that
   also contains an exact one.
2. **Separate function names**: `viterbi()` (hard, exact, in `ml.hmm` and
   `torch.ml.hmm`) vs. `soft_viterbi()` (in `torch.diff.hmm` only). No shared name is
   ever overloaded with an implicit "soft" flag that silently changes semantics.
3. **Explicit, loud docstrings** on every `torch.diff.*` symbol stating in the first
   sentence that outputs are an approximation for gradient-based training and are *not*
   verified against golden fixtures the way the exact backends are -- golden-fixture
   testing (per this repo's `CLAUDE.md`) applies only to `madmom_infer.*` and
   `madmom_infer.torch.*` (exact), never to `madmom_infer.torch.diff.*`.

### B.4 Phase-2/3 pieces, one line each

- **Comb-filter tempo estimation** (`madmom/audio/comb_filters.pyx`) -- feed-forward
  comb (`y[tau:] += alpha * signal[:-tau]`, `comb_filters.pyx:53-57`) is a pure
  vectorized FIR op, trivially differentiable. Feed-backward comb
  (`for n in range(tau, len(signal)): y[n] += alpha * y[n - tau]`,
  `comb_filters.pyx:113-118,134-138`) is a genuine IIR linear recurrence -- smooth and
  differentiable in principle via a sequential scan, but inherently sequential (not
  parallelizable across `n`), similar in spirit to the Viterbi frame loop. No hard
  argmax anywhere in this file.
- **CRF-based beat tracking** (`madmom/features/beats_crf.pyx`) -- *is* a hard-decoding
  DP, structurally identical to `ml/hmm.pyx`'s Viterbi: `viterbi()`
  (`beats_crf.pyx:143-242`) runs a `max` over candidate predecessors with an explicit
  integer backpointer array (`if new_prob > v_c[i]: v_c[i] = new_prob; bps[k, i] = i - j`,
  `beats_crf.pyx:203-221`) and backtracks through `bps` (`beats_crf.pyx:225-239`).
  Non-differentiable as written, for the same reason as B.2 -- would need the same
  soft-max/entropy-regularization treatment (tier ii) to get gradients through it.
- **NN runtime** (`madmom/ml/nn/*`) -- forward-pass-only numpy math: `FeedForwardLayer.
  activate` is `out = np.dot(data, weights) + bias; activation_fn(out, out=out)`
  (`madmom/ml/nn/layers.py:210-229`), LSTM/GRU gates are the same matmul+bias+elementwise
  -nonlinearity shape (`layers.py:402-430,538,741`). A straightforward torch port
  (`torch.matmul`/`@` + `torch.tanh`/`sigmoid`/`relu`) is automatically differentiable --
  no argmax or hard control flow anywhere in this module. Grep confirms zero
  `backward`/`train`/`fit`/`grad`/`optimizer`/`loss` symbols in the whole `ml/nn/`
  package -- it is exactly the "forward-pass-only NN runtime" the prior research
  described. This is phase-2 scope (gated on the pretrained-weights question per this
  repo's README) but differentiability itself is not the hard part there.
- **Correction to a prior-research detail**: the task brief mentioned "one compiled
  `layers.pyx` hot path" for the NN runtime. `find madmom-upstream -name "*.pyx"`
  returns exactly three files repo-wide: `madmom/ml/hmm.pyx`,
  `madmom/features/beats_crf.pyx`, `madmom/audio/comb_filters.pyx` -- there is no
  `ml/nn/layers.pyx`; `ml/nn/layers.py` is plain Python/NumPy. All three actual compiled
  units are decoding/recurrence hot loops (two Viterbi-style decoders + one IIR comb
  filter), not NN layers. Worth noting since it slightly changes the phase-3 "odds and
  ends" framing: there is no NN-layer Cython code to port at all, only the three
  decoder/filter units already named in this repo's README roadmap.

---

## C. Modernization specifics

### C.1 numpy 2.x gotchas actually present in madmom's code

- **`np.array(..., copy=False)` semantics changed in numpy 2.0** -- pre-2.0, `copy=False`
  meant "avoid a copy if possible, but silently copy anyway if one is required"; numpy
  2.0 changed this to "never copy; raise `ValueError` if a copy would be required" (the
  old behavior is now spelled `copy=None`). madmom's own code uses the *old* meaning in
  three places that would misbehave (raise, where they used to silently succeed) on
  numpy >= 2.0:
  - `madmom/features/beats_hmm.py:564`: `np.array(observations, copy=False, subok=True, ndmin=1)`
  - `madmom/ml/nn/__init__.py:114`: `np.array(data, subok=True, copy=False, ndmin=2)`
  - `madmom/features/downbeats.py:879`: `np.array(features.T, copy=False, ndmin=2).T`
  This is a real, non-cosmetic numpy-2.x incompatibility in madmom's current code (not
  just stale docstrings) -- madmom-infer's port must use `copy=None` (or restructure to
  avoid needing the distinction) everywhere this pattern would otherwise be copied.
- **`from distutils.extension import Extension`** (`madmom-upstream/setup.py:10`) --
  `distutils` was removed from the standard library in Python 3.12. madmom's own build
  script, as it exists in this clone, cannot run on stock Python 3.12+ without relying on
  setuptools' vendored/shimmed distutils (behavior has shifted across setuptools
  versions). This is a concrete, current-code citation for "why is madmom hard to install
  on modern Python" beyond the general Cython-compilation argument.
- **Build-vs-runtime numpy pin mismatch in the upstream repo itself**: its
  `pyproject.toml` build-system requires `numpy>2` (`pyproject.toml:6`, i.e. building the
  Cython extensions today needs numpy 2.x's C-API headers), while `setup.py`'s runtime
  `install_requires` pins `numpy>=1.13.4` (`setup.py:72`). This gap (build against numpy
  2.x headers, but claim to run against numpy as old as 1.13) is itself evidence for why
  "stop compiling it" (mission 1) is the actual fix, rather than chasing a tighter pin
  range on the original package.
- **`np.float`/`np.int`/`np.bool`/`np.object`/`np.complex` bare aliases** (removed in
  numpy 1.24+/2.0): **zero occurrences** as real code anywhere in the package (grep
  across all `.py`/`.pyx` files). The only two hits are docstring prose in
  `madmom/ml/hmm.pyx:278,301` ("should be np.float"), not executable code -- not a real
  numpy-2.x blocker, contrary to what a naive grep-and-panic might suggest.
- **`scipy.fftpack`** (legacy namespace, still shipped but superseded by `scipy.fft` /
  `numpy.fft`): used in `madmom/audio/stft.py:15,131` (the actual per-frame FFT call
  driving every STFT in the codebase) and `madmom/audio/cepstrogram.py:13`
  (`from scipy.fftpack import dct`). Not broken today, but worth modernizing to
  `numpy.fft.rfft` (real FFT, matches STFT's real-valued frames, and is what a torch
  `torch.fft.rfft`/`torch.stft` port should be checked against numerically) rather than
  perpetuating the legacy alias.
- **`scipy.io.wavfile.read(filename, mmap=True)`** (`madmom/io/audio.py:643`) returns a
  **read-only** memory-mapped array. madmom's own gain/normalize functions already avoid
  in-place mutation of that buffer (they reconstruct via `np.asanyarray(signal/scaling,
  dtype=signal.dtype)`, e.g. `signal.py:167`), so this isn't currently broken, but it's a
  sharp edge worth a defensive test in madmom-infer (any future code path that tries true
  in-place mutation of a freshly-loaded `Signal` will hit a read-only-buffer error, and
  scipy has also tightened `wavfile.write`'s behavior around non-native dtypes across
  versions).
- **`np.fromstring(data, 'float32')`** (`madmom/audio/signal.py:1472`, in the `Stream`
  class for parsing binary audio chunks) -- long-deprecated in favor of
  `np.frombuffer` for binary (non-text) data; still works today but is a one-line fix
  worth making during the port rather than carrying forward.

### C.2 `Signal` (and `Filterbank`/`Filter`/`ShortTimeFourierTransform`): subclass vs. composition

madmom's `Signal` is a real `np.ndarray` subclass (`madmom/audio/signal.py:506`,
`class Signal(np.ndarray):`), with `__new__` doing the actual construction/loading
(`signal.py:600-632`), `__array_finalize__` propagating `sample_rate`/`start`/`stop`
across numpy-internal views (`signal.py:634-640`), and a custom `__reduce__`/
`__setstate__` pair purely to keep those extra attributes alive across pickling
(`signal.py:642-656`). `Filterbank` and `Filter` (`madmom/audio/filters.py:413,692`) and
`ShortTimeFourierTransform` (`madmom/audio/stft.py:196`) follow the identical pattern.

**Recommendation: composition, not subclassing**, for `madmom_infer`'s numpy-reference
classes. Reasoning:

- ndarray subclassing is exactly the kind of "sharp edges, most use cases better served
  by composition" pattern numpy's own documentation has long steered people away from --
  and madmom's own code shows the cost directly: every subclass needs matching
  `__array_finalize__`/pickle-support boilerplate (`signal.py:634-656`) just to keep
  metadata attached, and there's an explicit historical scar in `signal_frame()`'s
  comments (`signal.py:917-919`) about a since-reverted workaround for a numpy memory
  leak (issue #321) that the subclassing-adjacent indexing pattern triggered. That's
  exactly the kind of "numpy-internals-dependent fragility" mission 1 (maintainable)
  should be spending zero future maintenance budget on.
- **The decisive argument for madmom-infer specifically**: torch tensors *cannot*
  subclass `np.ndarray`. If the numpy reference's `Signal` is `np.ndarray`-derived,
  there is no structurally-parallel way to build a torch twin (`torch.Tensor` has its
  own, differently-shaped subclassing story via `__torch_function__`, with its own
  fragility). Composition -- a thin wrapper class holding a plain backing array/tensor
  plus explicit metadata attributes (`sample_rate`, `num_channels`, etc.) -- is the one
  shape that lets `madmom_infer.audio.signal.Signal` (numpy-backed) and
  `madmom_infer.torch.audio.signal.Signal` (torch-backed) share an API and a mental
  model, which matters more here than in single-backend madmom.
- Composition does not have to give up drop-in ergonomics: implement `__array__(self,
  dtype=None)` (so `np.asarray(sig)`, `np.dot(sig, x)`, etc. keep working transparently)
  and forward the handful of attribute/indexing accesses the actual phase-1 call sites
  need (`all-in-one-infer` only ever constructs `Signal(path_or_array, sample_rate=...,
  num_channels=...)`, slices it via `FramedSignal`, and reads `.dtype`/`.sample_rate` --
  see `all-in-one-fix/src/allin1_infer/spectrogram.py:19-22,90-104`). This covers every
  real phase-1 usage while sidestepping `__array_finalize__` entirely.
- **Honest caveat**: this is one intentional, documented API deviation from strict
  madmom parity -- code that does `isinstance(sig, np.ndarray)` or hands a `Signal`
  directly to arbitrary third-party numpy-consuming code expecting true ndarray identity
  would need `np.asarray(sig)` first. Given the project's own stated priority order
  (modernize/maintain over pixel-parity API mimicry), this is the right trade, but it
  should be called out plainly in migration docs, not discovered by surprise.

### C.3 Python version floor, packaging, CI

- **Python floor**: the scaffold's `pyproject.toml` currently sets `requires-python =
  ">=3.9"`; mission 1's own stated target is "3.10-3.13+", and Python 3.9 has already
  reached end-of-life (per its published schedule) as of the current date. Recommend
  bumping the floor to `>=3.10` to match the mission statement and drop a dead-EOL
  target from the support matrix (tracked as a phase-1 task, see D).
- **Packaging**: hatchling is already in place and is the right choice -- pure Python,
  no build backend complexity needed since there is (by design, mission 1) nothing to
  compile.
- **CI matrix**: `{3.10, 3.11, 3.12, 3.13} x numpy 2.x` as the primary matrix (this is a
  from-scratch project targeting current numpy, not one straddling 1.x/2.x compatibility
  the way upstream madmom's stale pin range does); a separate, optional job for the
  `torch` extra (matching whatever Python versions the pinned `torch>=2.0.0` supports on
  each Python version). OS matrix: since the whole point is "zero compilation", a
  cross-platform (Linux/macOS/Windows) pure-Python-wheel smoke test is cheap and is
  itself a selling point over original madmom's platform-specific compiled wheels --
  worth including even though there's no per-OS build variance to catch.

### C.4 Golden-fixture generation recipe

The concrete, already-available recipe: `/home/worzpro/Desktop/dev/openmirlab/
all-in-one-fix/.venv` has a **working madmom 0.17.dev0 install today** (confirmed:
Python 3.10.18, numpy 1.23.5, scipy 1.15.3, installed via `pip install git+https://
github.com/CPJKU/madmom` per `all-in-one-fix/src/allin1_infer/_install_madmom.py:17-21`).
No new environment needs to be built to start generating fixtures.

Proposed process:

1. Write a **standalone** fixture-generation script (no dependency on `madmom_infer`
   itself, only on real `madmom`) that runs each phase-1 target's exact call pattern --
   `Signal` -> `FramedSignalProcessor` -> `ShortTimeFourierTransformProcessor` ->
   `FilteredSpectrogramProcessor`/`LogarithmicSpectrogramProcessor` with all-in-one-infer's
   exact parameters (`frame_size=2048, fps=100, num_bands=12, fmin=30, fmax=17000,
   norm_filters=True, mul=1, add=1`, per `all-in-one-fix/src/allin1_infer/spectrogram.
   py:27-40`), plus `DBNDownBeatTrackingProcessor` with its exact parameters
   (`beats_per_bar=[3,4], min_bpm=55, max_bpm=215, num_tempi=60, transition_lambda=100,
   observation_lambda=16, fps=100`) fed a representative beat-activation array -- and
   dumps input/output pairs as small `.npz` files.
2. Run it once against the existing `all-in-one-fix/.venv` interpreter (fast path,
   available now) to bootstrap fixtures; separately document the fully-reproducible
   from-scratch recipe for anyone without that checkout: `uv venv --python 3.10
   fixture-venv && uv pip install --python fixture-venv numpy==1.23.5 scipy==1.15.3
   cython 'git+https://github.com/CPJKU/madmom'` (pinned to the exact versions already
   proven to work, rather than guessing at a range).
3. Keep fixture inputs short (1-2 second clips) so `.npz` fixtures are small enough to
   commit directly to the repo (no LFS needed) -- reuse madmom-upstream's own bundled
   test audio under `madmom-upstream/tests/data/audio/*.wav` if license-clean and small,
   or a short synthetic/openly-licensed clip otherwise.
4. Commit the generation script itself (e.g. `tools/generate_golden_fixtures.py`) so
   fixtures are regenerable and auditable, with a short `tests/fixtures/README.md`
   documenting exact provenance (madmom commit/version, numpy/scipy versions, generation
   command).
5. `madmom_infer`'s test suite loads these fixtures and asserts against the port's
   output: `np.testing.assert_allclose(..., atol=0)` (or the smallest tolerance that
   survives float64 arithmetic reordering) for the spectrogram chain, and **exact integer
   equality** on decoded state/beat-position sequences for the Viterbi/DBN path (a
   discrete decode either matches madmom's path or it doesn't -- there is no meaningful
   "close enough" for a state index).

---

## D. Phase-1 work breakdown

Ordered by dependency; tasks in the same bullet group have no code dependency on each
other and can run in parallel (separate engineers/agents).

1. **Golden-fixture generation harness** (script + first fixtures committed) -- no
   dependency on any new madmom_infer code, only needs the existing madmom env (C.4).
   *Effort: ~0.5-1 day.* **Parallelizable with #2.**
2. **`processors.py`: `Processor`/`SequentialProcessor` base classes** -- pure
   infrastructure, no DSP math (`madmom/processors.py:30-114,288-410`).
   *Effort: ~0.5 day.* **Parallelizable with #1.**
3. **`audio/signal.py`: `Signal` (composition-based, C.2) + `FramedSignalProcessor`** --
   the module with the most subtle semantics to get exactly right (origin/padding math,
   B.1's int16-vs-window-scaling convention, `end='normal'`/`'extend'` frame-count
   formulas). *Effort: ~1-2 days.* Can start once #2's `Processor` base exists (or in
   parallel, wiring the `Processor` subclass in last).
4. **`audio/filters.py`: filterbank construction** -- independent, pure-math module (no
   dependency on `Signal`/`FramedSignal`). *Effort: ~1 day.* **Parallelizable with #3.**
5. **`audio/stft.py`: `ShortTimeFourierTransformProcessor`** -- depends on #3's
   `FramedSignal`. *Effort: ~1 day.*
6. **`audio/spectrogram.py`: `FilteredSpectrogramProcessor` + `LogarithmicSpectrogramProcessor`**
   -- depends on #4 and #5. *Effort: ~0.5-1 day.*
7. **Golden-fixture tests wired for the full spectrogram chain** (depends on #1, #3, #5,
   #6) -- verify bit-identical output against all-in-one-infer's exact parameter set.
   *Effort: ~0.5 day.*
8. **`ml/hmm.py`: numpy Viterbi port** (CSR-sparse `TransitionModel`, log-domain
   per-frame segment-max recursion, backtracking) -- the highest-risk, highest-effort
   task, and shares no code with #3-#7, so it can run as its **own fully parallel
   workstream** starting immediately alongside #3. *Effort: ~3-5 days (the phase-1
   centerpiece).*
9. **`features/beats_hmm.py`: bar-length state-space/transition-model construction** --
   depends on #8's `TransitionModel` interface. *Effort: ~2 days.*
10. **`features/downbeats.py`: `DBNDownBeatTrackingProcessor`** -- wires #8+#9 together
    with all-in-one-infer's exact parameters (`beats_per_bar=[3,4], min_bpm=55,
    max_bpm=215, num_tempi=60, transition_lambda=100, observation_lambda=16, fps=100`).
    *Effort: ~1-2 days.*
11. **Golden-fixture tests for the full downbeat pipeline** (activation array -> beat/
    downbeat positions; exact integer path match) -- depends on #1 and #10.
    *Effort: ~1 day.*
12. **End-to-end acceptance test**: swap all-in-one-infer's `madmom` dependency for
    `madmom_infer` (on a branch), run `analyze()` on a small reference track, diff
    against the madmom-based baseline output. This *is* the phase-1 "done" definition.
    *Effort: ~0.5-1 day, depends on everything above.*

**Rough total**: ~2.5-3 weeks single-engineer serial; ~1.5-2 weeks with two parallel
workstreams (spectrogram chain: #1-7; HMM/DBN chain: #8-11), converging at #12.

**First three tasks** (the ones to start immediately): #1 (golden-fixture harness) and
#2 (`Processor`/`SequentialProcessor` base) can begin in parallel today with zero
blocking dependencies; #3 (`Signal`/`FramedSignalProcessor`) is the natural third task,
starting as soon as #2 lands (or in parallel, deferring the `Processor` wiring).
