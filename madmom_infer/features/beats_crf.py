"""Numpy reimplementation of madmom's Cython CRF beat-tracking Viterbi
decoder (`features/beats_crf.pyx`) -- backs `features/beats.py`'s
`CRFBeatDetectionProcessor`. Wave 4f of the complete-port campaign; see
CLAUDE.md's audit table, `features/beats.py` row (`CRFBeatDetectionProcessor`)
and the module docstring below for the specific porting approach.

**Same playbook as Phase 1's `ml/hmm.py` (hmm.pyx -> numpy)**: keep the
genuinely sequential recursion (here, the "one beat variable at a time" loop
over `range(num_x - 1)`, `num_x` = number of modeled beats -- each depends
on the previous beat's Viterbi variables) as a Python loop, but vectorize
the per-iteration double loop over *states* and *look-back offsets* with
numpy, replacing `beats_crf.pyx`'s `for i in range(num_st): for j in
range(min(i, num_tr)):` nested loop.

`initial_distribution`, `transition_distribution`, `normalisation_factors`,
and `best_sequence` are NOT Cython-typed in the original `.pyx` file (only
`viterbi` is `@cython.cdivision`/`boundscheck`/`wraparound`-decorated with
typed memoryview arguments) -- they are already plain numpy/scipy Python
functions, so those four are verbatim ports, unchanged line-for-line except
import paths. Only `viterbi` needed real translation work.

**Bit-identity finding, the crux of this port -- got wrong twice before
being verified right, both mistakes caught by fuzzing against real madmom
directly rather than trusting a reading of the `.pyx` source.**
`viterbi.pyx`'s inner accumulator `new_prob`/`path_prob` are declared `cdef
double` (`beats_crf.pyx:191`) even though `v_p`/`transition`/`v_c` are all
typed `float` (32-bit) memoryviews. The naive reading -- "the addition
happens in double precision because it's stored into a double variable" --
is WRONG: C's (and Cython's) usual arithmetic conversions determine an
expression's precision from its OPERANDS, not its assignment target, so
`v_p[i - j] + transition[j]` (both `float` operands) is computed in
**float32** precision and only WIDENED (exactly, no further rounding) to
double when stored into `new_prob`; the `double` declaration changes
nothing about the arithmetic itself. A first attempt at this port assumed
double-precision addition, matching the winning *index* `j` (hence decoded
beat POSITIONS) in every one of 120 randomized fuzz trials against real
madmom (reference venv) but leaving the scalar `log_prob` off by 1-2 float32
ULP in roughly a third of them. Switching the candidate arithmetic to
float32 (matching the corrected reading above) did NOT fully fix it --
`log_prob` still mismatched ~35% of trials -- because of a SECOND,
independent mistake: `beats_crf.pyx:221`'s `v_c[i] += activations[i] +
norm_factor[i]` is a compound-assignment, which groups as `v_c[i] = v_c[i]
+ (activations[i] + norm_factor[i])` -- `activations[i] + norm_factor[i]`
is evaluated as ONE float32 addition first, THEN added to `v_c[i]` -- NOT
`(v_c[i] + activations[i]) + norm_factor[i]`, the left-to-right chaining a
naive three-term `a + b + c` translation produces. Floating-point addition
is not associative, so these two groupings can (and empirically did)
produce different last-bit results. Fixing BOTH -- pure float32 arithmetic
throughout (no float64 intermediate anywhere; the `double` declarations in
the original are inert for the actual computed values, only widening
already-rounded results) AND the exact `v_c[i] + (activations[i] +
norm_factor[i])` grouping -- reproduces real madmom's `log_prob` bit-for-bit
in 200/200 randomized fuzz trials against the reference venv, not just the
decoded path.

Given pure float32 precision is the correct model, `viterbi()` below
computes the per-state max as a single, clean, BATCHED `np.argmax` over all
`num_tr` candidates at once (fill a `(num_tr, num_st)` array, one
`np.argmax(..., axis=0)`) rather than replaying the Cython loop's literal
"running max, re-compared after every `j`" procedure. This is NOT an
approximation the way the double-precision-addition assumption was: `max`
has no rounding/associativity concerns the way repeated addition does --
selecting the largest of `N` already-float32-valued candidates gives an
identical result whether they're compared all-at-once or one-at-a-time
against a running max, since no arithmetic (hence no intermediate rounding)
happens during the comparison itself. `np.argmax`'s first-occurrence
tie-break also matches the Cython loop's strict `>` comparison (a later `j`
with an equal candidate value never replaces an earlier one).

The `j < i` (strictly-less-than, not `<=`) look-back boundary in
`beats_crf.pyx:205`'s `for j in range(min(i, num_tr))` means state 0 can
never be a valid Viterbi-transition destination beyond the very first beat
(there is no `j` for which `i - j == i` and `j < i` simultaneously except
`i == j == 0`, excluded since `range(0)` is empty) -- `viterbi()` below
encodes this exactly via its `candidates[j, j + 1:]` fill (see inline
comments), not by special-casing state 0.

`-inf - (-inf)` (an unreachable state's `v_p` minus its own -inf-valued
`norm_factor`, e.g. a padding/edge frame near the correlation boundary) is
NaN, not -inf -- unlike the Cython loop's `if v_p[i] > path_prob:` (a NaN
comparison is always False, so a NaN candidate can never win against the
initial `path_prob = -inf`, i.e. it behaves like -inf for selection
purposes), `np.argmax` does NOT ignore NaN -- it can make a NaN "stick" as
the running max once encountered. `viterbi()` replaces NaN with -inf before
the final `np.argmax`, restoring the "NaN/unreachable states never win"
semantics explicitly -- the same precedent as `ml/hmm.py`'s own
NaN-normalization comment in its `viterbi()`.

Reads: numpy, scipy.stats.norm (transition_distribution only), scipy.ndimage
.correlate1d (normalisation_factors only); read by: madmom_infer/features/
beats.py (CRFBeatDetectionProcessor).
"""

import numpy as np
from scipy.ndimage import correlate1d


def initial_distribution(num_states, interval):
    """Un-normalized initial distribution: uniform over the first `interval`
    states, zero elsewhere.

    Verbatim port of `madmom.features.beats_crf.initial_distribution`
    (`beats_crf.pyx:29-51`). Deliberately left un-normalized (see upstream's
    own comment, preserved below) so the position of the first beat doesn't
    bias the probability of the beat sequence -- normalizing would favor
    shorter intervals.
    """
    # We leave the initial distribution un-normalised because we want the
    # position of the first beat not to influence the probability of the
    # beat sequence. Normalising would favour shorter intervals.
    init_dist = np.ones(num_states, dtype=np.float32)
    init_dist[interval:] = 0
    return init_dist


def transition_distribution(interval, interval_sigma):
    """Log-normal-shaped transition distribution over move distances
    `0..2*interval-1` [frames], centered on `interval` with spread
    `interval_sigma` (in log2 space).

    Verbatim port of `madmom.features.beats_crf.transition_distribution`
    (`beats_crf.pyx:54-81`).
    """
    from scipy.stats import norm

    move_range = np.arange(interval * 2, dtype=float)
    # to avoid floating point hell due to np.log2(0)
    move_range[0] = 0.000001

    trans_dist = norm.pdf(
        np.log2(move_range), loc=np.log2(interval), scale=interval_sigma
    )
    trans_dist /= trans_dist.sum()
    return trans_dist.astype(np.float32)


def normalisation_factors(activations, transition_distribution):
    """Per-frame normalization factors: `activations` correlated with
    `transition_distribution`, centered so the correlation looks FORWARD
    from each frame (`origin=-len(transition_distribution)//2`).

    Verbatim port of `madmom.features.beats_crf.normalisation_factors`
    (`beats_crf.pyx:84-103`).
    """
    return correlate1d(
        activations,
        transition_distribution,
        mode="constant",
        cval=0,
        origin=-int(transition_distribution.shape[0] / 2),
    )


def best_sequence(activations, interval, interval_sigma):
    """Extract the best beat sequence for a piece with the Viterbi
    algorithm, given a fixed dominant `interval`.

    Verbatim port of `madmom.features.beats_crf.best_sequence`
    (`beats_crf.pyx:106-140`).

    Parameters
    ----------
    activations : numpy array
        Beat activation function of the piece.
    interval : int
        Beat interval of the piece.
    interval_sigma : float
        Allowed deviation from the interval per beat.

    Returns
    -------
    beat_pos : numpy array
        Extracted beat positions [frame indices].
    log_prob : float
        Log probability of the beat sequence.
    """
    init = initial_distribution(activations.shape[0], interval)
    trans = transition_distribution(interval, interval_sigma)
    norm_fact = normalisation_factors(activations, trans)

    # ignore division by zero warnings when taking the logarithm of 0.0,
    # the result -inf is fine anyways!
    with np.errstate(divide="ignore"):
        init = np.log(init)
        trans = np.log(trans)
        norm_fact = np.log(norm_fact)
        log_act = np.log(activations)

    return viterbi(init, trans, norm_fact, log_act, interval)


def viterbi(pi, transition, norm_factor, activations, tau):
    """Viterbi algorithm to compute the most likely beat sequence from the
    given (log-domain) activations and the dominant interval `tau`.

    Numpy translation of `madmom.features.beats_crf.viterbi`
    (`beats_crf.pyx:146-242`, the typed-Cython inner loop) -- see this
    module's header for the precision/grouping findings behind this
    implementation.

    Parameters
    ----------
    pi : numpy array
        Initial distribution (log domain).
    transition : numpy array
        Transition distribution (log domain).
    norm_factor : numpy array
        Normalisation factors (log domain).
    activations : numpy array
        Beat activations (log domain).
    tau : int
        Dominant interval [frames].

    Returns
    -------
    beat_pos : numpy array
        Extracted beat positions [frame indices].
    log_prob : float
        Log probability of the beat sequence.
    """
    tau = int(tau)
    # everything below is pure float32 arithmetic, matching the Cython
    # memoryviews' actual computed precision -- see this module's header.
    pi = np.ascontiguousarray(pi, dtype=np.float32)
    transition = np.ascontiguousarray(transition, dtype=np.float32)
    norm_factor = np.ascontiguousarray(norm_factor, dtype=np.float32)
    activations = np.ascontiguousarray(activations, dtype=np.float32)

    # number of states
    num_st = activations.shape[0]
    # number of transitions
    num_tr = transition.shape[0]
    # number of beat variables
    num_x = num_st // tau

    # back-tracking pointers, one row per beat (matches beats_crf.pyx's
    # `bps = np.empty((num_x - 1, num_st), dtype=int)` -- a negative first
    # dimension here (num_x == 0) raises ValueError, same failure class as
    # upstream's own np.empty call with a negative shape).
    bps = np.zeros((num_x - 1, num_st), dtype=np.int64)

    # init first beat: `v_p[i] = pi[i] + activations[i] + norm_factor[i]`
    # (beats_crf.pyx:194-195), left-to-right float32 addition.
    v_p = pi + activations + norm_factor

    idx_i = np.arange(num_st)
    for k in range(num_x - 1):
        # candidate[j, i] = v_p[i - j] + transition[j], valid only when
        # i - j >= 1 (i.e. i > j) -- exactly the `for j in range(min(i,
        # num_tr))` boundary of beats_crf.pyx:203-205, see this module's
        # header for the derivation. -inf everywhere else.
        candidates = np.full((num_tr, num_st), -np.inf, dtype=np.float32)
        for j in range(num_tr):
            if j + 1 >= num_st:
                break
            # candidates[j, i] for i in [j + 1, num_st) = v_p[i - j] +
            # transition[j] = v_p[1 : num_st - j] + transition[j]
            candidates[j, j + 1:] = v_p[1:num_st - j] + transition[j]

        # first-occurrence-of-the-max argmax over j (axis 0) matches the
        # Cython loop's strict `>` tie-break -- see this module's header.
        best_j = np.argmax(candidates, axis=0)
        v_c = candidates[best_j, idx_i]

        bps[k] = idx_i - best_j

        # `v_c[i] += activations[i] + norm_factor[i]` (beats_crf.pyx:221) --
        # compound-assignment grouping: `activations[i] + norm_factor[i]`
        # is ONE float32 addition, THEN added to v_c[i] -- NOT left-to-right
        # `(v_c[i] + activations[i]) + norm_factor[i]`. See this module's
        # header for why this grouping is load-bearing for bit-identity.
        v_c = v_c + (activations + norm_factor)
        v_p = v_c

    # add the final best state to the path: `v_p[i] -= norm_factor[i]`
    # then find the argmax (beats_crf.pyx:226-234) -- see this module's
    # header for the NaN-normalization step below. `-inf - (-inf) = NaN` is
    # an expected, benign outcome for unreachable states (see header) --
    # silence the resulting RuntimeWarning rather than let it leak to
    # callers, same "expected -inf/NaN is fine" precedent as
    # `best_sequence`'s own `np.errstate(divide='ignore')` above.
    with np.errstate(invalid="ignore"):
        v_p_final = v_p - norm_factor
    v_p_final = np.where(np.isnan(v_p_final), -np.inf, v_p_final)
    next_state = int(np.argmax(v_p_final))
    path_prob = float(v_p_final[next_state])

    path = np.empty(num_x, dtype=np.int64)
    path[num_x - 1] = next_state
    # track the path backwards
    for i in range(num_x - 2, -1, -1):
        next_state = int(bps[i, next_state])
        path[i] = next_state

    # return the best sequence and its log probability
    return path, path_prob
