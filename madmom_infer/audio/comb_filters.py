"""Comb-filter / comb-filterbank functionality -- numpy port of
`madmom.audio.comb_filters` (`madmom-upstream/madmom/audio/comb_filters.pyx`,
a Cython module, same porting playbook as Phase 1's `ml/hmm.py` port of
`hmm.pyx`). Wave 4c of the complete-port campaign; feeds
`madmom_infer/features/tempo.py`'s `CombFilterTempoHistogramProcessor`.

`feed_forward_comb_filter` (`y[n] = x[n] + alpha * x[n - tau]`) is a single
vectorized numpy slice-add in real madmom too (its `def` is plain, untyped
Python, no Cython `cdef` machinery) -- ported as-is, `alpha` used at full
float64 precision.

`feed_backward_comb_filter` (`y[n] = x[n] + alpha * y[n - tau]`) is a
genuinely SEQUENTIAL IIR recursion (each output sample depends on an
earlier OUTPUT sample, not just earlier inputs) -- real madmom implements it
with an explicit Cython `for` loop, not a vectorized formula, and
`scipy.signal.lfilter` cannot reproduce it directly (`lfilter`'s recursion
has a fixed unit lag; this filter's feedback lag is `tau` samples, and for
`tau > 1` that is not expressible as a single-`lfilter` all-pole IIR without
interleaving `tau` independent lag-1 sub-sequences -- doable, but yields a
DIFFERENT floating-point summation order than the direct loop, which this
project's bit-identity mandate requires; an explicit Python loop, iterating
in the exact same order as the Cython one, is therefore the only construction
proven bit-identical here). Ported as a scalar Python loop over the exact
same `y[n] += alpha * y[n - tau]` update, for both the 1D and 2D cases
(`_feed_backward_comb_filter_1d`/`_2d`, matching upstream's own split).

**Found and reproduced a real upstream precision quirk, confirmed empirically
against the reference venv (not guessed):** `_feed_backward_comb_filter_1d`/
`_2d`'s Cython signatures declare `alpha` as a C `float` (32-bit) parameter
-- `cdef ... float alpha` -- so any Python `float` (64-bit) passed in gets
silently ROUNDED TO float32 precision by Cython's own argument coercion
BEFORE the accumulation loop runs, even though the recursion itself operates
on a float64 array (`signal.astype(float)`). `feed_forward_comb_filter`
has NO such truncation (its `def` is untyped, no Cython helper). Verified
directly: `feed_backward_comb_filter(x, tau=3, alpha=0.1)` against the
reference venv's real madmom matches a Python port that does
`float(np.float32(alpha))` before the loop bit-for-bit
(`np.array_equal`, both 1D and 2D), and does NOT match a same-loop variant
that keeps `alpha` at full float64 precision (differs by ~1.7e-9 after 50
samples) -- this is a small but real quirk that would otherwise silently
break bit-identity for any `alpha` not exactly representable in float32
(which is most of them, e.g. the module's own default `ALPHA = 0.79` in
`features/tempo.py`).

Reads: numpy, madmom_infer.processors (Processor); read by:
madmom_infer/features/tempo.py (CombFilterTempoHistogramProcessor's
interval_histogram_comb).
"""

import numpy as np

from ..processors import Processor


def feed_forward_comb_filter(signal, tau, alpha):
    """Filter `signal` with a feed forward comb filter:
    `y[n] = x[n] + alpha * x[n - tau]`.

    Verbatim port of `madmom.audio.comb_filters.feed_forward_comb_filter`
    (`comb_filters.pyx:19-57`). `alpha` is used at full float64 precision
    (no Cython type coercion applies to this function, see module header).
    """
    if tau <= 0:
        raise ValueError("`tau` must be greater than 0")
    y = signal.astype(float)
    y[tau:] += alpha * signal[:-tau]
    return y


def _feed_backward_comb_filter_1d(signal, tau, alpha):
    """Feed backward comb filter for 1D signals.

    Port of `madmom.audio.comb_filters._feed_backward_comb_filter_1d`
    (`comb_filters.pyx:103-120`). `alpha` is rounded to float32 precision
    before the loop, reproducing the Cython `cdef ... float alpha` parameter
    coercion real madmom performs here -- see module header.
    """
    if tau <= 0:
        raise ValueError("`tau` must be greater than 0")
    y = signal.copy()
    tau = int(tau)
    alpha = float(np.float32(alpha))
    for n in range(tau, len(signal)):
        y[n] += alpha * y[n - tau]
    return y


def _feed_backward_comb_filter_2d(signal, tau, alpha):
    """Feed backward comb filter for 2D signals.

    Port of `madmom.audio.comb_filters._feed_backward_comb_filter_2d`
    (`comb_filters.pyx:124-140`). Same float32 `alpha`-rounding quirk as the
    1D variant, see module header.
    """
    if tau <= 0:
        raise ValueError("`tau` must be greater than 0")
    y = signal.copy()
    tau = int(tau)
    alpha = float(np.float32(alpha))
    for d in range(2):
        for n in range(tau, len(signal)):
            y[n, d] += alpha * y[n - tau, d]
    return y


def feed_backward_comb_filter(signal, tau, alpha):
    """Filter `signal` with a feed backward comb filter:
    `y[n] = x[n] + alpha * y[n - tau]`.

    Port of `madmom.audio.comb_filters.feed_backward_comb_filter`
    (`comb_filters.pyx:61-99`); dispatches to the 1D/2D scalar-loop helpers
    above (both reproduce the real float32-`alpha`-rounding quirk).
    """
    if signal.ndim == 1:
        return _feed_backward_comb_filter_1d(signal.astype(float), tau, alpha)
    elif signal.ndim == 2:
        return _feed_backward_comb_filter_2d(signal.astype(float), tau, alpha)
    else:
        raise ValueError("signal must be 1d or 2d")


def comb_filter(signal, filter_function, tau, alpha):
    """Filter `signal` with a bank of either feed forward or feed backward
    comb filters, one per `(tau, alpha)` pair, stacked along a new last axis.

    Verbatim port of `madmom.audio.comb_filters.comb_filter`
    (`comb_filters.pyx:144-223`).
    """
    tau = np.array(tau, dtype=int, ndmin=1)
    if tau.ndim != 1:
        raise ValueError("`tau` must be a 1D numpy array")
    alpha = np.array(alpha, dtype=float, ndmin=1)
    if len(alpha) == 1:
        alpha = np.repeat(alpha, len(tau))
    if alpha.ndim != 1:
        raise ValueError("`alpha` must be a 1D numpy array")
    if len(tau) != len(alpha):
        raise ValueError("`tau` and `alpha` must have the same length")
    y = []
    for i, t in np.ndenumerate(tau):
        # NOTE: `float(alpha[i])` here (not in upstream, which just does
        # `alpha[i]`, a numpy float64 SCALAR) -- found and fixed a genuine
        # numpy-2.x-vs-1.23.5 divergence, same class as docs/DESIGN.md C.1
        # (see `features/onsets.py`'s `normalized_weighted_phase_deviation`
        # for the precedent). Under numpy >= 2.0's NEP 50 strict scalar-
        # promotion rules, a numpy float64 SCALAR ("strong") multiplied
        # against a float32 array upcasts the whole expression to float64,
        # whereas a plain Python float ("weak" scalar) does not -- but
        # numpy < 2.0 (including the reference venv's 1.23.5, real madmom's
        # own recorded environment) uses value-based casting and treats
        # BOTH the same way (keeps float32, since 0.79-scale alphas fit).
        # `feed_forward_comb_filter`/`feed_backward_comb_filter` called
        # DIRECTLY with a Python-float `alpha` literal already get the
        # "weak scalar" (float32-preserving) behavior on every numpy
        # version; without this cast, going through THIS bank function
        # (which extracts `alpha[i]` from a numpy array) silently diverges
        # from that on numpy >= 2.0 by ~1e-9 relative error -- confirmed
        # empirically (this port's own in-process bank-call output
        # differed from its own direct-call output on the SAME inputs
        # under numpy 2.4.6, but matched exactly under the reference
        # venv's numpy 1.23.5).
        y.append(filter_function(signal, t, float(alpha[i])))
    if signal.ndim == 1:
        return np.vstack(y).T
    elif signal.ndim == 2:
        return np.dstack(y)
    else:
        raise ValueError("only 1D and 2D signals supported")


class CombFilterbankProcessor(Processor):
    """Comb-filterbank processor: a bank of either feed forward or feed
    backward comb filters, one per `(tau, alpha)` pair.

    Port of `madmom.audio.comb_filters.CombFilterbankProcessor`
    (`comb_filters.pyx:226-305`). `filter_function` may be the function
    itself or the string literal `'forward'`/`'backward'`.
    """

    def __init__(self, filter_function, tau, alpha):
        self.tau = np.array(tau, dtype=int, ndmin=1)
        self.alpha = np.array(alpha, dtype=float, ndmin=1)
        if filter_function in ("forward", feed_forward_comb_filter):
            self.filter_function = feed_forward_comb_filter
        elif filter_function in ("backward", feed_backward_comb_filter):
            self.filter_function = feed_backward_comb_filter
        else:
            raise ValueError(
                "unknown `filter_function`: %s" % filter_function)

    def process(self, data, **kwargs):
        """Filter `data` with the configured bank of comb filters."""
        # pylint: disable=arguments-differ, unused-argument
        return comb_filter(data, self.filter_function, self.tau, self.alpha)
