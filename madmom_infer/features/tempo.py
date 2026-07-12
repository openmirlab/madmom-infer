"""Reimplementation of madmom.features.tempo -- tempo histogram construction
(autocorrelation, resonating comb filters, or a DBN beat tracker) plus
`TempoEstimationProcessor`, which smooths a beat activation function, builds
one of those histograms, and picks its peak(s) as the estimated tempo/tempi.
Wave 4c of the complete-port campaign; see CLAUDE.md's audit table,
`features/tempo.py` row (comb variant unblocked by this wave's own
`madmom_infer/audio/comb_filters.py` port).

Ports every function/class the 4.0 audit table's `features/tempo.py` row
lists as TO-PORT(4c): `smooth_histogram`, `interval_histogram_acf`,
`interval_histogram_comb`, `dominant_interval`, `detect_tempo`,
`TempoHistogramProcessor`, `ACFTempoHistogramProcessor`,
`CombFilterTempoHistogramProcessor`, `DBNTempoHistogramProcessor`,
`TempoEstimationProcessor`. `TCNTempoHistogramProcessor` stays EXCLUDED per
the audit table's own note (it only consumes `TCNBeatProcessor` output,
which cannot exist -- `BEATS_TCN` is not shipped by a real madmom install,
see CLAUDE.md's 4.0 audit corrections).

**All of `TempoHistogramProcessor` and its subclasses, plus
`TempoEstimationProcessor`, are OFFLINE-ONLY** -- upstream subclasses
`OnlineProcessor` (`process_offline`/`process_online`, a `BufferProcessor`-
backed streaming continuation path, and online-mode-only constructor
state), but this project's processors are offline/whole-clip only
(`madmom_infer/processors.py`'s module header: `OnlineProcessor`/streaming
machinery is a stated permanent exclusion, same precedent as `features/
onsets.py`'s `OnsetPeakPickingProcessor` and `features/beats.py`'s
`DBNBeatTrackingProcessor`). Every class below is a plain `Processor` wired
directly to upstream's `process_offline` bodies; the `online=True`
constructor flag, `process_online`, `reset`, the `_hist_buffer`/
`_comb_buffer`/`_act_buffer` `BufferProcessor` state, and the visualisation
branch are all dropped entirely, not silently stubbed. An `online` keyword
is still silently ACCEPTED (via `**kwargs`, discarded) for the same reason
`features/beats.py`'s `DBNBeatTrackingProcessor` accepts it: nothing in
this project's call graph will ever pass `online=True` and expect streaming
behavior, but accepting-and-discarding the keyword lets constructor
signatures stay drop-in compatible with upstream's own (e.g.
`TempoEstimationProcessor.__init__` forwarding `**kwargs` into whichever
`TempoHistogramProcessor` subclass it builds).

Reads: madmom_infer/audio/comb_filters.py (CombFilterbankProcessor, this
wave's own new port), madmom_infer/audio/signal.py (smooth),
madmom_infer/features/beats.py (DBNBeatTrackingProcessor, reused by
DBNTempoHistogramProcessor), madmom_infer/processors.py (Processor); read
by: nothing yet (Wave 4c's own end-to-end target, alongside features/
beats.py).
"""

import warnings

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import argrelmax

from madmom_infer.audio.signal import smooth as smooth_signal
from madmom_infer.processors import Processor

METHOD = "comb"
ALPHA = 0.79
MIN_BPM = 40.
MAX_BPM = 250.
ACT_SMOOTH = 0.14
HIST_SMOOTH = 9
HIST_BUFFER = 10.
NO_TEMPO = np.nan


# ---------------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------------
def smooth_histogram(histogram, smooth):
    """Smooth `histogram`'s bins (not its corresponding delays/tempi) with
    `smooth` (an int -> Hamming-window size, or an explicit kernel array).

    Verbatim port of `madmom.features.tempo.smooth_histogram`
    (`tempo.py:32-58`).
    """
    return smooth_signal(histogram[0], smooth), histogram[1]


def interval_histogram_acf(activations, min_tau=1, max_tau=None):
    """Interval histogram of `activations` via auto-correlation.

    Verbatim port of `madmom.features.tempo.interval_histogram_acf`
    (`tempo.py:62-106`).
    """
    if activations.ndim != 1:
        raise NotImplementedError(
            "too many dimensions for autocorrelation interval histogram "
            "calculation.")
    if max_tau is None:
        max_tau = len(activations) - min_tau
    taus = list(range(min_tau, max_tau + 1))
    bins = []
    for tau in taus:
        bins.append(np.sum(np.abs(activations[tau:] * activations[0:-tau])))
    return np.array(bins), np.array(taus)


def interval_histogram_comb(activations, alpha, min_tau=1, max_tau=None):
    """Interval histogram of `activations` via a bank of resonating comb
    filters.

    Verbatim port of `madmom.features.tempo.interval_histogram_comb`
    (`tempo.py:109-163`).
    """
    from madmom_infer.audio.comb_filters import CombFilterbankProcessor

    if max_tau is None:
        max_tau = len(activations) - min_tau
    taus = np.arange(min_tau, max_tau + 1)
    cfb = CombFilterbankProcessor("backward", taus, alpha)
    if activations.ndim in (1, 2):
        act = cfb.process(activations)
        act_max = act == np.max(act, axis=-1)[..., np.newaxis]
        histogram_bins = np.sum(act * act_max, axis=0)
    else:
        raise NotImplementedError(
            "too many dimensions for comb filter interval histogram "
            "calculation.")
    return histogram_bins, taus


def dominant_interval(histogram, smooth=None):
    """Extract the dominant interval of `histogram`.

    Verbatim port of `madmom.features.tempo.dominant_interval`
    (`tempo.py:167-194`).
    """
    if smooth:
        histogram = smooth_histogram(histogram, smooth)
    return histogram[1][np.argmax(histogram[0])]


def detect_tempo(histogram, fps=None, interpolate=False):
    """Detect the dominant tempi from `histogram`.

    Verbatim port of `madmom.features.tempo.detect_tempo`
    (`tempo.py:198-255`).
    """
    bins, tempi = histogram
    if interpolate:
        interpolation_fn = interp1d(tempi, bins, "quadratic")
        tempi = np.arange(tempi[0], tempi[-1], 0.01)
        bins = interpolation_fn(tempi)
    if fps is not None:
        tempi = 60.0 * fps / tempi
    peaks = argrelmax(bins, mode="wrap")[0]
    if len(peaks) == 0:
        tempi = np.asarray([NO_TEMPO, 0.])
    elif len(peaks) == 1:
        tempi = np.asarray([tempi[peaks[0]], 1.])
    else:
        sorted_peaks = peaks[np.argsort(bins[peaks])[::-1]]
        strengths = bins[sorted_peaks]
        strengths /= np.sum(strengths)
        tempi = np.asarray(list(zip(tempi[sorted_peaks], strengths)))
    return np.atleast_2d(tempi)


# ---------------------------------------------------------------------------
# tempo histogram processor classes (OFFLINE ONLY, see module header)
# ---------------------------------------------------------------------------
class TempoHistogramProcessor(Processor):
    """Abstract base class for tempo histogram construction. Use one of
    `CombFilterTempoHistogramProcessor`, `ACFTempoHistogramProcessor`, or
    `DBNTempoHistogramProcessor`.

    Port of `madmom.features.tempo.TempoHistogramProcessor`
    (`tempo.py:259-314`), offline-only -- see module header.
    """

    def __init__(self, min_bpm, max_bpm, hist_buffer=HIST_BUFFER, fps=None,
                 **kwargs):
        # pylint: disable=unused-argument
        self.min_bpm = float(min_bpm)
        self.max_bpm = float(max_bpm)
        self.hist_buffer = hist_buffer
        self.fps = fps

    @property
    def min_interval(self):
        """Minimum beat interval [frames]."""
        return int(np.floor(60. * self.fps / self.max_bpm))

    @property
    def max_interval(self):
        """Maximum beat interval [frames]."""
        return int(np.ceil(60. * self.fps / self.min_bpm))

    @property
    def intervals(self):
        """Beat intervals [frames]."""
        return np.arange(self.min_interval, self.max_interval + 1)


class CombFilterTempoHistogramProcessor(TempoHistogramProcessor):
    """Tempo histogram via a bank of resonating comb filters.

    Port of `madmom.features.tempo.CombFilterTempoHistogramProcessor`,
    offline-only -- see module header. Wires directly to upstream's
    `process_offline` (`tempo.py:354-373`).
    """

    def __init__(self, min_bpm=MIN_BPM, max_bpm=MAX_BPM, alpha=ALPHA,
                 hist_buffer=HIST_BUFFER, fps=None, **kwargs):
        # pylint: disable=unused-argument
        super().__init__(min_bpm=min_bpm, max_bpm=max_bpm,
                          hist_buffer=hist_buffer, fps=fps, **kwargs)
        self.alpha = alpha

    def process(self, activations, **kwargs):
        """Compute the beat interval histogram with a bank of resonating
        comb filters."""
        # pylint: disable=arguments-differ, unused-argument
        return interval_histogram_comb(activations, self.alpha,
                                       self.min_interval, self.max_interval)


class ACFTempoHistogramProcessor(TempoHistogramProcessor):
    """Tempo histogram via autocorrelation.

    Port of `madmom.features.tempo.ACFTempoHistogramProcessor`,
    offline-only -- see module header. Wires directly to upstream's
    `process_offline` (`tempo.py:454-474`).
    """

    def __init__(self, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 hist_buffer=HIST_BUFFER, fps=None, **kwargs):
        # pylint: disable=unused-argument
        super().__init__(min_bpm=min_bpm, max_bpm=max_bpm,
                          hist_buffer=hist_buffer, fps=fps, **kwargs)

    def process(self, activations, **kwargs):
        """Compute the beat interval histogram with the autocorrelation
        function."""
        # pylint: disable=arguments-differ, unused-argument
        return interval_histogram_acf(activations, self.min_interval,
                                      self.max_interval)


class DBNTempoHistogramProcessor(TempoHistogramProcessor):
    """Tempo histogram via a dynamic Bayesian network (DBN) beat tracker.

    Port of `madmom.features.tempo.DBNTempoHistogramProcessor`,
    offline-only -- see module header. Wires directly to upstream's
    `process_offline` (`tempo.py:548-574`), reusing `features/beats.py`'s
    `DBNBeatTrackingProcessor` (4c) rather than re-instantiating the HMM
    machinery directly.
    """

    def __init__(self, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 hist_buffer=HIST_BUFFER, fps=None, **kwargs):
        # pylint: disable=unused-argument
        super().__init__(min_bpm=min_bpm, max_bpm=max_bpm,
                          hist_buffer=hist_buffer, fps=fps, **kwargs)
        from madmom_infer.features.beats import DBNBeatTrackingProcessor
        self.dbn = DBNBeatTrackingProcessor(
            min_bpm=self.min_bpm, max_bpm=self.max_bpm, fps=self.fps,
            **kwargs)

    def process(self, activations, **kwargs):
        """Compute the beat interval histogram with a DBN."""
        # pylint: disable=arguments-differ, unused-argument
        path, _ = self.dbn.hmm.viterbi(activations.astype(np.float32))
        intervals = self.dbn.st.state_intervals[path]
        bins = np.bincount(intervals,
                           minlength=self.dbn.st.intervals.max() + 1)
        bins = bins[self.dbn.st.intervals.min():]
        return bins, self.dbn.st.intervals


# ---------------------------------------------------------------------------
# tempo estimation processor
# ---------------------------------------------------------------------------
class TempoEstimationProcessor(Processor):
    """Estimate the dominant tempo/tempi from a beat activation function:
    smooth the activations, build a tempo histogram (one of `'comb'`,
    `'acf'`, `'dbn'`), smooth that histogram, and detect its peak(s).

    Port of `madmom.features.tempo.TempoEstimationProcessor`, offline-only
    -- see module header. Wires directly to upstream's `process_offline`
    (`tempo.py:786-811`).
    """

    def __init__(self, method=METHOD, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 act_smooth=ACT_SMOOTH, hist_smooth=HIST_SMOOTH, fps=None,
                 histogram_processor=None, interpolate=False, **kwargs):
        # pylint: disable=unused-argument
        if method is not None:
            warnings.warn(
                "Usage of `method` is deprecated as of version 0.17. "
                "Please pass a dedicated `TempoHistogramProcessor` "
                "instance as `histogram_processor`."
                "Functionality will be removed in version 0.19.")
            self.method = method
        self.act_smooth = act_smooth
        self.fps = fps
        if histogram_processor is None:
            if method == "acf":
                histogram_processor = ACFTempoHistogramProcessor
            elif method == "comb":
                histogram_processor = CombFilterTempoHistogramProcessor
            elif method == "dbn":
                histogram_processor = DBNTempoHistogramProcessor
                self.act_smooth = None
            else:
                raise ValueError("tempo histogram method unknown.")
            histogram_processor = histogram_processor(
                min_bpm=min_bpm, max_bpm=max_bpm, fps=fps, **kwargs)
        self.histogram_processor = histogram_processor
        self.fps = fps
        self.hist_smooth = hist_smooth
        self.interpolate = interpolate

    @property
    def min_bpm(self):
        """Minimum tempo [bpm]."""
        return self.histogram_processor.min_bpm

    @property
    def max_bpm(self):
        """Maximum tempo [bpm]."""
        return self.histogram_processor.max_bpm

    @property
    def intervals(self):
        """Beat intervals [frames]."""
        return self.histogram_processor.intervals

    @property
    def min_interval(self):
        """Minimum beat interval [frames]."""
        return self.histogram_processor.min_interval

    @property
    def max_interval(self):
        """Maximum beat interval [frames]."""
        return self.histogram_processor.max_interval

    def process(self, activations, **kwargs):
        """Detect the tempi from the (beat) activations."""
        # pylint: disable=arguments-differ, unused-argument
        if self.act_smooth is not None:
            act_smooth = int(round(self.fps * self.act_smooth))
            activations = smooth_signal(activations, act_smooth)
        histogram = self.histogram_processor(activations)
        histogram = smooth_histogram(histogram, self.hist_smooth)
        return detect_tempo(histogram, self.fps, interpolate=self.interpolate)

    def interval_histogram(self, activations, **kwargs):
        """Compute the histogram of the beat intervals."""
        return self.histogram_processor(activations, **kwargs)

    def dominant_interval(self, histogram):
        """Extract the dominant interval of `histogram`."""
        return dominant_interval(histogram, self.hist_smooth)
