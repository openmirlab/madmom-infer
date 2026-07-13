"""Reimplementation of madmom.features.onsets -- the complete pure-DSP
spectral-flux onset detection family (`spectral_diff`, `spectral_flux`,
`superflux`, `complex_flux`, `complex_domain`, `rectified_complex_domain`,
`high_frequency_content`, `modified_kullback_leibler`, `phase_deviation`,
`weighted_phase_deviation`, `normalized_weighted_phase_deviation`,
`correlation_diff`, `wrap_to_pi`) plus `SpectralOnsetProcessor` (the
configurable processor that wires any one of them onto a spectrogram
pre-processing chain), the two RNN/CNN activation-function processors
(`RNNOnsetProcessor`, `CNNOnsetProcessor`), and `peak_picking`/
`OnsetPeakPickingProcessor` (activations -> onset times, seconds). Wave 4b of
the complete-port campaign -- see CLAUDE.md's audit table, `features/
onsets.py` rows.

Port of `madmom-upstream/madmom/features/onsets.py` (1259 lines). Every
function/class the 4.0 audit table's `features/onsets.py` rows list as
TO-PORT is here.

`correlation_diff` uses explicit integer midpoint arithmetic so the original
algorithm works under Python 3.

**`OnsetPeakPickingProcessor` is offline-only** -- upstream subclasses
`OnlineProcessor` (`process_offline`/`process_online`/`reset`/a stateful
`BufferProcessor`-backed streaming continuation path), but this project's
processors are offline/whole-clip only (`madmom_infer/processors.py`'s
module header: `OnlineProcessor`/streaming machinery is a stated permanent
exclusion). This port is a plain `Processor` wired directly to upstream's
`process_offline` logic (`onsets.py:1076-1107`) -- the `online=True`
constructor flag, `process_online`, and `reset` are dropped entirely, not
silently stubbed.

**`RNNOnsetProcessor(online=True)` IS fully supported** (unlike
`OnsetPeakPickingProcessor`'s online mode) -- despite the name, upstream's
`online` flag on THIS processor only selects a different (causal,
unidirectional) pretrained RNN ensemble and smaller frame sizes; it does not
touch `FramedSignalProcessor`'s own framing/streaming behavior at all
(`onsets.py:768-773` -- the `online` kwarg is never forwarded into a
`origin='stream'` framing choice), so this is ordinary whole-clip batch
inference under a different set of weights, fully compatible with this
project's offline-only processor stack.

**Kwargs-forwarding note**: unlike `RNNDownBeatProcessor`/
`CNNKeyRecognitionProcessor` (which build their pipelines from fixed,
explicit stage parameters), `SpectralOnsetProcessor` below is a literal port
of upstream's own blind-`**kwargs`-forwarding design (one shared `kwargs`
dict threaded through `SignalProcessor`/`FramedSignalProcessor`/
`ShortTimeFourierTransformProcessor`/`SpectrogramProcessor`/
`FilteredSpectrogramProcessor`/`LogarithmicSpectrogramProcessor`) -- this
needed `audio/signal.py`'s `FramedSignalProcessor.__init__` and
`audio/spectrogram.py`'s `SpectrogramProcessor.__init__` to gain a `**kwargs`
catch-all (both Wave 4b additions, matching upstream's own signatures
exactly, see those modules' headers) since this project's other processors
had deliberately tightened to explicit-only signatures. `RNNOnsetProcessor`/
`CNNOnsetProcessor` do NOT use this pattern -- they build fixed pipelines
like their sibling processors, and additionally accept an explicit
`nn_files=None` override (not in upstream, matching this project's own
`CNNKeyRecognitionProcessor` convention) so the cross-BLAS test can point
them at a local `.pkl` file.

Reads: madmom_infer/audio/{signal,stft,spectrogram,filters}.py (the
pre-processing cascade + MelFilterbank), madmom_infer/ml/nn/__init__.py
(NeuralNetwork, NeuralNetworkEnsemble), madmom_infer/models.py
(onsets_rnn/onsets_brnn/onsets_cnn download), madmom_infer/processors.py
(Processor, ParallelProcessor, SequentialProcessor), madmom_infer/utils.py
(combine_events); read by: nothing yet (Wave 4b's own end-to-end target).
"""

import inspect

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter

from ..audio.filters import MelFilterbank
from ..audio.signal import (
    FramedSignalProcessor, SignalProcessor, smooth as smooth_signal,
)
from ..audio.spectrogram import (
    FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
    SpectrogramDifference, SpectrogramDifferenceProcessor, SpectrogramProcessor,
)
from ..audio.stft import ShortTimeFourierTransformProcessor
from ..processors import ParallelProcessor, Processor, SequentialProcessor
from ..utils import combine_events

EPSILON = np.spacing(1)


# ---------------------------------------------------------------------------
# onset detection helper functions
# ---------------------------------------------------------------------------
def wrap_to_pi(phase):
    """Wrap `phase` to the range -pi..pi.

    Verbatim port of `madmom.features.onsets.wrap_to_pi`
    (`onsets.py:24-39`).
    """
    return np.mod(phase + np.pi, 2.0 * np.pi) - np.pi


def correlation_diff(spec, diff_frames=1, pos=False, diff_bins=1):
    """Correlation-shifted difference of `spec` relative to the
    `diff_frames`-th previous frame.

    Port of `madmom.features.onsets.correlation_diff` (`onsets.py:42-95`),
    with Python 3 integer-index semantics.
    """
    diff_spec = np.zeros_like(spec)
    if diff_frames < 1:
        raise ValueError("number of `diff_frames` must be >= 1")
    frames, bins = diff_spec.shape
    corr = np.zeros((frames, diff_bins * 2 + 1))
    for f in range(diff_frames, frames):
        c = np.correlate(spec[f], spec[f - diff_frames], mode="full")
        # NOTE: `len(c) / 2` is a float under Python 3 (true division) --
        # this is the exact line that crashes, replicated verbatim, see
        # module header.
        centre = len(c) // 2
        corr[f] = c[centre - diff_bins: centre + diff_bins + 1]
        bin_offset = diff_bins - np.argmax(corr[f])
        bin_start = diff_bins + bin_offset
        bin_stop = bins - 2 * diff_bins + bin_start
        diff_spec[f, diff_bins:-diff_bins] = (
            spec[f, diff_bins:-diff_bins]
            - spec[f - diff_frames, bin_start:bin_stop]
        )
    if pos:
        np.maximum(diff_spec, 0, diff_spec)
    return np.asarray(diff_spec)


# ---------------------------------------------------------------------------
# onset detection functions pluggable into SpectralOnsetProcessor -- all
# expect a Spectrogram-like object (real `.stft`/`.filterbank` attributes,
# `.diff()` method) as their sole positional argument.
# ---------------------------------------------------------------------------
def high_frequency_content(spectrogram):
    """High Frequency Content onset detection function.

    Verbatim port of `madmom.features.onsets.high_frequency_content`
    (`onsets.py:102-127`).
    """
    hfc = spectrogram * np.arange(spectrogram.num_bins)
    return np.asarray(np.mean(hfc, axis=1))


def spectral_diff(spectrogram, diff_frames=None):
    """Spectral Diff onset detection function.

    Verbatim port of `madmom.features.onsets.spectral_diff`
    (`onsets.py:130-160`).
    """
    if not isinstance(spectrogram, SpectrogramDifference):
        spectrogram = spectrogram.diff(diff_frames=diff_frames,
                                        positive_diffs=True)
    # NOTE: `np.asarray(...)` here (not in upstream, which subclasses
    # np.ndarray so `spectrogram ** 2` works natively) -- this port's
    # composition-style wrapper classes (docs/DESIGN.md C.2) have no
    # `__pow__`; a plain ndarray on one side of `**` is required for numpy's
    # array-like-coercion fallback to kick in (a bare Python `int` RHS, like
    # the literal `2` here, does not trigger it). See this module's header.
    return np.asarray(np.sum(np.asarray(spectrogram) ** 2, axis=1))


def spectral_flux(spectrogram, diff_frames=None):
    """Spectral Flux onset detection function.

    Verbatim port of `madmom.features.onsets.spectral_flux`
    (`onsets.py:163-193`).
    """
    if not isinstance(spectrogram, SpectrogramDifference):
        spectrogram = spectrogram.diff(diff_frames=diff_frames,
                                        positive_diffs=True)
    return np.asarray(np.sum(spectrogram, axis=1))


def superflux(spectrogram, diff_frames=None, diff_max_bins=3):
    """SuperFlux onset detection function (max-filter vibrato suppression).

    Verbatim port of `madmom.features.onsets.superflux`
    (`onsets.py:196-240`).
    """
    if not isinstance(spectrogram, SpectrogramDifference):
        spectrogram = spectrogram.diff(diff_frames=diff_frames,
                                        diff_max_bins=diff_max_bins,
                                        positive_diffs=True)
    return np.asarray(np.sum(spectrogram, axis=1))


def complex_flux(spectrogram, diff_frames=None, diff_max_bins=3,
                  temporal_filter=3, temporal_origin=0):
    """ComplexFlux onset detection function (SuperFlux + local-group-delay
    based tremolo suppression).

    Verbatim port of `madmom.features.onsets.complex_flux`
    (`onsets.py:245-317`).
    """
    lgd = np.abs(spectrogram.stft.phase().lgd()) / np.pi
    if temporal_filter > 0:
        lgd = maximum_filter(lgd, size=[temporal_filter, 1],
                              origin=temporal_origin)
    try:
        mask = np.zeros_like(spectrogram)
        num_bins = lgd.shape[1]
        for b in range(mask.shape[1]):
            corner_bins = np.nonzero(spectrogram.filterbank[:, b])[0]
            start_bin = corner_bins[0] - 1
            stop_bin = corner_bins[-1] + 2
            if start_bin < 0:
                start_bin = 0
            if stop_bin > num_bins:
                stop_bin = num_bins
            mask[:, b] = np.amin(lgd[:, start_bin:stop_bin], axis=1)
    except AttributeError:
        mask = minimum_filter(lgd, size=[1, 3])
    diff = spectrogram.diff(diff_frames=diff_frames,
                             diff_max_bins=diff_max_bins,
                             positive_diffs=True)
    return np.asarray(np.sum(diff * mask, axis=1))


def modified_kullback_leibler(spectrogram, diff_frames=1, epsilon=EPSILON):
    """Modified Kullback-Leibler onset detection function.

    Verbatim port of `madmom.features.onsets.modified_kullback_leibler`
    (`onsets.py:320-362`).
    """
    if epsilon <= 0:
        raise ValueError("a positive value must be added before division")
    mkl = np.zeros_like(spectrogram)
    mkl[diff_frames:] = (spectrogram[diff_frames:]
                          / (spectrogram[:-diff_frames] + epsilon))
    return np.asarray(np.mean(np.log(1 + mkl), axis=1))


def _phase_deviation(phase):
    """Helper for `phase_deviation`/`weighted_phase_deviation`: second-order
    phase difference, wrapped to -pi..pi.

    Verbatim port of `madmom.features.onsets._phase_deviation`
    (`onsets.py:365-387`).
    """
    pd = np.zeros_like(phase)
    pd[2:] = phase[2:] - 2 * phase[1:-1] + phase[:-2]
    return np.asarray(wrap_to_pi(pd))


def phase_deviation(spectrogram):
    """Phase Deviation onset detection function.

    Verbatim port of `madmom.features.onsets.phase_deviation`
    (`onsets.py:390-414`).
    """
    pd = np.abs(_phase_deviation(spectrogram.stft.phase()))
    return np.asarray(np.mean(pd, axis=1))


def weighted_phase_deviation(spectrogram):
    """Weighted Phase Deviation onset detection function.

    Verbatim port of `madmom.features.onsets.weighted_phase_deviation`
    (`onsets.py:417-446`). Requires an UNFILTERED spectrogram -- raises
    `ValueError` if `spectrogram`'s shape doesn't match its own STFT's raw
    phase shape (matches upstream exactly, not this port's own addition).
    """
    phase = spectrogram.stft.phase()
    if np.shape(phase) != np.shape(spectrogram):
        raise ValueError("spectrogram and phase must be of same shape")
    wpd = np.abs(_phase_deviation(phase) * spectrogram)
    return np.asarray(np.mean(wpd, axis=1))


def normalized_weighted_phase_deviation(spectrogram, epsilon=EPSILON):
    """Normalized Weighted Phase Deviation onset detection function.

    Verbatim port of
    `madmom.features.onsets.normalized_weighted_phase_deviation`
    (`onsets.py:449-478`).
    """
    if epsilon <= 0:
        raise ValueError("a positive value must be added before division")
    # NOTE: `.astype(np.float32)` here (not in upstream) -- `epsilon`
    # (`EPSILON = np.spacing(1)`) is a genuine numpy `float64` SCALAR, not a
    # plain Python `float`. Under numpy >= 2.0's NEP 50 strict scalar-
    # promotion rules, `float32_array + np.float64_scalar` upcasts the
    # result to `float64`; numpy < 2.0 (including the reference venv's
    # 1.23.5, real madmom's own environment) keeps `float32` via legacy
    # value-based scalar casting. Confirmed empirically: without this cast,
    # this port's output dtype (and therefore its bit pattern) silently
    # diverges from real madmom's own recorded `float32` output on numpy
    # >= 2.0 -- same numpy-2.x-incompatibility class as docs/DESIGN.md C.1.
    norm = np.add(np.mean(spectrogram, axis=1), epsilon).astype(np.float32)
    return np.asarray(weighted_phase_deviation(spectrogram) / norm)


def _complex_domain(spectrogram):
    """Helper for `complex_domain`/`rectified_complex_domain`: complex
    spectrogram minus its constant-phase-change-predicted target.

    Verbatim port of `madmom.features.onsets._complex_domain`
    (`onsets.py:481-522`). Requires an UNFILTERED spectrogram (same shape
    constraint as `weighted_phase_deviation`).
    """
    phase = spectrogram.stft.phase()
    if np.shape(phase) != np.shape(spectrogram):
        raise ValueError("spectrogram and phase must be of same shape")
    cd_target = np.zeros_like(phase)
    cd_target[1:] = 2 * phase[1:] - phase[:-1]
    cd_target = spectrogram * np.exp(1j * cd_target)
    # NOTE: `np.asarray(phase)` here (not in upstream -- see spectral_diff's
    # comment on this port's wrapper classes vs. upstream's np.ndarray
    # subclassing): `phase` is the whole, un-sliced `Phase` wrapper at this
    # point (unlike `phase[1:]`/`phase[:-1]` above, which already de-wrap to
    # plain arrays via `__getitem__`), and `1j * phase` (scalar-times-wrapper,
    # no ndarray on either side) has no numpy coercion path to fall back on.
    cd = spectrogram * np.exp(1j * np.asarray(phase))
    cd[1:] -= cd_target[:-1]
    return np.asarray(cd)


def complex_domain(spectrogram):
    """Complex Domain onset detection function.

    Verbatim port of `madmom.features.onsets.complex_domain`
    (`onsets.py:525-548`).
    """
    return np.asarray(np.sum(np.abs(_complex_domain(spectrogram)), axis=1))


def rectified_complex_domain(spectrogram, diff_frames=None):
    """Rectified Complex Domain onset detection function.

    Verbatim port of `madmom.features.onsets.rectified_complex_domain`
    (`onsets.py:551-581`).
    """
    rcd = _complex_domain(spectrogram)
    pos_diff = spectrogram.diff(diff_frames=diff_frames, positive_diffs=True)
    # NOTE: `np.asarray(pos_diff)` here (not in upstream) -- this port's
    # `SpectrogramDifference` wrapper has no `.astype()` method (only real
    # np.ndarray does); see spectral_diff's comment for the general pattern.
    rcd *= np.asarray(pos_diff).astype(bool)
    return np.asarray(np.sum(np.abs(rcd), axis=1))


class SpectralOnsetProcessor(SequentialProcessor):
    """Configurable processor implementing most of the common spectral-flux
    onset detection functions above.

    Port of `madmom.features.onsets.SpectralOnsetProcessor`
    (`onsets.py:584-712`) -- see this module's header for the
    kwargs-forwarding note (why `FramedSignalProcessor`/`SpectrogramProcessor`
    needed a `**kwargs` catch-all added elsewhere in this wave to support
    this class's literal upstream design).

    Examples
    --------
    >>> from madmom_infer.features.onsets import SpectralOnsetProcessor
    >>> sodf = SpectralOnsetProcessor()
    >>> sodf('track.wav')  # doctest: +SKIP
    array([...], dtype=float32)
    """

    METHODS = ["superflux", "complex_flux", "high_frequency_content",
               "spectral_diff", "spectral_flux", "modified_kullback_leibler",
               "phase_deviation", "weighted_phase_deviation",
               "normalized_weighted_phase_deviation", "complex_domain",
               "rectified_complex_domain"]

    def __init__(self, onset_method="spectral_flux", **kwargs):
        # for certain methods we need to circular shift the signal before STFT
        if any(odf in onset_method for odf in ("phase", "complex")):
            kwargs["circular_shift"] = True
        # always use mono signals
        kwargs["num_channels"] = 1
        # define processing chain
        sig = SignalProcessor(**kwargs)
        frames = FramedSignalProcessor(**kwargs)
        stft = ShortTimeFourierTransformProcessor(**kwargs)
        spec = SpectrogramProcessor(**kwargs)
        processors = [sig, frames, stft, spec]
        # filtering needed?
        if "filterbank" in kwargs and kwargs["filterbank"] is not None:
            processors.append(FilteredSpectrogramProcessor(**kwargs))
        # scaling needed?
        if "log" in kwargs and kwargs["log"] is not None:
            processors.append(LogarithmicSpectrogramProcessor(**kwargs))
        # odf function
        # NOTE: upstream only `processors.append(onset_method)` INSIDE the
        # `not inspect.isfunction(...)` branch (onsets.py:676-682) -- if a
        # caller passes an already-callable `onset_method` directly (instead
        # of one of the `METHODS` strings), it is silently NOT appended to
        # the pipeline. This looks like an upstream oversight, not a
        # documented feature, but this port replicates it exactly rather
        # than "fixing" it -- see module header, bit-parity mandate.
        if not inspect.isfunction(onset_method):
            try:
                onset_method = globals()[onset_method]
            except KeyError:
                raise ValueError(
                    "%s not a valid onset detection function, choose %s."
                    % (onset_method, self.METHODS)
                )
            processors.append(onset_method)
        super().__init__(processors)


# ---------------------------------------------------------------------------
# classes for detecting onsets with NNs
# ---------------------------------------------------------------------------
class RNNOnsetProcessor(SequentialProcessor):
    """Onset activation function from an ensemble of RNNs.

    Port of `madmom.features.onsets.RNNOnsetProcessor`
    (`onsets.py:716-797`). `online=True` selects the causal, unidirectional
    `ONSETS_RNN` ensemble (smaller frame sizes); the default `online=False`
    selects the bidirectional `ONSETS_BRNN` ensemble -- see this module's
    header for why `online=True` is fully offline-compatible despite the
    name (it only changes which pretrained weights/frame sizes are used).
    `nn_files`, if given, overrides the model list entirely (not in
    upstream -- matches `CNNKeyRecognitionProcessor`'s own convention, used
    by the cross-BLAS test to point at a local `.pkl` copy).
    """

    def __init__(self, online=False, nn_files=None, **kwargs):
        from ..ml.nn import NeuralNetworkEnsemble
        from ..models import onsets_brnn, onsets_rnn

        if online:
            model_files = nn_files or onsets_rnn()
            frame_sizes = [512, 1024, 2048]
        else:
            model_files = nn_files or onsets_brnn()
            frame_sizes = [1024, 2048, 4096]

        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        multi = ParallelProcessor([])
        for frame_size in frame_sizes:
            frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
            stft = ShortTimeFourierTransformProcessor()  # caching FFT window
            filt = FilteredSpectrogramProcessor(
                num_bands=6, fmin=30, fmax=17000, norm_filters=True)
            spec = LogarithmicSpectrogramProcessor(mul=5, add=1)
            diff = SpectrogramDifferenceProcessor(
                diff_ratio=0.25, positive_diffs=True, stack_diffs=np.hstack)
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
        pre_processor = SequentialProcessor((sig, multi, np.hstack))

        nn = NeuralNetworkEnsemble.load(model_files, **kwargs)
        super().__init__((pre_processor, nn))


def _cnn_onset_processor_pad(data):
    """Pad `data` by repeating its first and last frame 7 times.

    Verbatim port of `madmom.features.onsets._cnn_onset_processor_pad`
    (`onsets.py:801-805`) -- must be a top-level function (not a closure) to
    stay picklable, matching upstream's own comment, even though this port
    never actually pickles a `SequentialProcessor`.
    """
    pad_start = np.repeat(data[:1], 7, axis=0)
    pad_stop = np.repeat(data[-1:], 7, axis=0)
    return np.concatenate((pad_start, data, pad_stop))


class CNNOnsetProcessor(SequentialProcessor):
    """Onset activation function from a CNN.

    Port of `madmom.features.onsets.CNNOnsetProcessor`
    (`onsets.py:808-871`). `nn_files`, if given, overrides `ONSETS_CNN`
    (not in upstream -- see `RNNOnsetProcessor`'s docstring for why).
    """

    def __init__(self, nn_files=None, **kwargs):
        # pylint: disable=unused-argument
        from ..ml.nn import NeuralNetwork
        from ..models import onsets_cnn

        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        multi = ParallelProcessor([])
        for frame_size in [2048, 1024, 4096]:
            frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
            stft = ShortTimeFourierTransformProcessor()  # caching FFT window
            filt = FilteredSpectrogramProcessor(
                filterbank=MelFilterbank, num_bands=80, fmin=27.5,
                fmax=16000, norm_filters=True, unique_filters=False)
            spec = LogarithmicSpectrogramProcessor(log=np.log, add=EPSILON)
            multi.append(SequentialProcessor((frames, stft, filt, spec)))
        stack = np.dstack
        pad = _cnn_onset_processor_pad
        pre_processor = SequentialProcessor((sig, multi, stack, pad))

        model_files = nn_files or onsets_cnn()
        nn = NeuralNetwork.load(model_files[0])
        super().__init__((pre_processor, nn))


# ---------------------------------------------------------------------------
# universal peak-picking method
# ---------------------------------------------------------------------------
def peak_picking(activations, threshold, smooth=None, pre_avg=0, post_avg=0,
                  pre_max=1, post_max=1):
    """Threshold and peak-pick the given activation function.

    Verbatim port of `madmom.features.onsets.peak_picking`
    (`onsets.py:875-963`).
    """
    activations = smooth_signal(activations, smooth)
    avg_length = pre_avg + post_avg + 1
    if avg_length > 1:
        avg_origin = int(np.floor((pre_avg - post_avg) / 2))
        if activations.ndim == 1:
            filter_size = avg_length
        elif activations.ndim == 2:
            filter_size = [avg_length, 1]
        else:
            raise ValueError("`activations` must be either 1D or 2D")
        mov_avg = uniform_filter(activations, filter_size, mode="constant",
                                  origin=avg_origin)
    else:
        mov_avg = 0
    detections = activations * (activations >= mov_avg + threshold)
    max_length = pre_max + post_max + 1
    if max_length > 1:
        max_origin = int(np.floor((pre_max - post_max) / 2))
        if activations.ndim == 1:
            filter_size = max_length
        elif activations.ndim == 2:
            filter_size = [max_length, 1]
        else:
            raise ValueError("`activations` must be either 1D or 2D")
        mov_max = maximum_filter(detections, filter_size, mode="constant",
                                  origin=max_origin)
        detections *= (detections == mov_max)
    if activations.ndim == 1:
        return np.nonzero(detections)[0]
    elif activations.ndim == 2:
        return np.nonzero(detections)
    else:
        raise ValueError("`activations` must be either 1D or 2D")


class OnsetPeakPickingProcessor(Processor):
    """Onset peak-picking: converts an onset activation function (frame
    indices) into onset times (seconds).

    Port of `madmom.features.onsets.OnsetPeakPickingProcessor`, OFFLINE ONLY
    -- see this module's header for why (`OnlineProcessor` is a stated
    permanent exclusion in this project). Wires directly to upstream's
    `process_offline` (`onsets.py:1076-1107`).

    Examples
    --------
    >>> from madmom_infer.features.onsets import OnsetPeakPickingProcessor
    >>> proc = OnsetPeakPickingProcessor(fps=100)
    >>> proc(act)  # doctest: +SKIP
    array([0.09, 0.29, 0.45, ..., 2.34, 2.49, 2.67])
    """

    FPS = 100
    THRESHOLD = 0.5
    SMOOTH = 0.0
    PRE_AVG = 0.0
    POST_AVG = 0.0
    PRE_MAX = 0.0
    POST_MAX = 0.0
    COMBINE = 0.03
    DELAY = 0.0

    def __init__(self, threshold=THRESHOLD, smooth=SMOOTH, pre_avg=PRE_AVG,
                 post_avg=POST_AVG, pre_max=PRE_MAX, post_max=POST_MAX,
                 combine=COMBINE, delay=DELAY, fps=FPS, **kwargs):
        # pylint: disable=unused-argument
        self.threshold = threshold
        self.smooth = smooth
        self.pre_avg = pre_avg
        self.post_avg = post_avg
        self.pre_max = pre_max
        self.post_max = post_max
        self.combine = combine
        self.delay = delay
        self.fps = fps

    def process(self, activations, **kwargs):
        """Detect onsets in `activations`, an onset activation function.

        Matches `OnsetPeakPickingProcessor.process_offline`
        (`onsets.py:1076-1107`).
        """
        # pylint: disable=arguments-differ, unused-argument
        timings = np.array([self.smooth, self.pre_avg, self.post_avg,
                             self.pre_max, self.post_max]) * self.fps
        timings = np.round(timings).astype(int)
        onsets = peak_picking(activations, self.threshold, *timings)
        onsets = onsets.astype(float) / self.fps
        if self.delay:
            onsets += self.delay
        if self.combine:
            onsets = combine_events(onsets, self.combine, "left")
        return np.asarray(onsets)
