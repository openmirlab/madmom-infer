"""Magnitude/filtered/log spectrograms -- composition port of
madmom.audio.spectrogram, the stage that turns a `ShortTimeFourierTransform`
into the filtered, log-compressed feature all-in-one-infer's models consume.

Phase-1 scope: `Spectrogram` (magnitude of the STFT), `FilteredSpectrogram` /
`FilteredSpectrogramProcessor` (apply a `filters.Filterbank`), and
`LogarithmicSpectrogram` / `LogarithmicSpectrogramProcessor` (log-compress) --
exactly the stages `all-in-one-infer`'s `build_spec_processor()` composes via
`SequentialProcessor([frames, stft, filt, spec])`
(`all-in-one-fix/src/allin1_infer/spectrogram.py:27-40`).

Phase-2 addition: `SpectrogramDifference`/`SpectrogramDifferenceProcessor`
(SuperFlux-style temporal first-order difference), needed because
`RNNDownBeatProcessor` (`madmom_infer/features/downbeats.py`) stacks a
diff-of-log-spectrogram feature onto each of its 3 frame-size branches
(`madmom-upstream/madmom/features/downbeats.py:84-87`).

Wave 4b addition: `SuperFluxProcessor` (`features/onsets.py`'s `superflux`
onset detection function needs its exact default parameterization), plus
`Spectrogram.diff()`/`.filter()`/`.log()` convenience methods (the pure-DSP
onset functions call `spectrogram.diff(...)` directly) and
`SpectrogramProcessor.__init__(self, **kwargs): pass` (needed for
`SuperFluxProcessor` to construct it with forwarded kwargs). Still NOT
ported (no call site so far, `madmom-upstream/madmom/audio/spectrogram.py`):
`MultiBandSpectrogram`/`MultiBandSpectrogramProcessor`,
`SemitoneBandpassSpectrogram`, `LogarithmicFilteredSpectrogram`/
`LogarithmicFilteredSpectrogramProcessor` (the fused filter+log convenience
class -- this project composes the same pipeline via two separate stages
instead, exactly like `build_spec_processor()` already does), and
`tuning_frequency()`/`Spectrogram.tuning_frequency()`.

Composition, not `np.ndarray` subclassing (docs/DESIGN.md C.2): each class
wraps a plain array in `.data`, and `FilteredSpectrogram`/
`LogarithmicSpectrogram`/`SpectrogramDifference` are ordinary Python
subclasses of `Spectrogram` (inheriting its array-interop dunders/properties
for free) rather than `np.ndarray` subclasses relying on
`__array_finalize__`.

Reads: madmom_infer/audio/stft.py (ShortTimeFourierTransform),
madmom_infer/audio/filters.py (Filterbank, LogarithmicFilterbank),
madmom_infer/processors.py (Processor, BufferProcessor), scipy.ndimage
(maximum_filter); read by: madmom_infer/features/downbeats.py
(RNNDownBeatProcessor's pre-processing cascade).
"""

import inspect

import numpy as np
from scipy.ndimage import maximum_filter

from ..processors import BufferProcessor, Processor, SequentialProcessor
from .filters import (
    FMAX,
    FMIN,
    NUM_BANDS,
    A4,
    Filterbank,
    LogarithmicFilterbank,
    NORM_FILTERS,
    UNIQUE_FILTERS,
)
from .stft import ShortTimeFourierTransform, ShortTimeFourierTransformProcessor

FILTERBANK = LogarithmicFilterbank

LOG = np.log10
MUL = 1.0
ADD = 1.0


def _stft_magnitude(stft_data):
    """Magnitude of a `complex64` STFT array, computed to be reproducible
    across numpy versions/builds.

    **Bit-identity finding, not a stylistic choice**: plain `np.abs()` on a
    `complex64` array is NOT correctly rounded on every numpy build --
    verified empirically (against `tests/fixtures/filterbank.npz`, cross-
    checked with `mpmath` at 50-digit precision) that this numpy/scipy
    environment's `np.abs(complex64_array)` disagrees with the mathematically
    correctly-rounded magnitude in roughly a third of bins (by exactly 1
    float32 ULP each time), while the real madmom install the fixtures were
    generated against (numpy 1.23.5) does NOT have this discrepancy --
    computing `sqrt(re**2 + im**2)` in `float64` and only rounding to
    `float32` at the very end reproduces the correctly-rounded value in
    100% of cases checked. Using `np.abs()` directly would make this port's
    output depend on which numpy version/build a user happens to have
    installed -- not a difference this project's dual-backend, golden-
    fixture-verified design can accept. This is the one deliberate departure
    from a literal `np.abs(stft)` port of `madmom.audio.spectrogram.spec`/
    `Spectrogram.__new__` (`madmom-upstream/madmom/audio/spectrogram.py:22-38,
    122-135`) -- mathematically equivalent, numerically more portable.
    """
    real64 = stft_data.real.astype(np.float64)
    imag64 = stft_data.imag.astype(np.float64)
    return np.sqrt(real64 * real64 + imag64 * imag64).astype(np.float32)


class Spectrogram:
    """The magnitude spectrogram of a `ShortTimeFourierTransform`.

    Composition port of `madmom.audio.spectrogram.Spectrogram`
    (`spectrogram.py:86-236`, minus `tuning_frequency()` -- see module
    header). `.data` is `np.abs(stft)`, a `float32` array (since the STFT's
    `complex64` dtype's magnitude is `float32`).
    """

    def __init__(self, stft, **kwargs):
        if isinstance(stft, Spectrogram):
            # already a Spectrogram
            self.data = stft.data
            self.stft = stft.stft
            return
        if not isinstance(stft, ShortTimeFourierTransform):
            # try to instantiate a ShortTimeFourierTransform
            stft = ShortTimeFourierTransform(stft, **kwargs)
        # take the abs of the STFT -- see _stft_magnitude()'s docstring for
        # why this is NOT simply `np.abs(np.asarray(stft))`
        self.data = _stft_magnitude(np.asarray(stft))
        self.stft = stft

    # -- numpy interop, mirroring audio/signal.py's Signal -------------
    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def num_frames(self):
        """Number of frames."""
        return len(self.data)

    @property
    def num_bins(self):
        """Number of bins."""
        return int(self.data.shape[1])

    @property
    def bin_frequencies(self):
        """Bin frequencies."""
        return self.stft.bin_frequencies

    # -- Wave 4b addition: convenience constructors, needed by
    # features/onsets.py's spectral_diff/spectral_flux/superflux (each calls
    # `spectrogram.diff(...)`) -----------------------------------------
    def diff(self, **kwargs):
        """Return the `SpectrogramDifference` of this spectrogram. Port of
        `Spectrogram.diff` (`spectrogram.py:164-179`)."""
        return SpectrogramDifference(self, **kwargs)

    def filter(self, **kwargs):
        """Return the `FilteredSpectrogram` of this spectrogram. Port of
        `Spectrogram.filter` (`spectrogram.py:181-196`)."""
        return FilteredSpectrogram(self, **kwargs)

    def log(self, **kwargs):
        """Return the `LogarithmicSpectrogram` of this spectrogram. Port of
        `Spectrogram.log` (`spectrogram.py:198-213`)."""
        return LogarithmicSpectrogram(self, **kwargs)


class SpectrogramProcessor(Processor):
    """Processor wrapper: compute the magnitude `Spectrogram` of an STFT.

    Port of `madmom.audio.spectrogram.SpectrogramProcessor`
    (`spectrogram.py:239-264`). Not on all-in-one-infer's own chain (which
    goes straight from STFT to `FilteredSpectrogramProcessor`, which builds
    its own internal `Spectrogram`), kept for API completeness/composability.
    `__init__(self, **kwargs): pass` (Wave 4b addition, matching upstream's
    own `SpectrogramProcessor.__init__` exactly) -- needed because
    `SuperFluxProcessor` instantiates `SpectrogramProcessor(**kwargs)` with
    whatever extra kwargs its own caller passed through; the base
    `Processor`/`object.__init__` has no catch-all and would raise
    `TypeError` on any non-empty kwargs otherwise.
    """

    def __init__(self, **kwargs):
        # pylint: disable=unused-argument
        pass

    def process(self, data, **kwargs):
        return Spectrogram(data, **kwargs)


class FilteredSpectrogram(Spectrogram):
    """A `Spectrogram` filtered through a `filters.Filterbank`.

    Composition port of `madmom.audio.spectrogram.FilteredSpectrogram`
    (`spectrogram.py:271-403`). `filterbank` may be a `Filterbank` class
    (one is constructed with `num_bands`/`fmin`/`fmax`/`fref`/
    `norm_filters`/`unique_filters`) or an already-built instance.
    """

    def __init__(self, spectrogram, filterbank=FILTERBANK, num_bands=NUM_BANDS,
                 fmin=FMIN, fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                 unique_filters=UNIQUE_FILTERS, **kwargs):
        if not isinstance(spectrogram, Spectrogram):
            spectrogram = Spectrogram(spectrogram, **kwargs)
        if inspect.isclass(filterbank) and issubclass(filterbank, Filterbank):
            filterbank = filterbank(
                spectrogram.bin_frequencies, num_bands=num_bands, fmin=fmin,
                fmax=fmax, fref=fref, norm_filters=norm_filters,
                unique_filters=unique_filters,
            )
        if not isinstance(filterbank, Filterbank):
            raise TypeError(
                "not a Filterbank type or instance: %s" % filterbank
            )
        # filter the spectrogram
        self.data = np.dot(np.asarray(spectrogram), np.asarray(filterbank))
        self.filterbank = filterbank
        self.stft = spectrogram.stft

    @property
    def bin_frequencies(self):
        """Bin frequencies (the filterbank's center frequencies)."""
        return self.filterbank.center_frequencies


class FilteredSpectrogramProcessor(Processor):
    """Processor wrapper: filter a `Spectrogram`/STFT through a filterbank.

    Composition port of
    `madmom.audio.spectrogram.FilteredSpectrogramProcessor`
    (`spectrogram.py:406-470`). `num_bands` is bands PER OCTAVE for the
    default `LogarithmicFilterbank` -- see `filters.py`'s module header.

    **Third caching gotcha, LOUDLY FLAGGED (found empirically while porting,
    not called out in the original task brief -- same shape of bug as
    `ShortTimeFourierTransformProcessor`'s window-caching trap, see
    `audio/stft.py`'s module header).** `process()` caches whichever
    `Filterbank` INSTANCE its `FilteredSpectrogram` ended up building
    (`self.filterbank = data.filterbank`, matching upstream
    `spectrogram.py:468-469`), and passes that cached instance into the next
    call's `args`. Since `FilteredSpectrogram.__init__` only builds a fresh
    filterbank when it's given a CLASS (`inspect.isclass(filterbank) and
    issubclass(filterbank, Filterbank)`) -- not when it's already an
    instance -- **a reused `FilteredSpectrogramProcessor` silently keeps the
    FIRST call's filterbank (built for that call's `bin_frequencies`, i.e.
    that call's sample rate) on every later call, even at a different
    sample rate**, with no error or warning. Confirmed empirically against
    the real madmom install: two calls through one shared instance --
    first at 44.1kHz, second at 48kHz -- yield `filtered_2.filterbank is
    filtered_1.filterbank == True`. This is exactly how
    `tests/fixtures/filterbank.npz`'s `filterbank_matrix_48000` was
    (unintentionally) recorded: `tools/generate_fixtures.py`'s
    `generate_filterbank_fixtures()`/`generate_logspec_fixtures()` each
    declare ONE `FilteredSpectrogramProcessor` and reuse it across both the
    44.1kHz and 48kHz test cases, so `filterbank_matrix_48000` is actually a
    byte-for-byte copy of `filterbank_matrix_44100` (see
    `tests/test_filters.py`'s dedicated pinned-behavior test) -- NOT a
    correctly-built 48kHz filterbank (compare with `full_chain.npz`, whose
    generator builds a fresh `SequentialProcessor`/`FilteredSpectrogramProcessor`
    per case and does NOT hit this bug). Harmless for all-in-one-infer today
    (its `build_spec_processor()` is constructed once per analysis at one
    fixed sample rate), but a real trap for any future multi-sample-rate
    caller reusing one processor instance.
    """

    def __init__(self, filterbank=FILTERBANK, num_bands=NUM_BANDS, fmin=FMIN,
                 fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                 unique_filters=UNIQUE_FILTERS, **kwargs):
        self.filterbank = filterbank
        self.num_bands = num_bands
        self.fmin = fmin
        self.fmax = fmax
        self.fref = fref
        self.norm_filters = norm_filters
        self.unique_filters = unique_filters

    def process(self, data, **kwargs):
        """Create a `FilteredSpectrogram` from the given data (an STFT)."""
        args = dict(filterbank=self.filterbank, num_bands=self.num_bands,
                    fmin=self.fmin, fmax=self.fmax, fref=self.fref,
                    norm_filters=self.norm_filters,
                    unique_filters=self.unique_filters)
        args.update(kwargs)
        data = FilteredSpectrogram(data, **args)
        # cache the filterbank, matching upstream (spectrogram.py:468-469)
        self.filterbank = data.filterbank
        return data


class LogarithmicSpectrogram(Spectrogram):
    """A logarithmically-scaled `Spectrogram`.

    Composition port of `madmom.audio.spectrogram.LogarithmicSpectrogram`
    (`spectrogram.py:479-563`): `log(mul * spectrogram + add)`, applied
    in-place on a private copy of the input's data (so the input spectrogram
    itself is never mutated).
    """

    def __init__(self, spectrogram, log=LOG, mul=MUL, add=ADD, **kwargs):
        if not isinstance(spectrogram, Spectrogram):
            spectrogram = Spectrogram(spectrogram, **kwargs)
        data = np.array(spectrogram.data, copy=True)
        if mul is not None:
            data *= mul
        if add is not None:
            data += add
        if log is not None:
            log(data, data)
        self.data = data
        self.mul = mul
        self.add = add
        self.stft = spectrogram.stft
        self.spectrogram = spectrogram

    @property
    def filterbank(self):
        """Filterbank (forwarded from the underlying spectrogram, if any)."""
        return self.spectrogram.filterbank

    @property
    def bin_frequencies(self):
        """Bin frequencies."""
        return self.spectrogram.bin_frequencies


class LogarithmicSpectrogramProcessor(Processor):
    """Processor wrapper: logarithmically scale a `Spectrogram`.

    Port of `madmom.audio.spectrogram.LogarithmicSpectrogramProcessor`
    (`spectrogram.py:566-660`, minus `add_arguments` -- argparse plumbing,
    out of phase-1 scope per CLAUDE.md).
    """

    def __init__(self, log=LOG, mul=MUL, add=ADD, **kwargs):
        self.log = log
        self.mul = mul
        self.add = add

    def process(self, data, **kwargs):
        """Create a `LogarithmicSpectrogram` from the given data."""
        args = dict(log=self.log, mul=self.mul, add=self.add)
        args.update(kwargs)
        return LogarithmicSpectrogram(data, **args)


# ---------------------------------------------------------------------------
# SpectrogramDifference -- Phase 2 addition (RNNDownBeatProcessor needs it)
# ---------------------------------------------------------------------------
DIFF_RATIO = 0.5
DIFF_FRAMES = None
DIFF_MAX_BINS = None
POSITIVE_DIFFS = False


def _diff_frames(diff_ratio, hop_size, frame_size, window=np.hanning):
    """Compute the number of `diff_frames` for the given ratio of overlap.

    Verbatim port of `madmom.audio.spectrogram._diff_frames`
    (`madmom-upstream/madmom/audio/spectrogram.py:834-862`): first sample of
    the window whose magnitude exceeds `diff_ratio` of the window's maximum,
    translated from a sample offset to a (rounded, floor-1) number of frames.
    """
    if hasattr(window, "__call__"):
        window = window(frame_size)
    sample = np.argmax(window > float(diff_ratio) * max(window))
    diff_samples = len(window) / 2 - sample
    return int(max(1, round(diff_samples / hop_size)))


class SpectrogramDifference(Spectrogram):
    """The temporal first-order difference of a `Spectrogram`.

    Composition port of `madmom.audio.spectrogram.SpectrogramDifference`
    (`spectrogram.py:867-1021`). `diff_frames` (if not given explicitly) is
    derived from `diff_ratio` via `_diff_frames`, using the underlying STFT's
    `frames.hop_size`/`frames.frame_size`/`window` -- this is why
    `spectrogram` must already carry a real `.stft` (i.e. be a `Spectrogram`/
    `FilteredSpectrogram`/`LogarithmicSpectrogram` instance, not a bare
    array). `diff_max_bins`, if set, widens the spectrogram this difference
    is computed AGAINST (not the spectrogram being subtracted FROM) via a
    frequency-axis `scipy.ndimage.maximum_filter` -- the "SuperFlux" vibrato-
    suppression trick (see upstream docstring, `spectrogram.py:903-912`);
    `RNNDownBeatProcessor` does not use this (leaves `diff_max_bins=None`),
    so it is ported but not exercised by this project's own fixtures.

    `keep_dims=True` (the default, for direct/API use) zero-pads the first
    `diff_frames` rows so the output has the same shape as the input.
    `keep_dims=False` (what `SpectrogramDifferenceProcessor` always uses,
    see below) instead returns a `diff_frames`-shorter array with no padding
    -- the processor buffers the missing context itself across calls.
    """

    def __init__(self, spectrogram, diff_ratio=DIFF_RATIO,
                 diff_frames=DIFF_FRAMES, diff_max_bins=DIFF_MAX_BINS,
                 positive_diffs=POSITIVE_DIFFS, keep_dims=True, **kwargs):
        if not isinstance(spectrogram, Spectrogram):
            spectrogram = Spectrogram(spectrogram, **kwargs)

        if diff_frames is None:
            diff_frames = _diff_frames(
                diff_ratio, hop_size=spectrogram.stft.frames.hop_size,
                frame_size=spectrogram.stft.frames.frame_size,
                window=spectrogram.stft.window,
            )

        spec_arr = np.asarray(spectrogram)
        if diff_max_bins is not None and diff_max_bins > 1:
            size = (1, int(diff_max_bins))
            diff_spec = maximum_filter(spec_arr, size=size)
        else:
            diff_spec = spec_arr

        if keep_dims:
            diff = np.zeros_like(spec_arr)
            diff[diff_frames:] = spec_arr[diff_frames:] - diff_spec[:-diff_frames]
        else:
            diff = spec_arr[diff_frames:] - diff_spec[:-diff_frames]

        if positive_diffs:
            np.maximum(diff, 0, out=diff)

        self.data = diff
        self.spectrogram = spectrogram
        self.stft = spectrogram.stft
        self.diff_ratio = diff_ratio
        self.diff_frames = diff_frames
        self.diff_max_bins = diff_max_bins
        self.positive_diffs = positive_diffs

    @property
    def bin_frequencies(self):
        """Bin frequencies (forwarded from the underlying spectrogram)."""
        return self.spectrogram.bin_frequencies


class SpectrogramDifferenceProcessor(Processor):
    """Processor wrapper: temporal difference of a `Spectrogram`, buffered
    across calls so streaming callers don't need `keep_dims`-padding.

    Port of `madmom.audio.spectrogram.SpectrogramDifferenceProcessor`
    (`spectrogram.py:1025-1230`, minus `add_arguments` -- argparse plumbing,
    out of scope per CLAUDE.md). `RNNDownBeatProcessor` uses this with
    `diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack`
    (`madmom-upstream/madmom/features/downbeats.py:84-85`): every call in
    this project's own tests is a single whole-clip call with `reset=True`
    (the default), which always takes the "(re-)init the buffer with
    `diff_frames` `inf`-valued rows" branch below -- the multi-call streaming
    continuation path (`reset=False`) is ported faithfully (via
    `BufferProcessor`) but not exercised by this project's own golden
    fixtures, see `processors.py`'s `BufferProcessor` docstring.
    """

    def __init__(self, diff_ratio=DIFF_RATIO, diff_frames=DIFF_FRAMES,
                 diff_max_bins=DIFF_MAX_BINS, positive_diffs=POSITIVE_DIFFS,
                 stack_diffs=None, **kwargs):
        # pylint: disable=unused-argument
        self.diff_ratio = diff_ratio
        self.diff_frames = diff_frames
        self.diff_max_bins = diff_max_bins
        self.positive_diffs = positive_diffs
        self.stack_diffs = stack_diffs
        # attributes needed for stateful processing -- do not init the
        # buffer here, since its size depends on the data (matches upstream)
        self._buffer = None

    def reset(self):
        """Reset the SpectrogramDifferenceProcessor's buffer."""
        self._buffer = None

    def process(self, data, reset=True, **kwargs):
        """Perform a temporal difference calculation on the given data.

        Returns a `SpectrogramDifference` (if `stack_diffs` is `None`) or
        the result of `stack_diffs((spectrogram, diff))` -- matching
        `SpectrogramDifferenceProcessor.process` (`spectrogram.py:1085-1139`).
        """
        args = dict(diff_ratio=self.diff_ratio, diff_frames=self.diff_frames,
                    diff_max_bins=self.diff_max_bins,
                    positive_diffs=self.positive_diffs)
        args.update(kwargs)
        if self.diff_frames is None:
            # Note: use diff_ratio from args, not self.diff_ratio
            self.diff_frames = _diff_frames(
                args["diff_ratio"], frame_size=data.stft.frames.frame_size,
                hop_size=data.stft.frames.hop_size, window=data.stft.window,
            )
        data_arr = np.asarray(data)
        if self._buffer is None or reset:
            # put diff_frames infs before the data (will be replaced by 0s)
            init = np.empty((self.diff_frames, data_arr.shape[1]))
            init[:] = np.inf
            buffered = np.insert(data_arr, 0, init, axis=0)
            self._buffer = BufferProcessor(init=buffered)
        else:
            buffered = self._buffer(data_arr)
        # compute difference based on the buffered data (reduce 1st dim);
        # wrap in a plain Spectrogram-shaped stand-in so SpectrogramDifference
        # can read .stft off of it (needed if diff_frames must be recomputed)
        buffered_spec = Spectrogram.__new__(Spectrogram)
        buffered_spec.data = buffered
        buffered_spec.stft = data.stft
        diff = SpectrogramDifference(buffered_spec, keep_dims=False, **args)
        diff_arr = np.asarray(diff)
        # set all inf-diffs to 0
        diff_arr[np.isinf(diff_arr)] = 0
        if self.stack_diffs is None:
            return diff
        # Note: don't use `data` directly (could be a str) -- use
        # diff.spectrogram (i.e. the converted data), sliced by diff_frames
        return self.stack_diffs(
            (np.asarray(diff.spectrogram)[self.diff_frames:], diff_arr)
        )


# ---------------------------------------------------------------------------
# SuperFluxProcessor -- Wave 4b addition (features/onsets.py's superflux)
# ---------------------------------------------------------------------------
class SuperFluxProcessor(SequentialProcessor):
    """Spectrogram processor with the default values suitable for the
    SuperFlux onset detection algorithm.

    Port of `madmom.audio.spectrogram.SuperFluxProcessor`
    (`spectrogram.py:1230-1262`): un-normalized `LogarithmicFilterbank`
    (24 bands/octave), max-filtered (`diff_max_bins=3`), positive-only
    temporal difference. Its own chain starts at the STFT stage (no
    `SignalProcessor`/`FramedSignalProcessor`) -- callers pass in anything
    `ShortTimeFourierTransformProcessor.process()` can turn into a
    `ShortTimeFourierTransform` (a raw file path included, via
    `FramedSignal`'s own `Signal(...)` auto-conversion fallback).
    """

    def __init__(self, **kwargs):
        # set the default values (can be overwritten if set)
        # we need an un-normalized LogarithmicFilterbank with 24 bands
        filterbank = kwargs.pop("filterbank", FILTERBANK)
        num_bands = kwargs.pop("num_bands", 24)
        norm_filters = kwargs.pop("norm_filters", False)
        # we want max filtered diffs
        diff_ratio = kwargs.pop("diff_ratio", 0.5)
        diff_max_bins = kwargs.pop("diff_max_bins", 3)
        positive_diffs = kwargs.pop("positive_diffs", True)
        # processing chain
        stft = ShortTimeFourierTransformProcessor(**kwargs)
        spec = SpectrogramProcessor(**kwargs)
        filt = FilteredSpectrogramProcessor(
            filterbank=filterbank, num_bands=num_bands,
            norm_filters=norm_filters, **kwargs)
        log = LogarithmicSpectrogramProcessor(**kwargs)
        diff = SpectrogramDifferenceProcessor(
            diff_ratio=diff_ratio, diff_max_bins=diff_max_bins,
            positive_diffs=positive_diffs, **kwargs)
        super().__init__((stft, spec, filt, log, diff))
