"""Magnitude/filtered/log spectrograms -- composition port of
madmom.audio.spectrogram, the stage that turns a `ShortTimeFourierTransform`
into the filtered, log-compressed feature all-in-one-infer's models consume.

Phase-1 scope: `Spectrogram` (magnitude of the STFT), `FilteredSpectrogram` /
`FilteredSpectrogramProcessor` (apply a `filters.Filterbank`), and
`LogarithmicSpectrogram` / `LogarithmicSpectrogramProcessor` (log-compress) --
exactly the stages `all-in-one-infer`'s `build_spec_processor()` composes via
`SequentialProcessor([frames, stft, filt, spec])`
(`all-in-one-fix/src/allin1_infer/spectrogram.py:27-40`). Deliberately NOT
ported (no phase-1 call site, `madmom-upstream/madmom/audio/spectrogram.py`):
`SpectrogramDifference`/`SpectrogramDifferenceProcessor` (SuperFlux-style
onset-detection differencing), `SuperFluxProcessor`, `MultiBandSpectrogram`/
`MultiBandSpectrogramProcessor`, `SemitoneBandpassSpectrogram`,
`LogarithmicFilteredSpectrogram`/`LogarithmicFilteredSpectrogramProcessor`
(the fused filter+log convenience class -- phase-1 composes the same
pipeline via two separate stages instead, exactly like `build_spec_processor()`
already does), and `tuning_frequency()`/`Spectrogram.tuning_frequency()`.

Composition, not `np.ndarray` subclassing (docs/DESIGN.md C.2): each class
wraps a plain array in `.data`, and `FilteredSpectrogram`/
`LogarithmicSpectrogram` are ordinary Python subclasses of `Spectrogram`
(inheriting its array-interop dunders/properties for free) rather than
`np.ndarray` subclasses relying on `__array_finalize__`.

Reads: madmom_infer/audio/stft.py (ShortTimeFourierTransform),
madmom_infer/audio/filters.py (Filterbank, LogarithmicFilterbank),
madmom_infer/processors.py (Processor); read by: nothing yet within this
package (top-level `build_spec_processor()`-equivalent composition is
all-in-one-infer's own responsibility).
"""

import inspect

import numpy as np

from ..processors import Processor
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
from .stft import ShortTimeFourierTransform

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


class SpectrogramProcessor(Processor):
    """Processor wrapper: compute the magnitude `Spectrogram` of an STFT.

    Port of `madmom.audio.spectrogram.SpectrogramProcessor`
    (`spectrogram.py:239-264`). Not on all-in-one-infer's own chain (which
    goes straight from STFT to `FilteredSpectrogramProcessor`, which builds
    its own internal `Spectrogram`), kept for API completeness/composability.
    """

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
