"""Cepstrogram/MFCC -- composition port of madmom.audio.cepstrogram.

`Cepstrogram` applies a transform (a DCT by default) to a `Spectrogram`;
`MFCC` composes the standard Mel-Frequency Cepstral Coefficient pipeline on
top of it: filter through a Mel filterbank (`audio/filters.py`'s
`MelFilterbank`, pulled forward into Wave 4b for `CNNOnsetProcessor`'s
sake -- this module is the reason that pull-forward exists, see
`MelFilterbank`'s own docstring), log-compress, then transform.

Composition, not `np.ndarray` subclassing (docs/DESIGN.md C.2): unlike
upstream, where `Cepstrogram(np.ndarray)` and `MFCC(Cepstrogram)` are BOTH
ndarray-view subclasses (`madmom-upstream/madmom/audio/cepstrogram.py:
20-240`), this port makes `Cepstrogram` its own composition class -- a peer
of `audio/spectrogram.py`'s `Spectrogram`, not a subclass of it (upstream's
`Cepstrogram` isn't a `Spectrogram` subclass either: it directly subclasses
`np.ndarray`, sharing none of `Spectrogram`'s properties, only taking one as
a constructor argument) -- implementing the same `__array__`/`__len__`/
`__getitem__`/`shape`/`dtype`/`ndim` interop dunders `Signal`/`Spectrogram`
already establish. `MFCC` IS a real subclass of `Cepstrogram` (matching
upstream's own `class MFCC(Cepstrogram)` exactly).

**`.bin_frequencies` is genuinely `None` for both classes, not a gap** --
confirmed by reading upstream directly: `Cepstrogram.__new__`
(`cepstrogram.py:44-60`) has a `# TODO: what are the frequencies of the
bins?` comment with the actual assignment (`obj.bin_frequencies = ???`)
commented out and dead; only `__array_finalize__` (a numpy-ndarray-view
mechanism this port has no equivalent for, since it isn't ndarray-based)
ever sets the attribute, and only by copying from a source view that itself
never got a real value. So a freshly-constructed `Cepstrogram`/`MFCC` in
real madmom has `.bin_frequencies is None` too -- this port sets it
explicitly rather than omitting the attribute, to make that `None`-ness
introspectable rather than an `AttributeError` trap.

`MFCC` accepts raw audio, plain spectrograms, and already-filtered
spectrograms. Optional filtering/scaling attributes are inspected
defensively, and `MFCCProcessor` forwards every stored option, including a
custom transform.

Reads: scipy.fftpack.dct, numpy, madmom_infer.audio.filters (Filterbank,
MelFilterbank), madmom_infer.audio.spectrogram.Spectrogram,
madmom_infer.processors.Processor; read by: nothing else in this project
(standalone feature-extraction utility, same "no other TO-PORT processor
depends on this" status as `audio/hpss.py`).
"""

import warnings

import numpy as np
from scipy.fftpack import dct

from ..processors import Processor
from .filters import Filterbank, MelFilterbank
from .spectrogram import Spectrogram


class Cepstrogram:
    """A transformed `Spectrogram` (usually a DCT).

    Composition port of `madmom.audio.cepstrogram.Cepstrogram`
    (`madmom-upstream/madmom/audio/cepstrogram.py:20-78`) -- see this
    module's header for why this is a standalone composition class, not a
    `Spectrogram` subclass.

    Parameters
    ----------
    spectrogram : Spectrogram instance
        Spectrogram.
    transform : numpy ufunc, optional
        Transformation applied to `spectrogram` (default: `scipy.fftpack.
        dct`).
    kwargs : dict, optional
        If no `Spectrogram` instance was given, one is instantiated with
        these additional keyword arguments.
    """

    def __init__(self, spectrogram, transform=dct, **kwargs):
        if not isinstance(spectrogram, Spectrogram):
            spectrogram = Spectrogram(spectrogram, **kwargs)
        self.data = transform(np.asarray(spectrogram))
        self.spectrogram = spectrogram
        self.bin_frequencies = None
        self.transform = transform

    # -- numpy interop, mirroring audio/spectrogram.py's Spectrogram ----
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


class CepstrogramProcessor(Processor):
    """Processor wrapper: compute the `Cepstrogram` of a spectrogram.

    Port of `madmom.audio.cepstrogram.CepstrogramProcessor`
    (`cepstrogram.py:81-111`).
    """

    def __init__(self, transform=dct, **kwargs):
        # pylint: disable=unused-argument
        self.transform = transform

    def process(self, data, **kwargs):
        return Cepstrogram(data, transform=self.transform)


MFCC_BANDS = 30
MFCC_FMIN = 40.0
MFCC_FMAX = 15000.0
MFCC_NORM_FILTERS = True
MFCC_MUL = 1.0
MFCC_ADD = 0.0


class MFCC(Cepstrogram):
    """Mel-Frequency Cepstral Coefficients (MFCC) of a `Spectrogram`.

    Port of `madmom.audio.cepstrogram.MFCC` (`cepstrogram.py:122-240`).

    Parameters
    ----------
    spectrogram : Spectrogram instance
        Spectrogram.
    transform : numpy ufunc, optional
        Transformation applied to the filtered/log-compressed spectrogram.
    filterbank : Filterbank class or instance, optional
        Filterbank used to filter `spectrogram`; if a class, one is built
        with `num_bands`/`fmin`/`fmax`/`norm_filters` (default:
        `audio/filters.py`'s `MelFilterbank`).
    num_bands : int, optional
        Number of filter bands.
    fmin : float, optional
        Minimum frequency of the filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the filterbank [Hz].
    norm_filters : bool, optional
        Normalize the filters to area 1.
    mul : float, optional
        Multiply the magnitude spectrogram by this factor before taking the
        logarithm.
    add : float, optional
        Add this value before taking the logarithm of the magnitudes.
    kwargs : dict, optional
        If no `Spectrogram` instance was given, one is instantiated with
        these additional keyword arguments.

    From https://en.wikipedia.org/wiki/Mel-frequency_cepstrum, MFCCs are
    commonly derived as: (1) FFT of a windowed signal excerpt, (2) map
    powers onto the mel scale via triangular overlapping windows, (3) take
    the log of the mel powers, (4) DCT of the log-mel-power list, (5) the
    MFCCs are the amplitudes of the resulting spectrum.
    """

    def __init__(self, spectrogram, transform=dct, filterbank=MelFilterbank,
                 num_bands=MFCC_BANDS, fmin=MFCC_FMIN, fmax=MFCC_FMAX,
                 norm_filters=MFCC_NORM_FILTERS, mul=MFCC_MUL, add=MFCC_ADD,
                 **kwargs):
        if not isinstance(spectrogram, Spectrogram):
            spectrogram = Spectrogram(spectrogram, **kwargs)

        if (getattr(spectrogram, "filterbank", None) is not None
                or getattr(spectrogram, "mul", None) is not None
                or getattr(spectrogram, "add", None) is not None):
            warnings.warn(
                "Spectrogram was filtered or scaled already, redo "
                "calculation!"
            )
            spectrogram = Spectrogram(spectrogram.stft)

        if isinstance(filterbank, type) and issubclass(filterbank, Filterbank):
            # `duplicate_filters=False` is forwarded verbatim, matching
            # upstream (`cepstrogram.py:206-212`) -- `MelFilterbank`'s own
            # `**kwargs` catch-all silently absorbs it (its real knob is
            # `unique_filters`, defaulting to `True`), a faithfully
            # reproduced upstream no-op, not a bug this port introduces.
            filterbank = filterbank(
                spectrogram.bin_frequencies, num_bands=num_bands, fmin=fmin,
                fmax=fmax, norm_filters=norm_filters, duplicate_filters=False,
            )
        if not isinstance(filterbank, Filterbank):
            raise ValueError(
                "not a Filterbank type or instance: %s" % filterbank
            )

        data = np.dot(np.asarray(spectrogram), np.asarray(filterbank))
        np.log10(mul * data + add, out=data)
        data = transform(data)

        self.data = data
        self.transform = transform
        self.spectrogram = spectrogram
        self.filterbank = filterbank
        self.mul = mul
        self.add = add
        self.bin_frequencies = None


class MFCCProcessor(Processor):
    """Processor wrapper: filter a magnitude spectrogram through a Mel
    filterbank, log-compress it, then DCT-transform the result into MFCCs.

    Modernized port of `madmom.audio.cepstrogram.MFCCProcessor`
    (`cepstrogram.py:243-298`); all stored options are forwarded to `MFCC`.

    Parameters
    ----------
    num_bands : int, optional
        Number of Mel filter bands.
    fmin : float, optional
        Minimum frequency of the Mel filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the Mel filterbank [Hz].
    norm_filters : bool, optional
        Normalize the filters to area 1.
    mul : float, optional
        Multiply the magnitude spectrogram by this factor before taking the
        logarithm.
    add : float, optional
        Add this value before taking the logarithm of the magnitudes.
    transform : numpy ufunc, optional
        Transformation applied by `process()`.
    """

    def __init__(self, num_bands=MFCC_BANDS, fmin=MFCC_FMIN, fmax=MFCC_FMAX,
                 norm_filters=MFCC_NORM_FILTERS, mul=MFCC_MUL, add=MFCC_ADD,
                 transform=dct, **kwargs):
        # pylint: disable=unused-argument
        self.num_bands = num_bands
        self.fmin = fmin
        self.fmax = fmax
        self.norm_filters = norm_filters
        self.mul = mul
        self.add = add
        self.transform = transform

    def process(self, data, **kwargs):
        """Return the `MFCC` of `data` (usually a spectrogram).

        The stored transform and MFCC options are applied to every call.
        """
        return MFCC(data, transform=self.transform,
                    num_bands=self.num_bands, fmin=self.fmin,
                    fmax=self.fmax, norm_filters=self.norm_filters,
                    mul=self.mul, add=self.add)
