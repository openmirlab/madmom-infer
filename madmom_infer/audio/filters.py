"""Filterbank construction -- composition port of madmom.audio.filters, the
subset `FilteredSpectrogramProcessor` (spectrogram.py) actually needs: a
logarithmically-spaced triangular filterbank.

Wave 4d addition (classic, non-DNN chroma path -- `CLAUDE.md`'s 4.0 audit
correction): `hz2midi`, `midi2hz`, `semitone_frequencies`,
`PitchClassProfileFilterbank`/`HarmonicPitchClassProfileFilterbank` (feed
`audio/chroma.py`'s `PitchClassProfile`/`HarmonicPitchClassProfile`),
`SimpleChromaFilterbank` (verbatim port INCLUDING its unconditional
`raise NotImplementedError` -- see that class's own docstring, this is not
a gap this port introduces), and `SemitoneBandpassFilterbank` (feeds
`audio/spectrogram.py`'s `SemitoneBandpassSpectrogram`, in turn
`audio/chroma.py`'s `CLPChroma`). `PitchClassProfileFilterbank`/
`HarmonicPitchClassProfileFilterbank` build a plain `(num_bins, num_classes)`
matrix and hand it to this module's own composition `Filterbank.__init__`
(which casts to `FILTER_DTYPE`/float32 as its last step) -- same
cast-happens-in-the-base-class order as upstream's own `Filterbank.__new__`
(`filters.py:724-738`), so no float32-vs-float64 rounding-order divergence
(unlike `LogarithmicFilterbank`'s per-filter cast-then-normalize trap, which
does NOT apply here: neither PCP/HPCP class normalizes its filter weights).

Phase-1 scope: only `LogarithmicFilterbank` (used by all-in-one-infer's
`build_spec_processor()` via `FilteredSpectrogramProcessor(num_bands=12,
fmin=30, fmax=17000, norm_filters=True)`,
`all-in-one-fix/src/allin1_infer/spectrogram.py:27-40`) plus the free
functions it's built from (`log_frequencies`, `frequencies2bins`,
`bins2frequencies`) and a generic `Filterbank` base holding the resulting
matrix. Still NOT ported (no call site so far,
`madmom-upstream/madmom/audio/filters.py`): the Bark/rectangular/chroma/
pitch-class-profile/semitone-bandpass filterbank variants (`BarkFilterbank`,
`RectangularFilterbank`, `*ChromaFilterbank`, `PitchClassProfileFilterbank`,
`HarmonicFilterbank`, `SemitoneBandpassFilterbank`) and their frequency-scale
helpers (`hz2bark`/`bark2hz`/`bark_frequencies`, `hz2erb`/`erb2hz`,
`hz2midi`/`midi2hz`). `Filter`/`TriangularFilter` are simplified from
upstream's `np.ndarray`-subclass hierarchy (`filters.py:413-605`) down to
two plain functions (`_triangular_filter_band_bins`, `_triangular_filters`)
that build `(start, data)` tuples directly, since no caller needs a
standalone Filter object -- only the finished filterbank matrix.

Wave 4b addition: `hz2mel`, `mel2hz`, `mel_frequencies`, `MelFilterbank`.
The 4.0 audit table originally slotted `MelFilterbank` into wave 4g
(`audio/cepstrogram.py`'s MFCC), but flagged in the same breath that it also
feeds `CNNOnsetProcessor`'s 80-band mel input (`features/onsets.py`,
`FilteredSpectrogramProcessor(filterbank=MelFilterbank, num_bands=80,
fmin=27.5, fmax=16000, norm_filters=True, unique_filters=False)`) -- ported
here instead, pulled forward per that flag; CLAUDE.md's audit table row is
updated to PORTED (4b) accordingly, so 4g's cepstrogram work reuses this
instead of re-porting it. `MelFilterbank` reuses this module's own
`_triangular_filters`/`_place_filters_into_matrix` helpers (identical
triangular-filter-placement math to `LogarithmicFilterbank`, just a
different underlying frequency scale) -- verbatim port of
`MelFilterbank.__new__` (`madmom-upstream/madmom/audio/filters.py:
1076-1092`), not a re-derivation.

**Bit-identity trap, replicated on purpose**: upstream's `Filter.__new__`
casts each filter's samples to `FILTER_DTYPE` (`float32`) BEFORE normalizing
(`filters.py:440-452`: `obj = np.asarray(data, dtype=FILTER_DTYPE).view(cls)`
happens first, `obj /= np.sum(obj)` after) -- so `norm_filters=True`
normalization happens in **float32** precision, not float64. This port
replicates that exact order (cast-then-normalize) in `_triangular_filters`
below; normalizing in float64 first and casting down afterwards would
produce different (still "correct-looking", but NOT bit-identical) rounding
in the last mantissa bit of every non-trivial filter coefficient.

Wave 4f addition: `bark_frequencies`, `bark_double_frequencies`,
`BarkFilterbank`, `RectangularFilterbank` (+ the internal `_rectangular_filters`
helper, mirroring `_triangular_filters`'s shape for `RectangularFilter`).
`RectangularFilterbank` is the load-bearing one -- it feeds `audio/
spectrogram.py`'s `MultiBandSpectrogram`/`MultiBandSpectrogramProcessor`,
in turn `features/downbeats.py`'s `PatternTrackingProcessor`.
`BarkFilterbank` is ported for API completeness (real, public,
`pickletools`-confirmed NOT needed by any target this project ships) --
see that class's own section header for the confirmation.

Wave 4g addition: `HarmonicFilterbank` -- ported INCLUDING its unconditional
`raise NotImplementedError`, same not-actually-implemented shape as
`SimpleChromaFilterbank` above (confirmed by reading upstream directly, see
that class's own docstring). The audit table's lowest-priority row: no
processor in this project needs it.

Reads: numpy; read by: madmom_infer/audio/spectrogram.py
(`FilteredSpectrogramProcessor`, `MultiBandSpectrogram`).
"""

import numpy as np

FILTER_DTYPE = np.float32
A4 = 440.0

FMIN = 30.0
FMAX = 17000.0
NUM_BANDS = 12
NORM_FILTERS = True
UNIQUE_FILTERS = True


def log_frequencies(bands_per_octave, fmin, fmax, fref=A4):
    """Frequencies aligned on a logarithmic (semitone-like) scale.

    Port of `madmom.audio.filters.log_frequencies` (`filters.py:186-223`). If
    `bands_per_octave=12` and `fref=440` (this module's defaults), the
    frequencies are equivalent to MIDI notes.
    """
    left = np.floor(np.log2(float(fmin) / fref) * bands_per_octave)
    right = np.ceil(np.log2(float(fmax) / fref) * bands_per_octave)
    frequencies = fref * 2.0 ** (np.arange(left, right) / float(bands_per_octave))
    # filter frequencies: needed because floor/ceil above can widen the range
    frequencies = frequencies[np.searchsorted(frequencies, fmin):]
    frequencies = frequencies[:np.searchsorted(frequencies, fmax, "right")]
    return frequencies


def frequencies2bins(frequencies, bin_frequencies, unique_bins=False):
    """Map `frequencies` to the closest corresponding FFT `bin_frequencies`.

    Verbatim port of `madmom.audio.filters.frequencies2bins`
    (`filters.py:348-388`).
    """
    frequencies = np.asarray(frequencies)
    bin_frequencies = np.asarray(bin_frequencies)
    indices = bin_frequencies.searchsorted(frequencies)
    indices = np.clip(indices, 1, len(bin_frequencies) - 1)
    left = bin_frequencies[indices - 1]
    right = bin_frequencies[indices]
    indices = indices - (frequencies - left < right - frequencies)
    if unique_bins:
        indices = np.unique(indices)
    return indices


def bins2frequencies(bins, bin_frequencies):
    """Convert bins to their corresponding frequencies [Hz].

    Port of `madmom.audio.filters.bins2frequencies` (`filters.py:391-409`).
    """
    return np.asarray(bin_frequencies, dtype=float)[np.asarray(bins)]


def _triangular_filter_band_bins(bins, overlap=True):
    """Yield (start, center, stop) bin triples for triangular filters.

    Port of `TriangularFilter.band_bins` (`filters.py:557-605`); only the
    `overlap=True` path is exercised by `LogarithmicFilterbank`, but the
    `overlap=False` branch is kept for fidelity (cheap, and matches
    upstream's own generality).
    """
    if len(bins) < 3:
        raise ValueError("not enough bins to create a TriangularFilter")
    index = 0
    while index + 3 <= len(bins):
        start, center, stop = bins[index:index + 3]
        if not overlap:
            start = int(np.floor((center + start) / 2.0))
            stop = int(np.ceil((center + stop) / 2.0))
        if stop - start < 2:
            center = start
            stop = start + 1
        yield start, center, stop
        index += 1


def _triangular_filters(bins, norm, overlap=True):
    """Build `(start, data)` triangular filter tuples for each band.

    `data` is the filter's samples (length `stop - start`), already cast to
    `FILTER_DTYPE` and (if `norm`) normalized IN THAT DTYPE -- see module
    header's bit-identity trap. Port of `TriangularFilter.__new__` +
    `Filter.filters` (`filters.py:440-457, 475-502`).
    """
    filters = []
    for start, center, stop in _triangular_filter_band_bins(bins, overlap=overlap):
        if not start <= center < stop:
            raise ValueError("`center` must be between `start` and `stop`")
        start = int(start)
        center = int(center)
        stop = int(stop)
        rel_center = center - start
        rel_stop = stop - start
        data = np.zeros(rel_stop)
        # rising edge (without the center)
        data[:rel_center] = np.linspace(0, 1, rel_center, endpoint=False)
        # falling edge (including the center, but without the last bin)
        data[rel_center:] = np.linspace(1, 0, rel_stop - rel_center, endpoint=False)
        # cast to FILTER_DTYPE (float32) BEFORE normalizing -- replicates
        # Filter.__new__'s exact order, see module header
        data = np.asarray(data, dtype=FILTER_DTYPE)
        if norm:
            data = data / np.sum(data)
        filters.append((start, data))
    return filters


def _place_filters_into_matrix(filters, bin_frequencies):
    """Combine `(start, data)` filter tuples into one filterbank matrix.

    Port of `Filterbank.from_filters`/`_put_filter`
    (`filters.py:746-821`): each filter is placed at its `start` offset,
    truncated if it runs off either edge of the bin axis, and combined with
    whatever's already there via elementwise max (only matters if multiple
    filters share a band, which never happens for `LogarithmicFilterbank` --
    one triangular filter per band).
    """
    fb = np.zeros((len(bin_frequencies), len(filters)))
    for band_id, (start, data) in enumerate(filters):
        stop = start + len(data)
        if start < 0:
            data = data[-start:]
            start = 0
        if stop > fb.shape[0]:
            data = data[:-(stop - fb.shape[0])]
            stop = fb.shape[0]
        band = fb[:, band_id]
        target = band[start:stop]
        np.maximum(data, target, out=target)
    return fb


class Filterbank:
    """A filterbank: a `(num_bins, num_bands)` matrix plus bin frequencies.

    Composition port of `madmom.audio.filters.Filterbank`
    (`filters.py:692-873`): filtering a `(num_frames, num_bins)` spectrogram
    with `np.dot(spectrogram, filterbank)` yields a `(num_frames, num_bands)`
    result, matching `FilteredSpectrogram`'s use in `audio/spectrogram.py`.
    """

    def __init__(self, data, bin_frequencies):
        data = np.asarray(data)
        if data.ndim != 2:
            raise TypeError(
                "wrong input data for Filterbank, must be a 2D np.ndarray"
            )
        self.data = np.asarray(data, dtype=FILTER_DTYPE)
        if len(bin_frequencies) != self.data.shape[0]:
            raise ValueError(
                "`bin_frequencies` must have the same length as the first "
                "dimension of `data`."
            )
        self.bin_frequencies = np.asarray(bin_frequencies, dtype=float)

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
    def num_bins(self):
        """Number of bins."""
        return self.data.shape[0]

    @property
    def num_bands(self):
        """Number of bands."""
        return self.data.shape[1]

    @property
    def corner_frequencies(self):
        """Corner frequencies of the filter bands."""
        freqs = []
        for band in range(self.num_bands):
            bins = np.nonzero(self.data[:, band])[0]
            freqs.append([np.min(bins), np.max(bins)])
        return bins2frequencies(freqs, self.bin_frequencies)

    @property
    def center_frequencies(self):
        """Center frequencies of the filter bands."""
        freqs = []
        for band in range(self.num_bands):
            bins = np.nonzero(self.data[:, band])[0]
            min_bin = np.min(bins)
            max_bin = np.max(bins)
            if self.data[min_bin, band] == self.data[max_bin, band]:
                center = int(min_bin + (max_bin - min_bin) / 2.0)
            else:
                center = min_bin + np.argmax(self.data[min_bin:max_bin, band])
            freqs.append(center)
        return bins2frequencies(freqs, self.bin_frequencies)

    @property
    def fmin(self):
        """Minimum frequency of the filterbank."""
        return self.bin_frequencies[np.nonzero(self.data)[0][0]]

    @property
    def fmax(self):
        """Maximum frequency of the filterbank."""
        return self.bin_frequencies[np.nonzero(self.data)[0][-1]]


class LogarithmicFilterbank(Filterbank):
    """A filterbank with logarithmically-spaced, overlapping triangular
    filters (semitone spacing if `num_bands=12`).

    Composition port of `madmom.audio.filters.LogarithmicFilterbank`
    (`filters.py:1153-1235`). **`num_bands` is bands PER OCTAVE, not a total
    band count** (`bands_per_octave=True`, the only mode ported -- see
    tests/fixtures/README.md's "Surprises": `num_bands=12, fmin=30,
    fmax=17000` yields an 81-band filterbank, not a 12-band one).
    """

    NUM_BANDS_PER_OCTAVE = 12

    def __init__(self, bin_frequencies, num_bands=NUM_BANDS_PER_OCTAVE,
                 fmin=FMIN, fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                 unique_filters=UNIQUE_FILTERS, bands_per_octave=True):
        if not bands_per_octave:
            # iteratively determining an exact total band count is not on
            # the phase-1 chain (all-in-one-infer always uses the
            # bands-per-octave mode)
            raise NotImplementedError(
                "`num_bands` with `bands_per_octave=False` is not "
                "implemented for LogarithmicFilterbank (out of phase-1 "
                "scope, see module header)."
            )
        frequencies = log_frequencies(num_bands, fmin, fmax, fref)
        bins = frequencies2bins(frequencies, bin_frequencies,
                                 unique_bins=unique_filters)
        filters = _triangular_filters(bins, norm=norm_filters, overlap=True)
        matrix = _place_filters_into_matrix(filters, bin_frequencies)
        super().__init__(matrix, bin_frequencies)
        self.fref = fref
        self.num_bands_per_octave = num_bands


# alias, matching upstream's LogFilterbank = LogarithmicFilterbank
LogFilterbank = LogarithmicFilterbank


# ---------------------------------------------------------------------------
# Mel filterbank -- Wave 4b addition (CNNOnsetProcessor's 80-band mel input)
# ---------------------------------------------------------------------------
def hz2mel(f):
    """Convert Hz frequencies to Mel. Verbatim port of
    `madmom.audio.filters.hz2mel` (`filters.py:21-36`)."""
    return 1127.01048 * np.log(np.asarray(f) / 700.0 + 1.0)


def mel2hz(m):
    """Convert Mel frequencies to Hz. Verbatim port of
    `madmom.audio.filters.mel2hz` (`filters.py:39-54`)."""
    return 700.0 * (np.exp(np.asarray(m) / 1127.01048) - 1.0)


def mel_frequencies(num_bands, fmin, fmax):
    """Frequencies aligned on the Mel scale. Verbatim port of
    `madmom.audio.filters.mel_frequencies` (`filters.py:57-77`)."""
    return mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), num_bands))


class MelFilterbank(Filterbank):
    """A filterbank with triangular filters spaced on the Mel scale.

    Composition port of `madmom.audio.filters.MelFilterbank`
    (`filters.py:1035-1092`). Unlike `LogarithmicFilterbank`, `num_bands` is
    a TOTAL band count (not bands-per-octave) -- 2 extra edge bands are
    requested from `mel_frequencies` internally, matching upstream exactly.
    """

    NUM_BANDS = 40
    FMIN = 20.0
    FMAX = 17000.0
    NORM_FILTERS = True
    UNIQUE_FILTERS = True

    def __init__(self, bin_frequencies, num_bands=NUM_BANDS, fmin=FMIN,
                 fmax=FMAX, norm_filters=NORM_FILTERS,
                 unique_filters=UNIQUE_FILTERS, **kwargs):
        # pylint: disable=unused-argument
        frequencies = mel_frequencies(num_bands + 2, fmin, fmax)
        bins = frequencies2bins(frequencies, bin_frequencies,
                                 unique_bins=unique_filters)
        filters = _triangular_filters(bins, norm=norm_filters, overlap=True)
        matrix = _place_filters_into_matrix(filters, bin_frequencies)
        super().__init__(matrix, bin_frequencies)


# ---------------------------------------------------------------------------
# Chroma filterbanks -- Wave 4d addition (classic, non-DNN chroma path)
# ---------------------------------------------------------------------------
def hz2midi(f, fref=A4):
    """Convert Hz frequencies to (fractional) MIDI note numbers. Verbatim
    port of `madmom.audio.filters.hz2midi` (`filters.py:250-273`)."""
    return (12.0 * np.log2(np.asarray(f, dtype=float) / fref)) + 69.0


def midi2hz(m, fref=A4):
    """Convert (fractional) MIDI note numbers to Hz frequencies. Verbatim
    port of `madmom.audio.filters.midi2hz` (`filters.py:276-293`)."""
    return 2.0 ** ((np.asarray(m, dtype=float) - 69.0) / 12.0) * fref


def semitone_frequencies(fmin, fmax, fref=A4):
    """Frequencies separated by semitones. Verbatim port of
    `madmom.audio.filters.semitone_frequencies` (`filters.py:226-246`) --
    exactly `log_frequencies(12, fmin, fmax, fref)`."""
    return log_frequencies(12, fmin, fmax, fref)


class PitchClassProfileFilterbank(Filterbank):
    """Filterbank for extracting pitch class profiles (PCP): each FFT bin is
    assigned to exactly one of `num_classes` pitch classes (a hard,
    one-hot-per-bin assignment, no overlap/weighting between classes).

    Composition port of `madmom.audio.filters.PitchClassProfileFilterbank`
    (`filters.py:1382-1459`).

    References
    ----------
    .. [1] T. Fujishima, "Realtime chord recognition of musical sound: a
           system using Common Lisp Music", Proceedings of the
           International Computer Music Conference (ICMC), 1999.
    """

    CLASSES = 12
    FMIN = 100.0
    FMAX = 5000.0

    def __init__(self, bin_frequencies, num_classes=CLASSES, fmin=FMIN,
                 fmax=FMAX, fref=A4):
        # init a filterbank
        fb = np.zeros((len(bin_frequencies), num_classes))
        # use only positive bin frequencies
        pos_bin_frequencies = bin_frequencies > 0
        # log deviation from the reference frequency
        log_dev = np.log2(bin_frequencies[pos_bin_frequencies] / fref)
        # map the log deviation to the closest pitch class profiles
        num_class = np.round(num_classes * log_dev) % num_classes
        # define the filterbank, skip all bins which were 0
        fb[pos_bin_frequencies, num_class.astype(int)] = 1
        # set all bins outside the allowed frequency range to 0
        fb[np.searchsorted(bin_frequencies, fmax, "right"):] = 0
        fb[:np.searchsorted(bin_frequencies, fmin)] = 0
        super().__init__(fb, bin_frequencies)
        self.fref = fref


class HarmonicPitchClassProfileFilterbank(PitchClassProfileFilterbank):
    """Filterbank for extracting harmonic pitch class profiles (HPCP): each
    positive-frequency FFT bin contributes a raised-cosine WEIGHT to every
    pitch class within `window` semitones of it (a soft, overlapping
    assignment, unlike the plain PCP filterbank's hard one-hot mapping).

    Composition port of
    `madmom.audio.filters.HarmonicPitchClassProfileFilterbank`
    (`filters.py:1462-1543`).

    References
    ----------
    .. [1] Emilia Gomez, "Tonal Description of Music Audio Signals", PhD
           thesis, Universitat Pompeu Fabra, Barcelona, Spain, 2006.
    """

    CLASSES = 36
    FMIN = 100.0
    FMAX = 5000.0
    WINDOW = 4

    def __init__(self, bin_frequencies, num_classes=CLASSES, fmin=FMIN,
                 fmax=FMAX, fref=A4, window=WINDOW):
        # pylint: disable=super-init-not-called
        # init a filterbank (deliberately NOT calling
        # PitchClassProfileFilterbank.__init__ -- the weighting math differs
        # entirely, only the class hierarchy is shared, matching upstream's
        # own PitchClassProfileFilterbank subclassing, which likewise
        # overrides __new__ completely rather than reusing the base class's)
        fb = np.zeros((len(bin_frequencies), num_classes))
        # use only positive bin frequencies
        pos_bin_frequencies = np.nonzero(bin_frequencies > 0)[0]
        # log deviation from the reference frequency
        log_dev = np.log2(bin_frequencies[pos_bin_frequencies] / fref)
        # map the log deviation to pitch class profiles
        num_class = (num_classes * log_dev) % num_classes
        # weight the bins
        for c in range(num_classes):
            # calculate the distance of the bins to the current class
            distance = num_class - c
            # unwrap
            distance[distance < -num_classes / 2.0] += num_classes
            distance[distance > num_classes / 2.0] -= num_classes
            # get all bins which are within the defined window
            idx = np.abs(distance) < window / 2.0
            # apply the weighting function
            filt = np.cos((num_class[idx] - c) * np.pi / window) ** 2.0
            # map these indices to the positive bin frequencies
            fb[pos_bin_frequencies[idx], c] = filt
        # set all bins outside the allowed frequency range to 0
        fb[np.searchsorted(bin_frequencies, fmax, "right"):] = 0
        fb[:np.searchsorted(bin_frequencies, fmin)] = 0
        Filterbank.__init__(self, fb, bin_frequencies)
        self.fref = fref
        self.window = window


class SimpleChromaFilterbank(Filterbank):
    """A simple chroma filterbank based on a (semitone) filterbank.

    Verbatim port of `madmom.audio.filters.SimpleChromaFilterbank`
    (`filters.py:1301-1366`) -- **including its unconditional
    `raise NotImplementedError`**. This is not a gap this port introduces:
    upstream's own `__new__` raises immediately, before ANY of its own
    (dead, TODO-commented) construction code runs -- confirmed by reading
    `filters.py:1340-1341` directly: `raise NotImplementedError("please
    check if produces correct/expected results and enable if yes.")`, with
    the actual filterbank-building code below it unreachable. No shipped
    madmom processor this project ports ever constructs one (the 4.0 audit's
    TO-PORT listing was itself only tracking public API surface, not
    reachability), so this port reproduces upstream's own not-actually-
    implemented state faithfully rather than "fixing" it by finishing code
    upstream itself never enabled.
    """

    NUM_BANDS = 12

    def __init__(self, bin_frequencies, num_bands=NUM_BANDS, fmin=FMIN,
                 fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                 unique_filters=UNIQUE_FILTERS):
        # pylint: disable=unused-argument
        raise NotImplementedError(
            "please check if produces correct/expected results and enable "
            "if yes. (verbatim port of upstream's own unconditional raise, "
            "see this class's docstring -- SimpleChromaFilterbank is not "
            "actually implemented in real madmom either.)"
        )


class HarmonicFilterbank(Filterbank):
    """Harmonic filterbank class.

    Verbatim port of `madmom.audio.filters.HarmonicFilterbank`
    (`madmom-upstream/madmom/audio/filters.py:1369-1379`) -- **including its
    unconditional `raise NotImplementedError`**, the same not-actually-
    implemented shape as `SimpleChromaFilterbank` right above (see that
    class's docstring for the general pattern). Confirmed by reading
    upstream directly: `__new__` raises immediately with `'please implement
    if needed!'`, no filterbank-construction code follows it at all (unlike
    `SimpleChromaFilterbank`, which at least has dead TODO-commented code
    below its raise -- this one has nothing). Wave 4g, the audit table's own
    lowest-priority row: no processor in this project needs it (not even a
    speculative one), ported anyway for API-surface completeness.
    """

    def __init__(self):
        raise NotImplementedError(
            "please implement if needed! (verbatim port of upstream's own "
            "unconditional raise, see this class's docstring -- "
            "HarmonicFilterbank is not actually implemented in real madmom "
            "either.)"
        )


class SemitoneBandpassFilterbank:
    """Time-domain semitone filterbank of elliptic (IIR) bandpass filters, as
    proposed in [1]_.

    Composition port of `madmom.audio.filters.SemitoneBandpassFilterbank`
    (`filters.py:1545-1600`) -- NOT a `Filterbank` subclass (matching
    upstream: this is a `scipy.signal.filtfilt`-driven time-domain
    filterbank, incompatible with the `np.dot(spectrogram, filterbank)`
    time-frequency-domain filtering every OTHER filterbank in this module
    implements -- see that class's own "Notes" section upstream).

    Feeds `audio/spectrogram.py`'s `SemitoneBandpassSpectrogram`, in turn
    `audio/chroma.py`'s `CLPChroma`. Each semitone band is filtered at ONE of
    3 fixed sample rates (882, 4410, or 22050 Hz, chosen by frequency range)
    -- `SemitoneBandpassSpectrogram` is the caller that actually resamples
    the input signal to match (`madmom_infer.audio.signal.resample`, Wave
    4d's ffmpeg-subprocess addition -- see that function's docstring for why
    this project's long-standing "no ffmpeg dependency" stance had to be
    revisited for this one, genuinely unavoidable, dependency).

    Parameters
    ----------
    order : int, optional
        Order of elliptic filters.
    passband_ripple : float, optional
        Maximum ripple allowed below unity gain in the passband [dB].
    stopband_rejection : float, optional
        Minimum attenuation required in the stop band [dB].
    q_factor : int, optional
        Q-factor of the filters.
    fmin : float, optional
        Minimum frequency of the filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the filterbank [Hz].
    fref : float, optional
        Reference frequency for the first bandpass filter [Hz].

    References
    ----------
    .. [1] Meinard Mueller, "Information retrieval for music and motion",
           Springer, 2007.
    """

    def __init__(self, order=4, passband_ripple=1, stopband_rejection=50,
                 q_factor=25, fmin=27.5, fmax=4200.0, fref=A4):
        from scipy.signal import ellip

        self.order = order
        self.passband_ripple = passband_ripple
        self.stopband_rejection = stopband_rejection
        self.q_factor = q_factor
        self.fref = fref
        self.center_frequencies = semitone_frequencies(fmin, fmax, fref=fref)
        # use different sample rates for the individual bands
        self.band_sample_rates = (
            np.ones_like(self.center_frequencies) * 4410
        )
        self.band_sample_rates[self.center_frequencies > 2000] = 22050
        self.band_sample_rates[self.center_frequencies < 250] = 882
        self.filters = []
        for freq, sample_rate in zip(self.center_frequencies,
                                     self.band_sample_rates):
            freqs = [(freq - freq / q_factor / 2.0) * 2.0 / sample_rate,
                     (freq + freq / q_factor / 2.0) * 2.0 / sample_rate]
            self.filters.append(ellip(order, passband_ripple,
                                      stopband_rejection, freqs,
                                      btype="bandpass"))


# ---------------------------------------------------------------------------
# Bark / rectangular filterbanks -- Wave 4f addition (audit table: feeds
# audio/spectrogram.py's MultiBandSpectrogram/MultiBandSpectrogramProcessor,
# in turn features/downbeats.py's PatternTrackingProcessor).
#
# **Finding, confirmed by reading upstream directly, not assumed**: only
# `RectangularFilterbank` is actually load-bearing for `MultiBandSpectrogram`
# (`madmom-upstream/madmom/audio/spectrogram.py:1310`, `from .filters import
# RectangularFilterbank`) -- `BarkFilterbank` is NOT referenced by
# `MultiBandSpectrogram`/`MultiBandSpectrogramProcessor`, or by anything else
# this port ships (no shipped model or processor constructs one). The 4.0
# audit table's row groups `BarkFilterbank`/`RectangularFilter`/
# `RectangularFilterbank` together under "feeds MultiBandSpectrogramProcessor"
# -- ported here anyway (real, public, cheap, self-contained -- same "port
# the real surface even if unreachable from a target processor" precedent as
# `NOTES_CNN_MIREX`/`ONSETS_BRNN_PP`'s registry-less-but-real model families).
# ---------------------------------------------------------------------------
def bark_frequencies(fmin=20.0, fmax=15500.0):
    """Frequencies aligned on the (normal) Bark scale, clipped to
    `[fmin, fmax]`.

    Verbatim port of `madmom.audio.filters.bark_frequencies`
    (`filters.py:124-149`).
    """
    frequencies = np.array([
        20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720,
        2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000,
        15500,
    ])
    frequencies = frequencies[np.searchsorted(frequencies, fmin):]
    frequencies = frequencies[:np.searchsorted(frequencies, fmax, "right")]
    return frequencies


def bark_double_frequencies(fmin=20.0, fmax=15500.0):
    """Frequencies aligned on the Bark scale, INCLUDING center frequencies
    between the corner frequencies (`num_bands='double'`), clipped to
    `[fmin, fmax]`.

    Verbatim port of `madmom.audio.filters.bark_double_frequencies`
    (`filters.py:152-182`).
    """
    frequencies = np.array([
        20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 510, 570, 630, 700,
        770, 840, 920, 1000, 1080, 1170, 1270, 1370, 1480, 1600, 1720, 1850,
        2000, 2150, 2320, 2500, 2700, 2900, 3150, 3400, 3700, 4000, 4400,
        4800, 5300, 5800, 6400, 7000, 7700, 8500, 9500, 10500, 12000, 13500,
        15500,
    ])
    frequencies = frequencies[np.searchsorted(frequencies, fmin):]
    frequencies = frequencies[:np.searchsorted(frequencies, fmax, "right")]
    return frequencies


def _rectangular_filters(bins, norm):
    """Build `(start, data)` non-overlapping rectangular filter tuples, one
    per adjacent pair of `bins`.

    Port of `RectangularFilter.__new__` + `RectangularFilter.band_bins`
    (`filters.py:608-679`) -- only the `overlap=False` path (the only mode
    `BarkFilterbank` ever requests; `band_bins`'s `overlap=True` branch is an
    unconditional `raise NotImplementedError` upstream, not ported since
    nothing calls it). `data` is a run of `1`s (length `stop - start`),
    already cast to `FILTER_DTYPE` and (if `norm`) normalized IN THAT DTYPE
    -- same cast-then-normalize order as `_triangular_filters`, see this
    module's header's "bit-identity trap" note (`Filter.__new__` casts
    before normalizing for every `Filter` subclass, including this one).
    """
    if len(bins) < 2:
        raise ValueError("not enough bins to create a RectangularFilter")
    filters = []
    for index in range(len(bins) - 1):
        start, stop = int(bins[index]), int(bins[index + 1])
        if start >= stop:
            raise ValueError("`start` must be smaller than `stop`")
        data = np.ones(stop - start, dtype=float)
        data = np.asarray(data, dtype=FILTER_DTYPE)
        if norm:
            data = data / np.sum(data)
        filters.append((start, data))
    return filters


class BarkFilterbank(Filterbank):
    """A filterbank with non-overlapping rectangular filters spaced on the
    Bark scale.

    Composition port of `madmom.audio.filters.BarkFilterbank`
    (`filters.py:1095-1150`). **Not consumed by any processor this project
    ports** -- see this section's header note.
    """

    NUM_BANDS = "normal"
    FMIN = 20.0
    FMAX = 15500.0

    def __init__(self, bin_frequencies, num_bands=NUM_BANDS, fmin=FMIN,
                 fmax=FMAX, norm_filters=NORM_FILTERS,
                 unique_filters=UNIQUE_FILTERS, **kwargs):
        # pylint: disable=unused-argument
        if num_bands == "normal":
            frequencies = bark_frequencies(fmin, fmax)
        elif num_bands == "double":
            frequencies = bark_double_frequencies(fmin, fmax)
        else:
            raise ValueError("`num_bands` must be {'normal', 'double'}")
        # Note: BarkFilterbank inverts the usual unique_bins/unique_filters
        # relationship (`unique_bins=not unique_filters`), matching upstream
        # (`filters.py:1144-1145`) exactly -- not a typo carried over.
        bins = frequencies2bins(frequencies, bin_frequencies,
                                 unique_bins=not unique_filters)
        filters = _rectangular_filters(bins, norm=norm_filters)
        matrix = _place_filters_into_matrix(filters, bin_frequencies)
        super().__init__(matrix, bin_frequencies)


class RectangularFilterbank(Filterbank):
    """A filterbank of contiguous, non-overlapping rectangular bands split
    at given crossover frequencies.

    Composition port of `madmom.audio.filters.RectangularFilterbank`
    (`filters.py:1240-1297`) -- direct matrix construction (each band is a
    contiguous run of bins set to `1`, optionally normalized), NOT built
    from individual `Filter` objects the way `BarkFilterbank`/
    `LogarithmicFilterbank`/`MelFilterbank` are (upstream's own `__new__`
    fills the matrix directly too -- no `RectangularFilter`/`from_filters`
    involved here despite the similar name). Feeds `audio/spectrogram.py`'s
    `MultiBandSpectrogram`.
    """

    def __init__(self, bin_frequencies, crossover_frequencies, fmin=FMIN,
                 fmax=FMAX, norm_filters=NORM_FILTERS,
                 unique_filters=UNIQUE_FILTERS, **kwargs):
        # pylint: disable=unused-argument
        fb = np.zeros((len(bin_frequencies), len(crossover_frequencies) + 1),
                      dtype=FILTER_DTYPE)
        corner_frequencies = np.r_[fmin, crossover_frequencies, fmax]
        corner_bins = frequencies2bins(corner_frequencies, bin_frequencies,
                                        unique_bins=unique_filters)
        for i in range(len(corner_bins) - 1):
            fb[corner_bins[i]:corner_bins[i + 1], i] = 1
        if norm_filters:
            # if the sum over a band is zero, do not normalize this band
            band_sum = np.sum(fb, axis=0)
            band_sum[band_sum == 0] = 1
            fb /= band_sum
        super().__init__(fb, bin_frequencies)
        self.crossover_frequencies = bins2frequencies(
            corner_bins[1:-1], bin_frequencies)
