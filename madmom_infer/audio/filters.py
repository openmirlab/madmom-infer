"""Filterbank construction -- composition port of madmom.audio.filters, the
subset `FilteredSpectrogramProcessor` (spectrogram.py) actually needs: a
logarithmically-spaced triangular filterbank.

Phase-1 scope: only `LogarithmicFilterbank` (used by all-in-one-infer's
`build_spec_processor()` via `FilteredSpectrogramProcessor(num_bands=12,
fmin=30, fmax=17000, norm_filters=True)`,
`all-in-one-fix/src/allin1_infer/spectrogram.py:27-40`) plus the free
functions it's built from (`log_frequencies`, `frequencies2bins`,
`bins2frequencies`) and a generic `Filterbank` base holding the resulting
matrix. Deliberately NOT ported (no phase-1 call site,
`madmom-upstream/madmom/audio/filters.py`): the Mel/Bark/rectangular/chroma/
pitch-class-profile/semitone-bandpass filterbank variants (`MelFilterbank`,
`BarkFilterbank`, `RectangularFilterbank`, `*ChromaFilterbank`,
`PitchClassProfileFilterbank`, `HarmonicFilterbank`,
`SemitoneBandpassFilterbank`) and their frequency-scale helpers
(`hz2mel`/`mel2hz`/`mel_frequencies`, `hz2bark`/`bark2hz`/`bark_frequencies`,
`hz2erb`/`erb2hz`, `hz2midi`/`midi2hz`) -- none of these are on the
phase-1 chain. `Filter`/`TriangularFilter` are simplified from upstream's
`np.ndarray`-subclass hierarchy (`filters.py:413-605`) down to two plain
functions (`_triangular_filter_band_bins`, `_triangular_filters`) that
build `(start, data)` tuples directly, since no phase-1 caller needs a
standalone Filter object -- only the finished filterbank matrix.

**Bit-identity trap, replicated on purpose**: upstream's `Filter.__new__`
casts each filter's samples to `FILTER_DTYPE` (`float32`) BEFORE normalizing
(`filters.py:440-452`: `obj = np.asarray(data, dtype=FILTER_DTYPE).view(cls)`
happens first, `obj /= np.sum(obj)` after) -- so `norm_filters=True`
normalization happens in **float32** precision, not float64. This port
replicates that exact order (cast-then-normalize) in `_triangular_filters`
below; normalizing in float64 first and casting down afterwards would
produce different (still "correct-looking", but NOT bit-identical) rounding
in the last mantissa bit of every non-trivial filter coefficient.

Reads: numpy; read by: madmom_infer/audio/spectrogram.py (planned,
`FilteredSpectrogramProcessor`).
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
