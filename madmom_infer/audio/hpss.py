"""Harmonic/percussive source separation -- port of madmom.audio.hpss.

`HarmonicPercussiveSourceSeparation` (alias `HPSS`) separates a magnitude
spectrogram into harmonic and percussive components via median filtering
(Fitzgerald 2010): a long filter along the time axis smooths out transients
to isolate sustained/harmonic content, a long filter along the frequency
axis does the opposite for percussive/transient content, and the two
filtered "slices" are compared (or soft-blended) into masks applied back to
the original spectrogram. Pure `scipy.ndimage.median_filter`, no BLAS, no
model weights -- a standalone preprocessing utility not consumed by any
other TO-PORT/PORTED processor in this project (confirmed by grepping
`../madmom-upstream/madmom/{audio,features,ml}/*` for `hpss`/`HPSS`
imports: none), the audit table's own framing for this Wave-4g target.

`process()` accepts any 2-D spectrogram-like input and composes the public
`slices()` and `masks()` helpers. Both binary and soft masks are supported.

Reads: scipy.ndimage.median_filter, madmom_infer.processors.Processor,
madmom_infer.audio.spectrogram.Spectrogram (isinstance check only); read
by: nothing else in this project (standalone utility, see above).
"""

import numpy as np

from ..processors import Processor


class HarmonicPercussiveSourceSeparation(Processor):
    """Separates a magnitude spectrogram into harmonic and percussive
    components via median filtering.

    Port of `madmom.audio.hpss.HarmonicPercussiveSourceSeparation`
    (`madmom-upstream/madmom/audio/hpss.py:20-192`), minus `add_arguments`
    (argparse plumbing, out of scope per this project's established
    precedent -- see `audio/spectrogram.py`'s header).

    Parameters
    ----------
    masking : float or str
        Either the literal `'binary'` (or `None`, equivalent) for a binary
        mask, or any float coefficient for a soft mask.
    harmonic_filter : tuple of ints
        Harmonic median-filter size `(frames, bins)`.
    percussive_filter : tuple of ints
        Percussive median-filter size `(frames, bins)`.

    References
    ----------
    .. [1] Derry FitzGerald, "Harmonic/percussive separation using median
           filtering.", Proceedings of the 13th International Conference on
           Digital Audio Effects (DAFx), Graz, Austria, 2010.
    """

    MASKING = "binary"
    HARMONIC_FILTER = (15, 1)
    PERCUSSIVE_FILTER = (1, 15)

    def __init__(self, masking=MASKING, harmonic_filter=HARMONIC_FILTER,
                 percussive_filter=PERCUSSIVE_FILTER):
        self.masking = masking
        self.harmonic_filter = np.asarray(harmonic_filter, dtype=int)
        self.percussive_filter = np.asarray(percussive_filter, dtype=int)

    def slices(self, data):
        """Return the `(harmonic_slice, percussive_slice)` median-filtered
        slices of `data` (usually a magnitude spectrogram).

        Port of `HarmonicPercussiveSourceSeparation.slices`
        (`hpss.py:55-77`).
        """
        from scipy.ndimage import median_filter

        harmonic_slice = median_filter(data, self.harmonic_filter)
        percussive_slice = median_filter(data, self.percussive_filter)
        return harmonic_slice, percussive_slice

    def masks(self, harmonic_slice, percussive_slice):
        """Return the `(harmonic_mask, percussive_mask)` given the harmonic
        and percussive slices.

        Port of `HarmonicPercussiveSourceSeparation.masks`
        (`hpss.py:79-112`). Binary masks (`masking in (None, 'binary')`)
        compare the two slices directly; otherwise a soft mask is computed
        via `slice ** p / (harmonic_slice ** p + percussive_slice ** p)`.
        """
        if self.masking in (None, "binary"):
            harmonic_mask = harmonic_slice > percussive_slice
            percussive_mask = percussive_slice >= harmonic_slice
        else:
            p = float(self.masking)
            harmonic_slice_ = harmonic_slice ** p
            percussive_slice_ = percussive_slice ** p
            slice_sum_ = harmonic_slice_ + percussive_slice_
            harmonic_mask = harmonic_slice_ / slice_sum_
            percussive_mask = percussive_slice_ / slice_sum_
        return harmonic_mask, percussive_mask

    def process(self, data, **kwargs):
        """Return the `(harmonic, percussive)` components of `data`.

        The input must be a 2-D spectrogram-like object.
        """
        spectrogram = np.asarray(data)
        if spectrogram.ndim != 2:
            raise ValueError("HPSS input must be a 2-D spectrogram")
        slices = self.slices(spectrogram)
        harmonic_mask, percussive_mask = self.masks(*slices)
        harmonic = spectrogram * harmonic_mask
        percussive = spectrogram * percussive_mask
        return harmonic, percussive


# alias
HPSS = HarmonicPercussiveSourceSeparation
