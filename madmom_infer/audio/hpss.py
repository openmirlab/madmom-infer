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

**Faithful bug-for-bug port, confirmed empirically, not assumed**:
`process()` is genuinely broken in real madmom 0.17.dev0 for EVERY input.
Reading `hpss.py:114-144` directly: `spectrogram` is only assigned inside
`if isinstance(data, Spectrogram): spectrogram = data.spec` -- but
`Spectrogram` (this project's `audio.spectrogram.Spectrogram`, matching
upstream's own class) has no `.spec` attribute at all (verified by
`hasattr()` against a real `Spectrogram` instance built through the
reference venv's own STFT->Spectrogram chain). So a `Spectrogram` input
raises `AttributeError` on `data.spec`; any OTHER input type never enters
the `if` branch at all, leaving `spectrogram` referenced-before-assignment
(`UnboundLocalError`/`NameError`). Confirmed directly against the reference
venv (`madmom-reference/.venv`): `HarmonicPercussiveSourceSeparation().
process(spec)` raises `AttributeError: 'Spectrogram' object has no
attribute 'spec'`; `.process(np.asarray(spec))` raises `UnboundLocalError:
local variable 'spectrogram' referenced before assignment`. This port
reproduces both failure modes exactly (pinned by `tests/test_hpss.py`'s
`pytest.raises` tests) rather than silently "fixing" what looks like a
typo for `data.data`/`np.asarray(data)` -- same bug-for-bug precedent as
`features/onsets.py`'s `correlation_diff`.

`slices()`/`masks()` (the two helper methods `process()` was presumably
meant to compose) work correctly and are the actually-usable surface --
verified against the reference venv on a real `mono_44100.wav` spectrogram,
both the binary-mask default and a float `masking` coefficient (soft mask).

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
    precedent -- see `audio/spectrogram.py`'s header). See this module's
    header for why `process()` is a faithful (broken) bug-for-bug port.

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

        Faithful (broken) port of
        `HarmonicPercussiveSourceSeparation.process` (`hpss.py:114-144`) --
        see this module's header for the confirmed, unconditional
        `AttributeError`/`UnboundLocalError` this raises for every input.
        """
        from .spectrogram import Spectrogram

        # matches upstream exactly: `spectrogram` is only assigned inside
        # this `if` branch, and `Spectrogram` has no `.spec` attribute --
        # both failure modes are load-bearing, not accidental.
        if isinstance(data, Spectrogram):
            spectrogram = data.spec
        slices = self.slices(spectrogram)
        harmonic_mask, percussive_mask = self.masks(*slices)
        harmonic = spectrogram * harmonic_mask
        percussive = spectrogram * percussive_mask
        return harmonic, percussive


# alias
HPSS = HarmonicPercussiveSourceSeparation
