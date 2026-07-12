"""Chroma feature extraction -- composition port of madmom.audio.chroma, all
three chroma paths the 4.0 audit's corrections flagged: classic
(`PitchClassProfile`/`HarmonicPitchClassProfile`, hand-designed filterbank
weighting), DNN (`DeepChromaProcessor`, a small `NeuralNetwork` trained to
predict chroma bins directly), and CLP (`CLPChroma`/`CLPChromaProcessor`,
Compressed Log Pitch -- a pure-DSP, time-domain-filterbank-based chroma,
needed by `features/downbeats.py`'s `RNNBarProcessor`).

`PitchClassProfile`/`HarmonicPitchClassProfile` are composition subclasses of
`audio/spectrogram.py`'s `Spectrogram` (matching that module's own
composition-not-ndarray-subclass convention, docs/DESIGN.md C.2) rather than
upstream's `FilteredSpectrogram`-via-`__new__`/`__array_finalize__` ndarray
views (`madmom-upstream/madmom/audio/chroma.py:25-187`) -- `np.dot(spectrogram,
filterbank)` still works identically since both operands implement
`__array__`.

`DeepChromaProcessor` (`chroma.py:197-271`) composes the SAME
"`FilteredSpectrogramProcessor` -> `LogarithmicSpectrogramProcessor`"
two-stage split every other end-to-end processor in this project uses
INSTEAD of upstream's fused `LogarithmicFilteredSpectrogramProcessor`
convenience class (see `madmom_infer/features/key.py`'s header for why
that's numerically identical) -- plus one extra composition wrinkle
upstream doesn't need: its own filtered-log-spectrogram stage's OUTPUT is
then re-wrapped as a fresh `Signal` (`spec_signal = SignalProcessor(
sample_rate=10)`, treating the chroma-rate spectrogram as if it were itself
an audio signal sampled at 10 Hz, purely to reuse `FramedSignalProcessor`'s
overlapping-window machinery for the DNN's own 15-frame context window) --
since `Signal.__init__` only accepts an already-`np.ndarray` `data`
argument (or a file path) and this project's `LogarithmicSpectrogram` is a
composition object, NOT an `np.ndarray` subclass, a plain `np.asarray` stage
is inserted between them (matching the same "insert a plain callable
between two Processor stages" pattern `RNNDownBeatProcessor`'s `np.hstack`
and `CNNKeyRecognitionProcessor`'s `add_axis`/`softmax` already establish).

`CLPChroma`/`CLPChromaProcessor` (`chroma.py:283-424`) are the pure-DSP path:
no `NeuralNetwork` weights, but a real, load-bearing dependency on
`audio/spectrogram.py`'s `SemitoneBandpassSpectrogram`, in turn `audio/
signal.py`'s ffmpeg-subprocess `resample` -- see those modules' headers for
why. Measured, not assumed, precision: this port's `SemitoneBandpassSpectrogram`
matches real madmom's own output up to a relative difference on the order of
1e-3 (NOT bit-identical, NOT mere ULP drift) when the two sides run under
DIFFERENT scipy versions (this project's dev venv: scipy 1.17.1; the
reference venv: scipy 1.15.3) -- traced to `scipy.signal.filtfilt`'s
recursive (IIR) nature amplifying tiny per-version `ellip()` filter-
coefficient differences over ~1.5s of audio, not a bug in this port (the
`resample()` step itself IS bit-identical across the same two environments,
confirmed separately -- see `audio/signal.py`'s header). `tests/test_chroma.py`
asserts this measured tolerance explicitly rather than claiming exactness
that doesn't hold; downstream, `RNNBarProcessor`'s DECODED bar-relative
positions still come out identical in every case tested (see
`tests/test_downbeats_rnn.py`), since the GRU ensemble's own forward pass
is proven bit-exact independently (Wave 4c) and empirically tolerates this
level of upstream feature noise on the shared test wavs.

Reads: madmom_infer/audio/filters.py (A4, Filterbank,
PitchClassProfileFilterbank, HarmonicPitchClassProfileFilterbank),
madmom_infer/audio/spectrogram.py (Spectrogram, FilteredSpectrogram,
SemitoneBandpassSpectrogram), madmom_infer/audio/signal.py (SignalProcessor,
FramedSignalProcessor), madmom_infer/audio/stft.py
(ShortTimeFourierTransformProcessor), madmom_infer/ml/nn/__init__.py
(NeuralNetworkEnsemble), madmom_infer/models.py (chroma_dnn download),
madmom_infer/processors.py (Processor, SequentialProcessor); read by:
madmom_infer/features/chords.py (DeepChromaChordRecognitionProcessor),
madmom_infer/features/downbeats.py (RNNBarProcessor.harm_feat).
"""

import warnings

import numpy as np

from ..processors import Processor, SequentialProcessor
from .filters import (
    A4,
    Filterbank,
    HarmonicPitchClassProfileFilterbank,
    PitchClassProfileFilterbank,
)
from .spectrogram import SemitoneBandpassSpectrogram, Spectrogram

PCP = PitchClassProfileFilterbank
HPCP = HarmonicPitchClassProfileFilterbank


class PitchClassProfile(Spectrogram):
    """Simple pitch class profile (PCP), i.e. chroma vector, extraction from
    a spectrogram.

    Composition port of `madmom.audio.chroma.PitchClassProfile`
    (`madmom-upstream/madmom/audio/chroma.py:25-108`).

    Parameters
    ----------
    spectrogram : Spectrogram instance
        `Spectrogram` instance (must NOT already be filtered).
    filterbank : Filterbank class or instance, optional
        `Filterbank` class or instance.
    num_classes : int, optional
        Number of pitch classes.
    fmin : float, optional
        Minimum frequency of the PCP filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the PCP filterbank [Hz].
    fref : float, optional
        Reference frequency for the first PCP bin [Hz].
    kwargs : dict, optional
        If no `Spectrogram` instance was given, one is instantiated with
        these additional keyword arguments.

    Notes
    -----
    If `fref` is `None`, upstream estimates it from `spectrogram.
    tuning_frequency()` -- that method is not ported in this project (see
    `audio/spectrogram.py`'s header), so `fref=None` raises
    `NotImplementedError` here rather than silently using a wrong value.

    References
    ----------
    .. [1] T. Fujishima, "Realtime chord recognition of musical sound: a
           system using Common Lisp Music", Proceedings of the
           International Computer Music Conference (ICMC), 1999.
    """

    def __init__(self, spectrogram, filterbank=PCP, num_classes=PCP.CLASSES,
                 fmin=PCP.FMIN, fmax=PCP.FMAX, fref=A4, **kwargs):
        if not isinstance(spectrogram, Spectrogram):
            spectrogram = Spectrogram(spectrogram, **kwargs)
        if hasattr(spectrogram, "filterbank"):
            warnings.warn("Spectrogram should not be filtered.",
                          RuntimeWarning)
        if fref is None:
            raise NotImplementedError(
                "fref=None (auto-estimate via Spectrogram.tuning_frequency) "
                "is not implemented -- tuning_frequency() is out of scope, "
                "see madmom_infer/audio/spectrogram.py's header. Pass an "
                "explicit fref instead."
            )
        if isinstance(filterbank, type) and issubclass(filterbank, Filterbank):
            filterbank = filterbank(
                spectrogram.bin_frequencies, num_classes=num_classes,
                fmin=fmin, fmax=fmax, fref=fref,
            )
        if not isinstance(filterbank, Filterbank):
            raise ValueError(
                "not a Filterbank type or instance: %s" % filterbank
            )
        self.data = np.dot(np.asarray(spectrogram), np.asarray(filterbank))
        self.filterbank = filterbank
        self.spectrogram = spectrogram
        self.stft = getattr(spectrogram, "stft", None)


class HarmonicPitchClassProfile(PitchClassProfile):
    """Harmonic pitch class profile (HPCP) extraction from a spectrogram.

    Composition port of `madmom.audio.chroma.HarmonicPitchClassProfile`
    (`madmom-upstream/madmom/audio/chroma.py:110-187`).

    Parameters
    ----------
    spectrogram : Spectrogram instance
        `Spectrogram` instance (must NOT already be filtered).
    filterbank : Filterbank class or instance, optional
        Filterbank class or instance.
    num_classes : int, optional
        Number of harmonic pitch classes.
    fmin : float, optional
        Minimum frequency of the HPCP filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the HPCP filterbank [Hz].
    fref : float, optional
        Reference frequency for the first HPCP bin [Hz].
    window : int, optional
        Length of the weighting window [bins].
    kwargs : dict, optional
        If no `Spectrogram` instance was given, one is instantiated with
        these additional keyword arguments.

    References
    ----------
    .. [1] Emilia Gomez, "Tonal Description of Music Audio Signals", PhD
           thesis, Universitat Pompeu Fabra, Barcelona, Spain, 2006.
    """

    # pylint: disable=super-init-not-called

    def __init__(self, spectrogram, filterbank=HPCP, num_classes=HPCP.CLASSES,
                 fmin=HPCP.FMIN, fmax=HPCP.FMAX, fref=A4, window=HPCP.WINDOW,
                 **kwargs):
        if not isinstance(spectrogram, Spectrogram):
            spectrogram = Spectrogram(spectrogram, **kwargs)
        if hasattr(spectrogram, "filterbank"):
            warnings.warn("Spectrogram should not be filtered.",
                          RuntimeWarning)
        if fref is None:
            raise NotImplementedError(
                "fref=None (auto-estimate via Spectrogram.tuning_frequency) "
                "is not implemented -- tuning_frequency() is out of scope, "
                "see madmom_infer/audio/spectrogram.py's header. Pass an "
                "explicit fref instead."
            )
        if isinstance(filterbank, type) and issubclass(filterbank, Filterbank):
            filterbank = filterbank(
                spectrogram.bin_frequencies, num_classes=num_classes,
                fmin=fmin, fmax=fmax, fref=fref, window=window,
            )
        if not isinstance(filterbank, Filterbank):
            raise ValueError(
                "not a Filterbank type or instance: %s" % filterbank
            )
        self.data = np.dot(np.asarray(spectrogram), np.asarray(filterbank))
        self.filterbank = filterbank
        self.spectrogram = spectrogram
        self.stft = getattr(spectrogram, "stft", None)


def _dcp_flatten(fs):
    """Flatten (stack + reshape) `DeepChromaProcessor`'s overlapping context
    frames into one `(num_frames, 15 * num_bins)` matrix.

    Verbatim port of `madmom.audio.chroma._dcp_flatten`
    (`madmom-upstream/madmom/audio/chroma.py:190-194`). Kept as a module-
    level function -- matches upstream's own reason (picklability for
    multiprocessing), which this project's `ParallelProcessor` doesn't use,
    but is also just a convenient, testable unit either way.
    """
    return np.concatenate(fs).reshape(len(fs), -1)


class DeepChromaProcessor(SequentialProcessor):
    """Compute chroma vectors from an audio file using a deep neural network
    that focuses on harmonically relevant spectral content, as described
    in [1]_.

    Composition port of `madmom.audio.chroma.DeepChromaProcessor`
    (`madmom-upstream/madmom/audio/chroma.py:197-271`) -- see this module's
    header for the composition wrinkle (`np.asarray` inserted before
    re-wrapping the filtered-log-spectrogram output as a fresh `Signal`).

    Parameters
    ----------
    fmin : int, optional
        Minimum frequency of the filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the filterbank [Hz].
    unique_filters : bool, optional
        Indicate if the filterbank should contain only unique filters, i.e.
        remove duplicate filters resulting from insufficient resolution at
        low frequencies.
    models : list of filenames, optional
        List of model filenames. Defaults to `None`, which loads madmom's
        own pretrained `CHROMA_DNN` model (downloaded at runtime,
        sha256-verified, via `madmom_infer.models.chroma_dnn()` -- CC
        BY-NC-SA 4.0, non-commercial use only, see `madmom_infer/models.py`).

    Notes
    -----
    Provided model files must be compatible with the processing pipeline and
    the values of `fmin`, `fmax`, and `unique_filters`.

    References
    ----------
    .. [1] Filip Korzeniowski and Gerhard Widmer, "Feature Learning for
           Chord Recognition: The Deep Chroma Extractor", Proceedings of the
           17th International Society for Music Information Retrieval
           Conference (ISMIR), 2016.
    """

    def __init__(self, fmin=65, fmax=2100, unique_filters=True, models=None,
                 **kwargs):
        from ..models import chroma_dnn
        from ..ml.nn import NeuralNetworkEnsemble
        from .signal import FramedSignalProcessor, SignalProcessor
        from .spectrogram import (
            FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
        )
        from .stft import ShortTimeFourierTransformProcessor

        # signal pre-processing
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        frames = FramedSignalProcessor(frame_size=8192, fps=10)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(
            num_bands=24, fmin=fmin, fmax=fmax, unique_filters=unique_filters)
        log = LogarithmicSpectrogramProcessor(mul=1, add=1)
        # split the spectrogram into overlapping frames
        spec_signal = SignalProcessor(sample_rate=10)
        spec_frames = FramedSignalProcessor(frame_size=15, hop_size=1, fps=10)
        # predict chroma bins with a DNN
        nn = NeuralNetworkEnsemble.load(models or chroma_dnn(), **kwargs)
        # instantiate a SequentialProcessor
        super().__init__((sig, frames, stft, filt, log, np.asarray,
                          spec_signal, spec_frames, _dcp_flatten, nn))


# Compressed Log Pitch (CLP) chroma stuff
CLP_FPS = 50
CLP_FMIN = 27.5
CLP_FMAX = 4200.0
CLP_COMPRESSION_FACTOR = 100
CLP_NORM = True
CLP_THRESHOLD = 0.001


class CLPChroma:
    """Compressed Log Pitch (CLP) chroma as proposed in [1]_ and [2]_.

    Composition port of `madmom.audio.chroma.CLPChroma`
    (`madmom-upstream/madmom/audio/chroma.py:283-367`) -- NOT an
    `np.ndarray` subclass, own composition class (see `audio/spectrogram.py`'s
    `SemitoneBandpassSpectrogram` docstring for the same deviation pattern).

    Parameters
    ----------
    data : str, Signal, or SemitoneBandpassSpectrogram
        Input data.
    fps : int, optional
        Desired frame rate of the signal [Hz].
    fmin : float, optional
        Lowest frequency of the spectrogram [Hz].
    fmax : float, optional
        Highest frequency of the spectrogram [Hz].
    compression_factor : float, optional
        Factor for compression of the energy.
    norm : bool, optional
        Normalize the energy of each frame to one (divide by the L2 norm).
    threshold : float, optional
        If the energy of a frame is below a threshold, the energy is
        equally distributed among all chroma bins.

    Notes
    -----
    The resulting chromagrams differ slightly from those obtained by the
    MATLAB chroma toolbox [2]_ because of different resampling and filter
    methods -- and, in THIS port specifically, measurably (not just
    ULP-level) from real madmom's own output whenever the two sides run
    under different scipy versions, see this module's header.

    References
    ----------
    .. [1] Meinard Mueller, "Information retrieval for music and motion",
           Springer, 2007.
    .. [2] Meinard Mueller and Sebastian Ewert, "Chroma Toolbox: MATLAB
           Implementations for Extracting Variants of Chroma-Based Audio
           Features", Proceedings of the International Conference on Music
           Information Retrieval (ISMIR), 2011.
    """

    def __init__(self, data, fps=CLP_FPS, fmin=CLP_FMIN, fmax=CLP_FMAX,
                 compression_factor=CLP_COMPRESSION_FACTOR, norm=CLP_NORM,
                 threshold=CLP_THRESHOLD, **kwargs):
        from .filters import hz2midi

        # check input type
        if not isinstance(data, SemitoneBandpassSpectrogram):
            data = SemitoneBandpassSpectrogram(data, fps=fps, fmin=fmin,
                                               fmax=fmax, **kwargs)
        # apply log compression
        log_pitch_energy = np.log10(np.asarray(data) * compression_factor + 1)
        # compute chroma by adding up bins that correspond to the same
        # pitch class
        obj = np.zeros((log_pitch_energy.shape[0], 12))
        midi_min = int(np.round(hz2midi(data.bin_frequencies[0])))
        for p in range(log_pitch_energy.shape[1]):
            # make sure that p maps to the correct bin_label (midi_min=12
            # corresponds to a C and therefore chroma_idx=0)
            chroma_idx = np.mod(midi_min + p, 12)
            obj[:, chroma_idx] += log_pitch_energy[:, p]
        if norm:
            # normalise the vectors according to the l2 norm
            mean_energy = np.sqrt((obj ** 2).sum(axis=1))
            idx_below_threshold = np.where(mean_energy < threshold)
            obj = obj / mean_energy[:, np.newaxis]
            obj[idx_below_threshold, :] = np.ones((1, 12)) / np.sqrt(12)
        self.data = obj
        self.bin_labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G",
                           "G#", "A", "A#", "B"]
        self.fps = fps

    # -- numpy interop, mirroring audio/signal.py's Signal -----------------
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


class CLPChromaProcessor(Processor):
    """Compressed Log Pitch (CLP) Chroma Processor.

    Port of `madmom.audio.chroma.CLPChromaProcessor`
    (`madmom-upstream/madmom/audio/chroma.py:369-424`).

    Parameters
    ----------
    fps : int, optional
        Desired frame rate of the signal [Hz].
    fmin : float, optional
        Lowest frequency of the spectrogram [Hz].
    fmax : float, optional
        Highest frequency of the spectrogram [Hz].
    compression_factor : float, optional
        Factor for compression of the energy.
    norm : bool, optional
        Normalize the energy of each frame to one (divide by the L2 norm).
    threshold : float, optional
        If the energy of a frame is below a threshold, the energy is
        equally distributed among all chroma bins.
    """

    def __init__(self, fps=CLP_FPS, fmin=CLP_FMIN, fmax=CLP_FMAX,
                 compression_factor=CLP_COMPRESSION_FACTOR, norm=CLP_NORM,
                 threshold=CLP_THRESHOLD, **kwargs):
        # pylint: disable=unused-argument
        self.fps = fps
        self.fmin = fmin
        self.fmax = fmax
        self.compression_factor = compression_factor
        self.norm = norm
        self.threshold = threshold

    def process(self, data, **kwargs):
        """Create a `CLPChroma` from the given data.

        Parameters
        ----------
        data : Signal instance or filename
            Data to be processed.

        Returns
        -------
        clp : CLPChroma instance
            CLPChroma.
        """
        args = dict(fps=self.fps, fmin=self.fmin, fmax=self.fmax,
                    compression_factor=self.compression_factor,
                    norm=self.norm, threshold=self.threshold)
        args.update(kwargs)
        return CLPChroma(data, **args)
