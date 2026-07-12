"""Reimplementation of madmom.features.chords -- major/minor chord
recognition, the two end-to-end pipelines Wave 4d's audit row names:
`DeepChromaChordRecognitionProcessor` (DNN chroma -> CRF) and
`CNNChordFeatureProcessor` + `CRFChordRecognitionProcessor` (CNN features ->
CRF), plus `majmin_targets_to_chord_labels`, the shared frame-labels ->
merged-segments decoding helper both CRF processors chain onto.

Neither chord-recognition path in this module touches `audio/chroma.py`'s
`CLPChroma`/`CLPChromaProcessor` -- confirmed by reading
`madmom-upstream/madmom/features/chords.py` directly, not assumed. This
matters: it means full audio-in, chord-segments-out recognition is
achievable and EXACT-testable without the ffmpeg/`SemitoneBandpassFilterbank`
dependency chain `RNNBarProcessor`'s harmonic branch needs (see
`audio/chroma.py`'s module header) -- `DeepChromaChordRecognitionProcessor`
uses `audio/chroma.py`'s `DeepChromaProcessor` (DNN chroma, ordinary
filtered-log-spectrogram frontend), `CNNChordFeatureProcessor` uses the same
filtered-log-spectrogram frontend directly (no chroma stage at all, a raw
CNN feature extractor).

`ConditionalRandomField.load(model or CHORDS_DCCRF[0])`/`.load(model or
CHORDS_CFCRF[0])` (`ml/crf.py`) do the actual chord decoding -- see that
module's header for how its `.load()` differs from upstream's inherited
`Processor.load` (restricted `SafeUnpickler`, not bare `pickle.load`).

`SEGMENT_DTYPE` (`[('start', float), ('end', float), ('label', object)]`) is
defined here directly rather than imported from a ported `madmom.io`
package -- `io/*` stays a permanent EXCLUDE per CLAUDE.md's 4.0 audit (out
of this project's `features/`/`audio/`/`ml/` scope); this one structured-
dtype constant is cheap enough to inline verbatim rather than pull in an
entire I/O subpackage for.

Reads: madmom_infer/audio/{signal,stft,spectrogram,chroma}.py (the
pre-processing cascades), madmom_infer/ml/crf.py (ConditionalRandomField),
madmom_infer/ml/nn/__init__.py (NeuralNetwork), madmom_infer/models.py
(chords_dccrf/chords_cnn_feat/chords_cfcrf download), madmom_infer/utils.py
(segment_axis), madmom_infer/processors.py (SequentialProcessor); read by:
nothing yet (Wave 4d's own end-to-end target).
"""

from functools import partial

import numpy as np

from madmom_infer.processors import SequentialProcessor

# structured dtype for a labelled segment (start, end, label) -- see this
# module's header for why it's inlined rather than imported from `io.*`.
SEGMENT_DTYPE = [("start", float), ("end", float), ("label", object)]


def majmin_targets_to_chord_labels(targets, fps):
    """Convert a series of major/minor chord targets to human-readable
    chord labels.

    Verbatim port of `madmom.features.chords.majmin_targets_to_chord_labels`
    (`madmom-upstream/madmom/features/chords.py:16-71`). Targets are assumed
    to be spaced equidistant in time as defined by `fps` (each target
    represents one frame).

    Ids 0-11 encode major chords starting with root 'A', 12-23 minor chords.
    Id 24 represents 'N', the no-chord class.

    Parameters
    ----------
    targets : iterable
        Iterable containing chord class ids.
    fps : float
        Frames per second.

    Returns
    -------
    chord labels : numpy array (structured, SEGMENT_DTYPE)
        Array of (start time, end time, chord label) tuples.
    """
    # create a map of semitone index to semitone name (e.g. 0 -> A, 1 -> A#)
    pitch_class_to_label = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F",
                            "F#", "G", "G#"]

    def pred_to_cl(pred):
        """Map a class id to a chord label: 0..11 major chords, 12..23
        minor chords, 24 no chord."""
        if pred == 24:
            return "N"
        return "{}:{}".format(pitch_class_to_label[pred % 12],
                              "maj" if pred < 12 else "min")

    # get labels per frame
    spf = 1.0 / fps
    labels = [(i * spf, pred_to_cl(p)) for i, p in enumerate(targets)]

    # join same consecutive predictions
    prev_label = (None, None)
    uniq_labels = []
    for label in labels:
        if label[1] != prev_label[1]:
            uniq_labels.append(label)
            prev_label = label

    # end time of last label is one frame duration after the last
    # prediction time
    start_times, chord_labels = zip(*uniq_labels)
    end_times = start_times[1:] + (labels[-1][0] + spf,)

    return np.array(list(zip(start_times, end_times, chord_labels)),
                    dtype=SEGMENT_DTYPE)


class DeepChromaChordRecognitionProcessor(SequentialProcessor):
    """Recognise major and minor chords from deep chroma vectors [1]_ using
    a Conditional Random Field.

    Port of `madmom.features.chords.DeepChromaChordRecognitionProcessor`
    (`madmom-upstream/madmom/features/chords.py:74-138`).

    Parameters
    ----------
    model : str, optional
        File containing the CRF model. If `None`, use the model supplied
        with madmom (`CHORDS_DCCRF`, downloaded at runtime, sha256-verified,
        via `madmom_infer.models.chords_dccrf()` -- CC BY-NC-SA 4.0,
        non-commercial use only, see `madmom_infer/models.py`).
    fps : float, optional
        Frames per second. Must correspond to the fps of the incoming
        activations and the model.

    References
    ----------
    .. [1] Filip Korzeniowski and Gerhard Widmer, "Feature Learning for
           Chord Recognition: The Deep Chroma Extractor", Proceedings of the
           17th International Society for Music Information Retrieval
           Conference (ISMIR), 2016.

    Examples
    --------
    To recognise chords in an audio file, first create a
    `madmom_infer.audio.chroma.DeepChromaProcessor` to extract the
    appropriate chroma vectors, then this class to decode a chord sequence
    from the extracted chromas:

    >>> from madmom_infer.audio.chroma import DeepChromaProcessor
    >>> from madmom_infer.features.chords import (
    ...     DeepChromaChordRecognitionProcessor,
    ... )
    >>> dcp = DeepChromaProcessor()
    >>> decode = DeepChromaChordRecognitionProcessor()
    >>> chordrec = SequentialProcessor([dcp, decode])
    >>> chordrec('track.wav')  # doctest: +SKIP
    array([(0. , 1.6, 'F:maj'), (1.6, 2.5, 'A:maj'), (2.5, 4.1, 'D:maj')],
          dtype=[('start', '<f8'), ('end', '<f8'), ('label', 'O')])
    """

    def __init__(self, model=None, fps=10, **kwargs):
        # pylint: disable=unused-argument
        from ..ml.crf import ConditionalRandomField
        from ..models import chords_dccrf

        crf = ConditionalRandomField.load(model or chords_dccrf()[0])
        lbl = partial(majmin_targets_to_chord_labels, fps=fps)
        super().__init__((crf, lbl))


# functions necessary for CNNChordFeatureProcessor -- kept outside the class
# so the processor stays picklable (same reason as upstream, this project
# doesn't actually pickle processors, but the split is free and matches the
# original 1:1).
def _cnncfp_pad(data):
    """Pad the input. Verbatim port of `madmom.features.chords._cnncfp_pad`
    (`madmom-upstream/madmom/features/chords.py:143-146`)."""
    pad_data = np.zeros((11, 113))
    return np.vstack([pad_data, np.asarray(data), pad_data])


def _cnncfp_superframes(data):
    """Segment input into superframes. Verbatim port of
    `madmom.features.chords._cnncfp_superframes`
    (`madmom-upstream/madmom/features/chords.py:149-152`) -- `segment_axis`
    here is `madmom_infer.utils.segment_axis`'s narrow (`axis=0`,
    `end='cut'`) carve-out, exactly the case this call needs."""
    from ..utils import segment_axis

    return segment_axis(data, 3, 1, axis=0)


def _cnncfp_avg(data):
    """Global average pool. Verbatim port of
    `madmom.features.chords._cnncfp_avg`
    (`madmom-upstream/madmom/features/chords.py:155-157`)."""
    return data.mean((1, 2))


class CNNChordFeatureProcessor(SequentialProcessor):
    """Extract learned features for chord recognition, as described in
    [1]_.

    Port of `madmom.features.chords.CNNChordFeatureProcessor`
    (`madmom-upstream/madmom/features/chords.py:160-212`). Composes the same
    "`FilteredSpectrogramProcessor` -> `LogarithmicSpectrogramProcessor`"
    two-stage split every other end-to-end processor in this project uses
    instead of upstream's fused `LogarithmicFilteredSpectrogramProcessor`
    (see `madmom_infer/features/key.py`'s header for why that's numerically
    identical).

    References
    ----------
    .. [1] Filip Korzeniowski and Gerhard Widmer, "A Fully Convolutional
           Deep Auditory Model for Musical Chord Recognition", Proceedings
           of IEEE International Workshop on Machine Learning for Signal
           Processing (MLSP), 2016.

    Examples
    --------
    >>> proc = CNNChordFeatureProcessor()
    >>> features = proc('track.wav')  # doctest: +SKIP
    >>> features.shape  # doctest: +SKIP
    (41, 128)
    """

    def __init__(self, nn_file=None, **kwargs):
        # pylint: disable=unused-argument
        from ..audio.signal import FramedSignalProcessor, SignalProcessor
        from ..audio.spectrogram import (
            FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
        )
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..ml.nn import NeuralNetwork
        from ..models import chords_cnn_feat

        # spectrogram computation
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        frames = FramedSignalProcessor(frame_size=8192, fps=10)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(
            num_bands=24, fmin=60, fmax=2600, unique_filters=True)
        log = LogarithmicSpectrogramProcessor(mul=1, add=1)

        # padding, neural network and global average pooling
        pad = _cnncfp_pad
        # `nn_file` override (not in upstream, which hardcodes CHORDS_CNN_
        # FEAT[0] -- added here purely for testability, matching the
        # `nn_files=`/`models=` override convention `CNNKeyRecognitionProcessor`/
        # `DeepChromaProcessor` already establish) lets tests point at a
        # local `.pkl` copy without needing madmom_infer.models's runtime
        # download.
        nn = NeuralNetwork.load(nn_file or chords_cnn_feat()[0])
        superframes = _cnncfp_superframes
        avg = _cnncfp_avg

        # create processing pipeline
        super().__init__((sig, frames, stft, filt, log, pad, nn,
                          superframes, avg))


class CRFChordRecognitionProcessor(SequentialProcessor):
    """Recognise major and minor chords from learned features extracted by
    a convolutional neural network, as described in [1]_.

    Port of `madmom.features.chords.CRFChordRecognitionProcessor`
    (`madmom-upstream/madmom/features/chords.py:215-278`). Note: upstream's
    own naming (preserved here, not "fixed") calls the model constant
    `CHORDS_CFCRF` (`CF` = "CNN Feature") -- see `madmom_infer/models.py`'s
    header.

    Parameters
    ----------
    model : str, optional
        File containing the CRF model. If `None`, use the model supplied
        with madmom (`CHORDS_CFCRF`, downloaded at runtime, sha256-verified,
        via `madmom_infer.models.chords_cfcrf()` -- CC BY-NC-SA 4.0,
        non-commercial use only, see `madmom_infer/models.py`).
    fps : float, optional
        Frames per second. Must correspond to the fps of the incoming
        activations and the model.

    References
    ----------
    .. [1] Filip Korzeniowski and Gerhard Widmer, "A Fully Convolutional
           Deep Auditory Model for Musical Chord Recognition", Proceedings
           of IEEE International Workshop on Machine Learning for Signal
           Processing (MLSP), 2016.

    Examples
    --------
    To recognise chords, first extract features using
    `CNNChordFeatureProcessor`, then this class to decode a chord sequence:

    >>> from madmom_infer.features.chords import (
    ...     CNNChordFeatureProcessor, CRFChordRecognitionProcessor,
    ... )
    >>> featproc = CNNChordFeatureProcessor()
    >>> decode = CRFChordRecognitionProcessor()
    >>> chordrec = SequentialProcessor([featproc, decode])
    >>> chordrec('track.wav')  # doctest: +SKIP
    array([(0. , 0.2, 'N'), (0.2, 1.6, 'F:maj'),
           (1.6, 2.4, 'A:maj'), (2.4, 4.1, 'D:min')],
          dtype=[('start', '<f8'), ('end', '<f8'), ('label', 'O')])
    """

    def __init__(self, model=None, fps=10, **kwargs):
        # pylint: disable=unused-argument
        from ..ml.crf import ConditionalRandomField
        from ..models import chords_cfcrf

        crf = ConditionalRandomField.load(model or chords_cfcrf()[0])
        lbl = partial(majmin_targets_to_chord_labels, fps=fps)
        super().__init__((crf, lbl))
