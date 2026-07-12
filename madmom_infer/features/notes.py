"""Reimplementation of madmom.features.notes -- piano note transcription:
`RNNPianoNoteProcessor`/`CNNPianoNoteProcessor` (audio -> per-pitch note
activation functions), `NoteOnsetPeakPickingProcessor`/
`NotePeakPickingProcessor` (activations -> [time, pitch] onset events, reused
onset-family peak-picking), and `ADSRNoteTrackingProcessor` (the CNN's
3-channel [note, onset, offset] activations -> [time, pitch, duration] note
segments, via `notes_hmm.py`'s ADSR HMM on the existing `ml/hmm.py` Viterbi
decoder). Wave 4e of the complete-port campaign -- see CLAUDE.md's audit
table, `features/notes.py` rows.

Port of `madmom-upstream/madmom/features/notes.py` (456 lines). Every
class the 4.0 audit table's `features/notes.py` rows list as TO-PORT is
here.

**`RNNPianoNoteProcessor` is a straightforward port** -- `notes_brnn.pkl`
(`NOTES_BRNN`) references only already-ported classes (`NeuralNetwork`,
`BidirectionalLayer`, `FeedForwardLayer`, `RecurrentLayer`, confirmed by
`pickletools`, see `ml/nn/unpickle.py`'s header), and its pre-processing
chain is the same multi-resolution (1024/2048/4096-sample frames)
filtered-log-diff-spectrogram cascade `RNNOnsetProcessor` already
establishes, just with different filterbank/diff parameters
(`num_bands=12, fmin=30, fmax=17000`, `LogarithmicSpectrogramProcessor(mul=5,
add=1)`, `SpectrogramDifferenceProcessor(diff_ratio=0.5, positive_diffs=True,
stack_diffs=np.hstack)`).

**`CNNPianoNoteProcessor` is where this wave's real surprise lives --
`notes_cnn.pkl` (`NOTES_CNN`) does not pickle a bare `NeuralNetwork`,**
it pickles the model's ENTIRE multi-task (note/onset/offset)
`SequentialProcessor`/`ParallelProcessor` graph directly (found by actually
`pickletools`-walking and then loading the file, not guessed -- see `ml/nn/
unpickle.py`'s header for the full finding, including two more new
allowlist primitives this required: `numpy.dstack` for the final merge
stage, and `itertools.imap`/`_codecs.encode`, two Python-2-pickle-compat
primitives real madmom's own bare `pickle.load` resolves transparently via
`pickle._compat_pickle.NAME_MAPPING` but `SafeUnpickler.find_class` does
not, needing explicit entries). This turned out to be a NON-EVENT for this
port's own code, though: `madmom_infer.ml.nn.NeuralNetwork.load`/
`NeuralNetworkEnsemble.load` were already fully generic (`unpickle.
load_model` just returns whatever top-level object type the pickle
actually contains, matching upstream's own `Processor.load`'s equally
generic behavior) -- `NeuralNetworkEnsemble.load(NOTES_CNN)` wraps the
unpickled `SequentialProcessor` in a size-1 `ParallelProcessor` +
`average_predictions` exactly like it would wrap a bare `NeuralNetwork`,
and `average_predictions` degrades to the identity function for a
length-1 list (`ml/nn/__init__.py`), so the model's own baked-in
`np.dstack` merge is simply the ensemble's single "network" -- no new code
needed in `ml/nn/__init__.py` at all, only the 2 new layer classes
(`ReshapeLayer`, `TransposeLayer`, `ml/nn/layers.py`) and the unpickle
allowlist entries this graph shape needs to reconstruct.

**`ADSRNoteTrackingProcessor`** decodes each of the 88 pitches
INDEPENDENTLY (one `HiddenMarkovModel.viterbi()` call per pitch column,
`activations[:, pitch, :]`, matching upstream's own per-pitch loop
exactly) -- `notes_hmm.py`'s `ADSRStateSpace`/`ADSRTransitionModel`/
`ADSRObservationModel` build the small (silence + attack + decay + sustain
+ release) per-pitch state space, and `ml/hmm.py`'s existing Viterbi
decoder (unmodified, Phase-1 machinery) does the actual decode. A decoded
note is kept only if its Viterbi path visits every ADSR phase (unless
`complete=False`) AND its peak note/onset activation clears
`note_threshold`/`onset_threshold` -- verbatim port of `notes.py:401-456`.

`NoteOnsetPeakPickingProcessor` subclasses `madmom_infer.features.onsets.
OnsetPeakPickingProcessor` (already offline-only, see that module's header)
and reuses its module-level `peak_picking` function directly -- adds a
`pitch_offset` (MIDI note number = 21 + array column, for the standard
88-key piano range) and returns `[time, pitch]` pairs instead of bare onset
times. `NotePeakPickingProcessor` is upstream's own deprecated-since-0.17
alias (`fps=100, pitch_offset=21` defaults) -- ported anyway since the 4.0
audit table lists it as a real, still-present public class, not dead code
this project gets to skip.

Reads: madmom_infer/audio/{signal,stft,spectrogram}.py (the pre-processing
cascades), madmom_infer/features/onsets.py (OnsetPeakPickingProcessor,
peak_picking), madmom_infer/features/notes_hmm.py (ADSRStateSpace,
ADSRTransitionModel, ADSRObservationModel), madmom_infer/ml/hmm.py
(HiddenMarkovModel), madmom_infer/ml/nn/__init__.py (NeuralNetwork,
NeuralNetworkEnsemble), madmom_infer/models.py (notes_brnn/notes_cnn
download), madmom_infer/utils.py (combine_events); read by: nothing yet
(Wave 4e's own end-to-end target).
"""

import numpy as np

from .onsets import OnsetPeakPickingProcessor, peak_picking
from ..processors import ParallelProcessor, Processor, SequentialProcessor
from ..utils import combine_events


# ---------------------------------------------------------------------------
# class for detecting notes with a RNN
# ---------------------------------------------------------------------------
class RNNPianoNoteProcessor(SequentialProcessor):
    """Processor to get a (piano) note onset activation function from a RNN.

    Port of `madmom.features.notes.RNNPianoNoteProcessor`
    (`madmom-upstream/madmom/features/notes.py:24-88`). `nn_file`, if given,
    overrides `NOTES_BRNN[0]` entirely (not in upstream -- matches
    `CNNChordFeatureProcessor`'s own singular-file-override convention,
    used by the cross-BLAS test to point at a local `.pkl` copy).

    References
    ----------
    .. [1] Sebastian Boeck and Markus Schedl,
           "Polyphonic Piano Note Transcription with Recurrent Neural
           Networks", Proceedings of the 37th International Conference on
           Acoustics, Speech and Signal Processing (ICASSP), 2012.
    """

    def __init__(self, nn_file=None, **kwargs):
        # pylint: disable=unused-argument
        from ..audio.signal import FramedSignalProcessor, SignalProcessor
        from ..audio.spectrogram import (
            FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
            SpectrogramDifferenceProcessor,
        )
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..ml.nn import NeuralNetwork
        from ..models import notes_brnn

        # define pre-processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        # process the multi-resolution spec & diff in parallel
        multi = ParallelProcessor([])
        for frame_size in [1024, 2048, 4096]:
            frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
            stft = ShortTimeFourierTransformProcessor()  # caching FFT window
            filt = FilteredSpectrogramProcessor(
                num_bands=12, fmin=30, fmax=17000, norm_filters=True)
            spec = LogarithmicSpectrogramProcessor(mul=5, add=1)
            diff = SpectrogramDifferenceProcessor(
                diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
            # process each frame size with spec and diff sequentially
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
        # stack the features and processes everything sequentially
        pre_processor = SequentialProcessor((sig, multi, np.hstack))

        # process the pre-processed signal with a NN
        nn = NeuralNetwork.load(nn_file or notes_brnn()[0])

        # instantiate a SequentialProcessor
        super().__init__((pre_processor, nn))


# ---------------------------------------------------------------------------
# note onset peak-picking
# ---------------------------------------------------------------------------
class NoteOnsetPeakPickingProcessor(OnsetPeakPickingProcessor):
    """Note onset peak-picking: converts a note onset activation function
    (per-pitch columns) into `[time, pitch]` note-onset events.

    Port of `madmom.features.notes.NoteOnsetPeakPickingProcessor`
    (`madmom-upstream/madmom/features/notes.py:91-218`). Reuses
    `OnsetPeakPickingProcessor`'s constructor (this project's version is
    already offline-only, see `features/onsets.py`'s module header) and its
    module-level `peak_picking` function directly.

    Parameters
    ----------
    threshold : float, optional
        Threshold for peak-picking.
    smooth : float, optional
        Smooth the activation function over `smooth` seconds.
    pre_avg : float, optional
        Use `pre_avg` seconds past information for moving average.
    post_avg : float, optional
        Use `post_avg` seconds future information for moving average.
    pre_max : float, optional
        Use `pre_max` seconds past information for moving maximum.
    post_max : float, optional
        Use `post_max` seconds future information for moving maximum.
    combine : float, optional
        Only report one note per pitch within `combine` seconds.
    delay : float, optional
        Report the detected notes `delay` seconds delayed.
    fps : float, optional
        Frames per second used for conversion of timings.
    pitch_offset : int, optional
        Pitch offset for the detected notes (21 for the standard 88-key
        piano range, mapping array column 0 to MIDI note 21/A0).

    Returns
    -------
    notes : numpy array, shape (N, 2)
        Detected notes `[seconds, pitch]`.

    Notes
    -----
    If no moving average is needed (e.g. the activations are independent
    of the signal's level, as for neural network activations), `pre_avg`
    and `post_avg` should be set to 0. For peak picking of local maxima,
    set `pre_max` >= 1. / `fps` and `post_max` >= 1. / `fps`.
    """

    THRESHOLD = 0.5  # binary threshold
    SMOOTH = 0.
    PRE_AVG = 0.
    POST_AVG = 0.
    PRE_MAX = 0.
    POST_MAX = 0.
    COMBINE = 0.03
    DELAY = 0.

    def __init__(self, threshold=THRESHOLD, smooth=SMOOTH, pre_avg=PRE_AVG,
                 post_avg=POST_AVG, pre_max=PRE_MAX, post_max=POST_MAX,
                 combine=COMBINE, delay=DELAY, fps=None, pitch_offset=0,
                 **kwargs):
        # pylint: disable=unused-argument
        super().__init__(threshold=threshold, smooth=smooth, pre_avg=pre_avg,
                          post_avg=post_avg, pre_max=pre_max,
                          post_max=post_max, combine=combine, delay=delay,
                          fps=fps)
        self.pitch_offset = pitch_offset

    def process(self, activations, **kwargs):
        """Detect the notes in the given activation function.

        Parameters
        ----------
        activations : numpy array
            Note activation function.

        Returns
        -------
        onsets : numpy array, shape (N, 2)
            Detected notes `[seconds, pitches]`.
        """
        # pylint: disable=arguments-differ, unused-argument
        # convert timing information to frames and set default values
        timings = np.array([self.smooth, self.pre_avg, self.post_avg,
                             self.pre_max, self.post_max]) * self.fps
        timings = np.round(timings).astype(int)
        # detect the peaks (function returns int indices)
        onsets, pitches = peak_picking(activations, self.threshold, *timings)
        # if no note onsets are detected, return empty array
        if not onsets.any():
            return np.empty((0, 2))
        # convert onset timing and apply pitch offset
        onsets = onsets.astype(float) / self.fps
        pitches = pitches + self.pitch_offset
        # shift if necessary
        if self.delay:
            onsets = onsets + self.delay
        # combine notes
        if self.combine > 0:
            notes = []
            # iterate over each detected note pitch separately
            for pitch in np.unique(pitches):
                # get all onsets for this pitch
                onsets_ = onsets[pitches == pitch]
                # combine onsets
                onsets_ = combine_events(onsets_, self.combine, "left")
                # zip onsets and pitches and add them to list of detections
                notes.extend(list(zip(onsets_, [pitch] * len(onsets_))))
        else:
            # just zip all detected notes
            notes = list(zip(onsets, pitches))
        # sort the detections and return as numpy array
        return np.array(sorted(notes))


class NotePeakPickingProcessor(NoteOnsetPeakPickingProcessor):
    """Deprecated as of version 0.17 (upstream). Kept for API parity -- use
    `NoteOnsetPeakPickingProcessor` directly and set `fps`/`pitch_offset`
    explicitly.

    Port of `madmom.features.notes.NotePeakPickingProcessor`
    (`madmom-upstream/madmom/features/notes.py:221-232`).
    """

    def __init__(self, fps=100, pitch_offset=21, **kwargs):
        # pylint: disable=unused-argument
        super().__init__(fps=fps, pitch_offset=pitch_offset, **kwargs)


# ---------------------------------------------------------------------------
# class for detecting notes with a CNN
# ---------------------------------------------------------------------------
def _cnn_pad(data):
    """Pad the data by repeating the first and last frame 5 times.

    Verbatim port of `madmom.features.notes._cnn_pad`
    (`madmom-upstream/madmom/features/notes.py:235-239`).
    """
    pad_start = np.repeat(data[:1], 5, axis=0)
    pad_stop = np.repeat(data[-1:], 5, axis=0)
    return np.concatenate((pad_start, data, pad_stop))


class CNNPianoNoteProcessor(SequentialProcessor):
    """Processor to get piano note activations from a CNN in a multi-task
    fashion which simultaneously detects onsets, sounding notes, and
    offsets.

    Port of `madmom.features.notes.CNNPianoNoteProcessor`
    (`madmom-upstream/madmom/features/notes.py:242-312`) -- see this
    module's header for the real surprise this wave found: `notes_cnn.pkl`
    (`NOTES_CNN`) pickles the model's entire multi-task
    `SequentialProcessor`/`ParallelProcessor` branch-and-`dstack` graph
    directly, not a bare `NeuralNetwork`; `NeuralNetworkEnsemble.load`
    handles this transparently (no code changes needed there), only the 2
    new layer classes (`ReshapeLayer`, `TransposeLayer`) and the unpickle
    allowlist entries the graph itself needs (`ml/nn/unpickle.py`'s
    header). `nn_files`, if given, overrides `NOTES_CNN` entirely (not in
    upstream -- matches `RNNOnsetProcessor`/`CNNOnsetProcessor`'s own
    convention).

    The activations are returned as a 3-dimensional array: the first axis
    is time, the second is MIDI note (0..87, add `pitch_offset=21` for the
    real MIDI number), and the third dimension holds the [note, onset,
    offset] activations `ADSRNoteTrackingProcessor` expects.

    References
    ----------
    .. [1] Rainer Kelz, Sebastian Boeck and Gerhard Widmer,
           "Deep Polyphonic ADSR Piano Note Transcription", Proceedings of
           the 44th International Conference on Acoustics, Speech and
           Signal Processing (ICASSP), 2019.
    """

    def __init__(self, nn_files=None, **kwargs):
        # pylint: disable=unused-argument
        from ..audio.signal import FramedSignalProcessor, SignalProcessor
        from ..audio.spectrogram import (
            FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
        )
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..ml.nn import NeuralNetworkEnsemble
        from ..models import notes_cnn

        # define pre-processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        frames = FramedSignalProcessor(frame_size=4096, fps=50)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(num_bands=24, fmin=30, fmax=10000)
        spec = LogarithmicSpectrogramProcessor(add=1)
        # pre-processes everything sequentially
        pre_processor = SequentialProcessor(
            (sig, frames, stft, filt, spec, _cnn_pad))
        # process the pre-processed signal with a NN
        nn = NeuralNetworkEnsemble.load(nn_files or notes_cnn())
        # instantiate a SequentialProcessor
        super().__init__((pre_processor, nn))


# ---------------------------------------------------------------------------
# ADSR HMM note tracking
# ---------------------------------------------------------------------------
class ADSRNoteTrackingProcessor(Processor):
    """Track the notes with an HMM based on a model of attack, decay,
    sustain, release (ADSR) envelopes.

    Port of `madmom.features.notes.ADSRNoteTrackingProcessor`
    (`madmom-upstream/madmom/features/notes.py:315-456`). Decodes each of
    the input's pitch columns INDEPENDENTLY, one `HiddenMarkovModel.
    viterbi()` call per pitch (`ml/hmm.py`, unmodified Phase-1 machinery),
    using `notes_hmm.py`'s `ADSRStateSpace`/`ADSRTransitionModel`/
    `ADSRObservationModel`.

    Parameters
    ----------
    onset_prob : float, optional
        Transition probability to enter an onset state.
    note_prob : float, optional
        Transition probability to enter a sounding note state.
    offset_prob : float, optional
        Transition probability to enter an offset state.
    attack_length : float, optional
        Minimum required attack (i.e. onset activation required) length,
        in seconds.
    decay_length : float, optional
        Minimum required decay (i.e. note activation required) length,
        in seconds.
    release_length : float, optional
        Minimum required release (i.e. note activation required) length,
        in seconds.
    complete : bool, optional
        Require notes to transition all states (i.e. discard incomplete
        notes).
    onset_threshold : float, optional
        Require notes to have an onset activation greater or equal this
        threshold.
    note_threshold : float, optional
        Require notes to have a note activation greater equal this
        threshold.
    fps : float, optional
        Frames per second.
    pitch_offset : int, optional
        Pitch offset for the detected notes.

    References
    ----------
    .. [1] Rainer Kelz, Sebastian Boeck and Gerhard Widmer,
           "Deep Polyphonic ADSR Piano Note Transcription", Proceedings of
           the 44th International Conference on Acoustics, Speech and
           Signal Processing (ICASSP), 2019.
    """

    def __init__(self, onset_prob=0.8, note_prob=0.8, offset_prob=0.5,
                 attack_length=0.04, decay_length=0.04, release_length=0.02,
                 complete=True, onset_threshold=0.5, note_threshold=0.5,
                 fps=50, pitch_offset=21, **kwargs):
        # pylint: disable=unused-argument
        from .notes_hmm import (
            ADSRObservationModel, ADSRStateSpace, ADSRTransitionModel,
        )
        from ..ml.hmm import HiddenMarkovModel

        # state space
        self.st = ADSRStateSpace(attack_length=int(attack_length * fps),
                                  decay_length=int(decay_length * fps),
                                  release_length=int(release_length * fps))
        # transition model
        self.tm = ADSRTransitionModel(self.st, onset_prob=onset_prob,
                                       note_prob=note_prob,
                                       offset_prob=offset_prob)
        # observation model
        self.om = ADSRObservationModel(self.st)
        # instantiate a HMM
        self.hmm = HiddenMarkovModel(self.tm, self.om, None)
        # save variables
        self.complete = complete
        self.onset_threshold = onset_threshold
        self.note_threshold = note_threshold
        self.pitch_offset = pitch_offset
        self.fps = fps

    def process(self, activations, **kwargs):
        """Detect the notes in the given activation function.

        Parameters
        ----------
        activations : numpy array, shape (N, num_pitches, 3)
            Combined [note, onset, offset] activation function.

        Returns
        -------
        notes : numpy array, shape (M, 3)
            Detected notes `[seconds, pitches, duration]`.
        """
        # pylint: disable=arguments-differ, unused-argument
        notes = []
        note_path = np.arange(self.st.attack, self.st.release)
        # process each pitch individually
        for pitch in range(activations.shape[1]):
            # decode activations for this pitch with HMM
            with np.errstate(divide="ignore"):
                # ignore warnings when taking the log of 0
                path, _ = self.hmm.viterbi(activations[:, pitch, :])
            # extract HMM note segments
            segments = np.logical_and(path > self.st.attack,
                                       path < self.st.release)
            # extract start and end positions (transition points)
            idx = np.nonzero(np.diff(segments.astype(int)))[0]
            # add end if needed
            if len(idx) % 2 != 0:
                idx = np.append(idx, [len(activations)])
            # all sounding frames
            frames = activations[:, pitch, 0]
            # all frames with onset activations
            onsets = activations[:, pitch, 1]
            # iterate over all segments to decide which to keep
            for onset, offset in idx.reshape((-1, 2)):
                # extract note segment
                segment = path[onset:offset]
                # discard segment which do not contain the complete note path
                if self.complete and np.setdiff1d(note_path, segment).any():
                    continue
                # discard segments without a real note
                if frames[onset:offset].max() < self.note_threshold:
                    continue
                # discard segments without a real onset
                if onsets[onset:offset].max() < self.onset_threshold:
                    continue
                # append segment as note
                notes.append([onset / self.fps, pitch + self.pitch_offset,
                              (offset - onset) / self.fps])
        # if no notes are detected, return empty array
        if len(notes) == 0:
            return np.empty((0, 3))
        # sort the notes, convert timing information and return them
        return np.array(sorted(notes), ndmin=2)
