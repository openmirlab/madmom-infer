"""Reimplementation of madmom.features.beats -- the beat-only counterpart to
`madmom_infer/features/downbeats.py`'s joint beat/downbeat machinery. Wave
4c ported `RNNBeatProcessor`, `DBNBeatTrackingProcessor`,
`MultiModelSelectionProcessor`; Wave 4f (this addition) ports the rest of
the module -- `detect_beats`, `BeatTrackingProcessor`,
`BeatDetectionProcessor`, `CRFBeatDetectionProcessor` -- closing the
audit-table gap 4c itself flagged (`CRFBeatDetectionProcessor` needs
`BeatTrackingProcessor` as a base class and a numpy port of
`features/beats_crf.pyx`, neither of which existed yet in 4c). See
CLAUDE.md's audit table, `features/beats.py` rows.

Wave 4c ported exactly the 3 classes the 4.0 audit table's `features/
beats.py` row marks TO-PORT(4c) -- `RNNBeatProcessor`,
`DBNBeatTrackingProcessor`, `MultiModelSelectionProcessor` (pickle refs
confirm no new NN layer classes beyond Phase-2's LSTM/BLSTM set, per that
row's own note).

**Wave 4f addition**: `detect_beats` (upstream's recursive, tempo-driven
look-aside beat-alignment helper, `beats.py:301-382`), `BeatTrackingProcessor`
(look-aside/look-ahead beat tracking around a locally-estimated tempo,
`beats.py:385-573`), `BeatDetectionProcessor` (`BeatTrackingProcessor`
subclass with `look_ahead=None`, i.e. one GLOBAL tempo for the whole piece,
`beats.py:575-637`), and `CRFBeatDetectionProcessor` (`BeatTrackingProcessor`
subclass that replaces `detect_beats`'s heuristic alignment with a proper
CRF Viterbi decode over several candidate intervals, `beats.py:666-841`) --
all verbatim ports. `CRFBeatDetectionProcessor` needs `features/
beats_crf.py`'s numpy-ported `best_sequence` (this wave's own CRF Viterbi
port, see that module's header for the bit-identity findings). Its
`num_threads`-driven `multiprocessing.Pool` dispatch (`beats.py:734-736`) is
DROPPED, matching `madmom_infer/processors.py`'s stated permanent exclusion
of multiprocessing plumbing (`self.map` is always the plain builtin `map`,
sequential over the candidate intervals) -- correctness, not throughput, is
this project's goal (same precedent `processors.py`'s module header states
for `ParallelProcessor`).

**`threshold_activations` is intentionally NOT redefined here** -- real
madmom's `features/beats.py` and `features/downbeats.py` each define their
own (textually identical) copy; this port already has one, added in Phase 2
(`madmom_infer/features/downbeats.py`), and this module imports/reuses it
rather than duplicating it (see the 4.0 audit table's `TCNBeatProcessor,
detect_beats, threshold_activations` EXCLUDE row: "`threshold_activations`
itself is already ported ... and reused, not duplicated").

**`DBNBeatTrackingProcessor` is OFFLINE-ONLY** -- upstream subclasses
`OnlineProcessor` (`process_offline`/`process_online`/`reset`, a
visualisation branch, and online-mode-only constructor state), but this
project's processors are offline/whole-clip only (`madmom_infer/
processors.py`'s module header: `OnlineProcessor`/streaming machinery is a
stated permanent exclusion, same precedent as `features/onsets.py`'s
`OnsetPeakPickingProcessor`). This port is a plain `Processor` wired
directly to upstream's `process_offline` (`beats.py:1001-1062`) -- the
`online=True` constructor flag, `process_online`/`process_forward`,
`reset`, and the visualisation state/branch are dropped entirely, not
silently stubbed. An `online` keyword is still silently ACCEPTED (via
`**kwargs`, discarded) purely so `features/tempo.py`'s
`DBNTempoHistogramProcessor.__init__` -- which forwards its own `online`
kwarg to a `DBNBeatTrackingProcessor(...)` it builds -- doesn't need a
special case; since `TempoEstimationProcessor`/`TempoHistogramProcessor` are
themselves offline-only for the same reason (see `features/tempo.py`'s
module header), that forwarded value is always `False` in practice.

**Still EXCLUDED (see CLAUDE.md's 4.0 audit table corrections)**:
`TCNBeatProcessor`/`_tcn_beat_processor_pad`/the TCN-specific parts of
`detect_beats` -- `BEATS_TCN` is not shipped by a real madmom install.
`detect_beats` itself is NOT TCN-specific (it's `BeatTrackingProcessor`'s
own helper, ported below in full) -- only its `TCNBeatProcessor`-only
callers stay excluded.

Reads: madmom_infer/audio/{signal,stft,spectrogram}.py (the pre-processing
cascade), madmom_infer/features/beats_crf.py (best_sequence, this wave's own
numpy CRF Viterbi port -- CRFBeatDetectionProcessor), madmom_infer/features/
beats_hmm.py (BeatStateSpace, BeatTransitionModel,
RNNBeatTrackingObservationModel), madmom_infer/features/downbeats.py
(threshold_activations, reused not duplicated), madmom_infer/features/
tempo.py (TempoEstimationProcessor, lazily imported -- BeatTrackingProcessor's
default tempo estimator), madmom_infer/ml/hmm.py (HiddenMarkovModel),
madmom_infer/ml/nn/__init__.py (NeuralNetworkEnsemble, average_predictions),
madmom_infer/models.py (BEATS_LSTM/BEATS_BLSTM download), madmom_infer/
processors.py (Processor, ParallelProcessor, SequentialProcessor); read by:
madmom_infer/features/tempo.py (DBNTempoHistogramProcessor).
"""

import sys

import numpy as np
from scipy.signal import argrelmin

from madmom_infer.audio.signal import (
    FramedSignalProcessor, SignalProcessor, signal_frame,
)
from madmom_infer.audio.signal import smooth as smooth_signal
from madmom_infer.audio.spectrogram import (
    FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
    SpectrogramDifferenceProcessor,
)
from madmom_infer.audio.stft import ShortTimeFourierTransformProcessor
from madmom_infer.features.beats_hmm import (
    BeatStateSpace, BeatTransitionModel, RNNBeatTrackingObservationModel,
)
from madmom_infer.features.downbeats import threshold_activations
from madmom_infer.ml.hmm import HiddenMarkovModel
from madmom_infer.ml.nn import NeuralNetworkEnsemble, average_predictions
from madmom_infer.processors import (
    ParallelProcessor, Processor, SequentialProcessor,
)


class RNNBeatProcessor(SequentialProcessor):
    """Beat activation function from an ensemble of RNNs.

    Port of `madmom.features.beats.RNNBeatProcessor` (`beats.py:23-112`).
    `online=True` selects the causal, unidirectional `BEATS_LSTM` ensemble
    (single 2048-sample frame size, 12 bands); the default `online=False`
    selects the bidirectional `BEATS_BLSTM` ensemble (3 frame sizes
    1024/2048/4096, 6 bands each) -- same offline-compatibility shape as
    `features/onsets.py`'s `RNNOnsetProcessor` (the `online` flag only
    changes which pretrained weights/frame sizes are used, not any actual
    streaming behavior). `post_processor=None` returns the list of all
    per-network predictions (for `MultiModelSelectionProcessor`); the
    default `average_predictions` averages them into one array, matching
    upstream's own default. `nn_files`, if given, overrides the model list
    entirely (not in upstream -- matches this project's own
    `CNNKeyRecognitionProcessor`/`RNNOnsetProcessor` convention, used by the
    cross-BLAS test to point at local `.pkl` copies).
    """

    def __init__(self, post_processor=average_predictions, online=False,
                 nn_files=None, **kwargs):
        from madmom_infer.models import beats_blstm, beats_lstm

        if online:
            model_files = nn_files or beats_lstm()
            frame_sizes = [2048]
            num_bands = 12
        else:
            model_files = nn_files or beats_blstm()
            frame_sizes = [1024, 2048, 4096]
            num_bands = 6

        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        multi = ParallelProcessor([])
        for frame_size in frame_sizes:
            frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
            stft = ShortTimeFourierTransformProcessor()  # caching FFT window
            filt = FilteredSpectrogramProcessor(
                num_bands=num_bands, fmin=30, fmax=17000, norm_filters=True)
            spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
            diff = SpectrogramDifferenceProcessor(
                diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
        pre_processor = SequentialProcessor((sig, multi, np.hstack))

        nn = NeuralNetworkEnsemble.load(
            model_files, ensemble_fn=post_processor, **kwargs)
        super().__init__((pre_processor, nn))


class MultiModelSelectionProcessor(Processor):
    """Select the most suitable prediction (beat activation function) from
    multiple models' predictions, by comparing each to a reference
    prediction (the mean-squared error between the two) and keeping the
    closest match.

    Verbatim port of `madmom.features.beats.MultiModelSelectionProcessor`
    (`beats.py:186-297`). If `num_ref_predictions` is 0 or `None`, the
    reference is the average of ALL given predictions; otherwise it is the
    average of the first `num_ref_predictions` of them (which must then be
    the actual reference/ground-truth predictions, listed first).
    """

    def __init__(self, num_ref_predictions, **kwargs):
        # pylint: disable=unused-argument
        self.num_ref_predictions = num_ref_predictions

    def process(self, predictions, **kwargs):
        """Select the most suitable prediction from `predictions`, a list of
        beat activation functions (one per model)."""
        # pylint: disable=arguments-differ, unused-argument
        num_refs = self.num_ref_predictions
        if num_refs in (None, 0):
            reference = average_predictions(predictions)
        elif num_refs > 0:
            reference = average_predictions(predictions[:num_refs])
        else:
            raise ValueError(
                "`num_ref_predictions` must be positive or None, %s given"
                % num_refs)
        best_error = len(reference)
        best_prediction = np.empty(0)
        for prediction in predictions[num_refs:]:
            error = np.sum((prediction - reference) ** 2.)
            if error < best_error:
                best_prediction = prediction
                best_error = error
        return best_prediction.ravel()


class DBNBeatTrackingProcessor(Processor):
    """Beat tracking with RNNs and a dynamic Bayesian network (DBN)
    approximated by a Hidden Markov Model (HMM) -- the beat-only counterpart
    of `madmom_infer.features.downbeats.DBNDownBeatTrackingProcessor`,
    reusing the SAME `beats_hmm.py`/`ml.hmm.py` machinery via `BeatStateSpace`/
    `BeatTransitionModel`/`RNNBeatTrackingObservationModel` (single-bar-length
    state space, no `beats_per_bar` argument -- one HMM, not one per bar
    length).

    Port of `madmom.features.beats.DBNBeatTrackingProcessor`, OFFLINE ONLY
    -- see this module's header for why (`OnlineProcessor` is a stated
    permanent exclusion in this project). Wires directly to upstream's
    `process_offline` (`beats.py:1001-1062`).
    """

    MIN_BPM = 55.
    MAX_BPM = 215.
    NUM_TEMPI = None
    TRANSITION_LAMBDA = 100
    OBSERVATION_LAMBDA = 16
    THRESHOLD = 0
    CORRECT = True

    def __init__(self, min_bpm=MIN_BPM, max_bpm=MAX_BPM, num_tempi=NUM_TEMPI,
                 transition_lambda=TRANSITION_LAMBDA,
                 observation_lambda=OBSERVATION_LAMBDA, correct=CORRECT,
                 threshold=THRESHOLD, fps=None, **kwargs):
        # pylint: disable=unused-argument
        min_interval = 60. * fps / max_bpm
        max_interval = 60. * fps / min_bpm
        self.st = BeatStateSpace(min_interval, max_interval, num_tempi)
        self.tm = BeatTransitionModel(self.st, transition_lambda)
        self.om = RNNBeatTrackingObservationModel(self.st, observation_lambda)
        self.hmm = HiddenMarkovModel(self.tm, self.om, None)
        self.correct = correct
        self.threshold = threshold
        self.fps = fps
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm

    def process(self, activations, **kwargs):
        """Detect the beats in the given activation function with Viterbi
        decoding. Matches `DBNBeatTrackingProcessor.process_offline`
        (`beats.py:1001-1062`)."""
        # pylint: disable=arguments-differ, unused-argument
        beats = np.empty(0, dtype=int)
        first = 0
        if self.threshold:
            activations, first = threshold_activations(
                activations, self.threshold)
        if not activations.any():
            return beats
        path, _ = self.hmm.viterbi(activations)
        if not path.any():
            return beats
        if self.correct:
            beat_range = self.om.pointers[path]
            idx = np.nonzero(np.diff(beat_range))[0] + 1
            if beat_range[0]:
                idx = np.r_[0, idx]
            if beat_range[-1]:
                idx = np.r_[idx, beat_range.size]
            if idx.any():
                for left, right in idx.reshape((-1, 2)):
                    peak = np.argmax(activations[left:right]) + left
                    beats = np.hstack((beats, peak))
        else:
            beats = argrelmin(self.st.state_positions[path], mode="wrap")[0]
            beats = beats[self.om.pointers[path[beats]] == 1]
        return (beats + first) / float(self.fps)


# ---------------------------------------------------------------------------
# Wave 4f addition: detect_beats, BeatTrackingProcessor, BeatDetectionProcessor,
# CRFBeatDetectionProcessor -- see this module's header for why these were
# deferred out of Wave 4c.
# ---------------------------------------------------------------------------
def detect_beats(activations, interval, look_aside=0.2):
    """Detect the beats in `activations` given a fixed dominant `interval`,
    by recursively searching a Hamming-windowed neighborhood around each
    expected next-beat position (starting position is picked by trying all
    `interval` possible offsets and keeping the one with the highest total
    activation).

    Verbatim port of `madmom.features.beats.detect_beats`
    (`beats.py:301-382`).

    Parameters
    ----------
    activations : numpy array
        Beat activations.
    interval : int
        Look for the next beat each `interval` frames.
    look_aside : float
        Look this fraction of the `interval` to each side to detect the
        beats.

    Returns
    -------
    numpy array
        Beat positions [frames].

    Notes
    -----
    A Hamming window of `2 * look_aside * interval` is applied around the
    position where the beat is expected, to prefer beats closer to the
    centre.
    """
    # TODO: make this faster!
    sys.setrecursionlimit(len(activations))
    # always look at least 1 frame to each side
    frames_look_aside = max(1, int(interval * look_aside))
    win = np.hamming(2 * frames_look_aside)

    # list to be filled with beat positions from inside the recursive function
    positions = []

    def recursive(position):
        """Recursively detect the next beat, starting at `position`."""
        # detect the nearest beat around the actual position
        act = signal_frame(activations, position, frames_look_aside * 2, 1)
        # apply a filtering window to prefer beats closer to the centre
        act = np.multiply(act, win)
        # search max
        if np.argmax(act) > 0:
            # maximum found, take that position
            position = np.argmax(act) + position - frames_look_aside
        # add the found position
        positions.append(position)
        # go to the next beat, until end is reached
        if position + interval < len(activations):
            recursive(position + interval)
        else:
            return

    # calculate the beats for each start position (up to the interval length)
    sums = np.zeros(interval)
    for i in range(interval):
        positions = []
        # detect the beats for this start position
        recursive(i)
        # calculate the sum of the activations at the beat positions
        sums[i] = np.sum(activations[positions])
    # take the winning start position
    start_position = np.argmax(sums)
    # and calc the beats for this start position
    positions = []
    recursive(start_position)
    # return indices
    return np.array(positions)


class BeatTrackingProcessor(Processor):
    """Track the beats according to a previously determined (local) tempo,
    by iteratively aligning them around the estimated position.

    Verbatim port of `madmom.features.beats.BeatTrackingProcessor`
    (`beats.py:385-573`).

    Parameters
    ----------
    look_aside : float, optional
        Look this fraction of the estimated beat interval to each side of
        the assumed next beat position to look for the most likely position
        of the next beat.
    look_ahead : float, optional
        Look `look_ahead` seconds in both directions to determine the local
        tempo and align the beats accordingly.
    tempo_estimator : `madmom_infer.features.tempo.TempoEstimationProcessor`,
        optional
        Use this processor to estimate the (local) tempo. If `None`, a
        default tempo estimator is created and used.
    fps : float, optional
        Frames per second.
    kwargs : dict, optional
        Keyword arguments passed to
        `madmom_infer.features.tempo.TempoEstimationProcessor` if no
        `tempo_estimator` was given.

    Notes
    -----
    If `look_ahead` is not set, a constant tempo throughout the whole piece
    is assumed. If `look_ahead` is set, the local tempo (in a range +/-
    `look_ahead` seconds around the actual position) is estimated and then
    the next beat is tracked accordingly. This procedure is repeated from
    the new position to the end of the piece.
    """

    LOOK_ASIDE = 0.2
    LOOK_AHEAD = 10.0

    def __init__(self, look_aside=LOOK_ASIDE, look_ahead=LOOK_AHEAD,
                 fps=None, tempo_estimator=None, **kwargs):
        # save variables
        self.look_aside = look_aside
        self.look_ahead = look_ahead
        self.fps = fps
        # tempo estimator
        if tempo_estimator is None:
            # import the TempoEstimation here otherwise we have a loop
            from madmom_infer.features.tempo import TempoEstimationProcessor

            # create default tempo estimator
            tempo_estimator = TempoEstimationProcessor(fps=fps, **kwargs)
        self.tempo_estimator = tempo_estimator

    def process(self, activations, **kwargs):
        """Detect the beats in the given activation function.

        Parameters
        ----------
        activations : numpy array
            Beat activation function.

        Returns
        -------
        beats : numpy array
            Detected beat positions [seconds].
        """
        # pylint: disable=arguments-differ, unused-argument
        # smooth activations
        act_smooth = int(self.fps * self.tempo_estimator.act_smooth)
        activations = smooth_signal(activations, act_smooth)
        # TODO: refactor interval stuff to use TempoEstimation
        # if look_ahead is not defined, assume a global tempo
        if self.look_ahead is None:
            # create a interval histogram
            histogram = self.tempo_estimator.interval_histogram(activations)
            # get the dominant interval
            interval = self.tempo_estimator.dominant_interval(histogram)
            # detect beats based on this interval
            detections = detect_beats(activations, interval, self.look_aside)
        else:
            # allow varying tempo
            look_ahead_frames = int(self.look_ahead * self.fps)
            # detect the beats
            detections = []
            pos = 0
            # TODO: make this _much_ faster!
            while pos < len(activations):
                # look N frames around the actual position
                act = signal_frame(activations, pos, look_ahead_frames * 2, 1)
                # create a interval histogram
                histogram = self.tempo_estimator.interval_histogram(act)
                # get the dominant interval
                interval = self.tempo_estimator.dominant_interval(histogram)
                # add the offset (i.e. the new detected start position)
                positions = detect_beats(act, interval, self.look_aside)
                # correct the beat positions
                positions += pos - look_ahead_frames
                # remove all positions < already detected beats + min_interval
                next_pos = (detections[-1] + self.tempo_estimator.min_interval
                           if detections else 0)
                positions = positions[positions >= next_pos]
                # search the closest beat to the predicted beat position
                pos = positions[(np.abs(positions - pos)).argmin()]
                # append to the beats
                detections.append(pos)
                pos += interval

        # convert detected beats to a list of timestamps
        detections = np.array(detections) / float(self.fps)
        # remove beats with negative times and return them
        return detections[np.searchsorted(detections, 0):]


class BeatDetectionProcessor(BeatTrackingProcessor):
    """Detect beats according to a previously determined GLOBAL tempo, by
    iteratively aligning them around the estimated position (i.e.
    `BeatTrackingProcessor` with `look_ahead=None`).

    Verbatim port of `madmom.features.beats.BeatDetectionProcessor`
    (`beats.py:575-637`).

    Parameters
    ----------
    look_aside : float
        Look this fraction of the estimated beat interval to each side of
        the assumed next beat position to look for the most likely position
        of the next beat.
    fps : float, optional
        Frames per second.

    Notes
    -----
    A constant tempo throughout the whole piece is assumed.

    See Also
    --------
    :class:`BeatTrackingProcessor`
    """

    LOOK_ASIDE = 0.2

    def __init__(self, look_aside=LOOK_ASIDE, fps=None, **kwargs):
        super().__init__(look_aside=look_aside, look_ahead=None, fps=fps,
                         **kwargs)


def _process_crf(process_tuple):
    """Extract the best beat sequence for a piece.

    Verbatim port of `madmom.features.beats._process_crf`
    (`beats.py:639-663`) -- upstream's own docstring notes this proxy
    function exists to process different intervals in parallel via
    `multiprocessing`; this project's own `CRFBeatDetectionProcessor` always
    runs it sequentially (see this module's header), but the free-function
    shape is kept for a faithful, drop-in-compatible port.

    Parameters
    ----------
    process_tuple : tuple
        Tuple with (activations, dominant_interval, allowed deviation from
        the dominant interval per beat).

    Returns
    -------
    beats : numpy array
        Extracted beat positions [frames].
    log_prob : float
        Log probability of the beat sequence.
    """
    from madmom_infer.features.beats_crf import best_sequence

    return best_sequence(*process_tuple)


class CRFBeatDetectionProcessor(BeatTrackingProcessor):
    """Conditional Random Field Beat Detection.

    Tracks the beats according to a previously determined global tempo
    using a conditional random field (CRF) model.

    Verbatim port of `madmom.features.beats.CRFBeatDetectionProcessor`
    (`beats.py:666-841`) -- see this module's header for why its
    `num_threads`/`multiprocessing.Pool` dispatch was dropped in favor of
    always using the plain sequential builtin `map`.

    Parameters
    ----------
    interval_sigma : float, optional
        Allowed deviation from the dominant beat interval per beat.
    use_factors : bool, optional
        Use dominant interval multiplied by factors instead of intervals
        estimated by tempo estimator.
    num_intervals : int, optional
        Maximum number of estimated intervals to try.
    factors : list or numpy array, optional
        Factors of the dominant interval to try.

    References
    ----------
    .. [1] Filip Korzeniowski, Sebastian Böck and Gerhard Widmer,
           "Probabilistic Extraction of Beat Positions from a Beat
           Activation Function", Proceedings of the 15th International
           Society for Music Information Retrieval Conference (ISMIR),
           2014.
    """

    INTERVAL_SIGMA = 0.18
    USE_FACTORS = False
    FACTORS = np.array([0.5, 0.67, 1.0, 1.5, 2.0])
    NUM_INTERVALS = 5
    # tempo defaults
    MIN_BPM = 20
    MAX_BPM = 240
    ACT_SMOOTH = 0.09
    HIST_SMOOTH = 7

    def __init__(self, interval_sigma=INTERVAL_SIGMA, use_factors=USE_FACTORS,
                 num_intervals=NUM_INTERVALS, factors=FACTORS, **kwargs):
        super().__init__(**kwargs)
        # save parameters
        self.interval_sigma = interval_sigma
        self.use_factors = use_factors
        self.num_intervals = num_intervals
        self.factors = factors
        # Note: upstream computes `num_threads` from kwargs here to decide
        # whether to build a `multiprocessing.Pool(...).map` -- this
        # project always runs sequentially (see this module's header, no
        # multiprocessing.Pool), so that computation is dropped entirely
        # rather than kept as dead code.
        self.map = map

    def process(self, activations, **kwargs):
        """Detect the beats in the given activation function.

        Parameters
        ----------
        activations : numpy array
            Beat activation function.

        Returns
        -------
        numpy array
            Detected beat positions [seconds].
        """
        # pylint: disable=arguments-differ, unused-argument
        import itertools as it

        # estimate the tempo
        tempi = self.tempo_estimator.process(activations)
        intervals = self.fps * 60.0 / tempi[:, 0]

        # compute possible intervals
        if self.use_factors:
            # use the dominant interval with different factors
            possible_intervals = [int(intervals[0] * f) for f in self.factors]
            possible_intervals = [
                i for i in possible_intervals
                if self.tempo_estimator.max_interval >= i >=
                self.tempo_estimator.min_interval
            ]
        else:
            # take the top n intervals from the tempo estimator
            possible_intervals = list(intervals[:self.num_intervals])

        # sort and start from the greatest interval
        possible_intervals.sort()
        possible_intervals = [int(i) for i in possible_intervals[::-1]]

        # smooth activations
        act_smooth = int(self.fps * self.tempo_estimator.act_smooth)
        activations = smooth_signal(activations, act_smooth)

        # since the original Cython code uses memory views, make sure the
        # activations are C-contiguous and of C-type float (np.float32)
        contiguous_act = np.ascontiguousarray(activations, dtype=np.float32)
        results = list(self.map(
            _process_crf,
            zip(it.repeat(contiguous_act), possible_intervals,
               it.repeat(self.interval_sigma)),
        ))

        # normalize their probabilities
        normalized_seq_probabilities = np.array(
            [r[1] / r[0].shape[0] for r in results])
        # pick the best one
        best_seq = results[normalized_seq_probabilities.argmax()][0]

        # convert the detected beat positions to seconds and return them
        return best_seq.astype(float) / self.fps
