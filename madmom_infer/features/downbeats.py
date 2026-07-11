"""Reimplementation of madmom.features.downbeats --
`DBNDownBeatTrackingProcessor`, the dynamic Bayesian network downbeat tracker
that wires together beats_hmm.py's bar-length state space with ml/hmm.py's
Viterbi decoder to turn a beat-activation function into downbeat positions.
This is the top-level Phase-1 deliverable that the sibling all-in-one-infer
package needs; signal/stft/spectrogram/filters + beats_hmm + hmm are all
building blocks toward this one entry point.

Only `DBNDownBeatTrackingProcessor` (plus the one helper it needs,
`threshold_activations` from madmom.features.beats) is ported -- the RNN
activation-function classes (`RNNDownBeatProcessor` and friends) need the NN
runtime (madmom.ml.nn), which is out of scope until phase 2 per docs/DESIGN.md.

madmom's `__init__` builds one `HiddenMarkovModel` per bar length and its
`process()` decodes each with `self.map` -- `map` (builtin, sequential) unless
`num_threads` > 1, in which case it swaps in `multiprocessing.Pool(...).map`
(downbeats.py:230-235). all-in-one-infer (the only phase-1 caller, see
all-in-one-fix/src/allin1_infer/postprocessing/metrical.py:26-30) never passes
`num_threads`, so madmom's own default is already sequential -- this port just
always uses a plain Python loop over `self.hmms` and drops the
`multiprocessing.Pool` branch entirely (documented here so a future reader
doesn't go looking for it and wonder if it was missed).

Reads: madmom_infer/features/beats_hmm.py (BarStateSpace, BarTransitionModel,
RNNDownBeatTrackingObservationModel), madmom_infer/ml/hmm.py
(HiddenMarkovModel), madmom_infer/processors.py (Processor)
"""

import numpy as np

from madmom_infer.features.beats_hmm import (
    BarStateSpace, BarTransitionModel, RNNDownBeatTrackingObservationModel,
)
from madmom_infer.ml.hmm import HiddenMarkovModel
from madmom_infer.processors import Processor


def threshold_activations(activations, threshold):
    """
    Threshold activations to include only the main segment exceeding the given
    threshold (i.e. first to last time/index exceeding the threshold).

    Parameters
    ----------
    activations : numpy array
        Activations to be thresholded.
    threshold : float
        Threshold value.

    Returns
    -------
    activations : numpy array
        Thresholded activations
    start : int
        Index of the first activation exceeding the threshold.

    Notes
    -----
    This function can be used to extract the main segment of beat activations
    to track only the beats where the activations exceed the threshold.

    """
    first = last = 0
    # use only the activations > threshold
    idx = np.nonzero(activations >= threshold)[0]
    if idx.any():
        first = max(first, np.min(idx))
        last = min(len(activations), np.max(idx) + 1)
    # return thresholded activations segment and first index
    return activations[first:last], first


class DBNDownBeatTrackingProcessor(Processor):
    """
    Downbeat tracking with RNNs and a dynamic Bayesian network (DBN)
    approximated by a Hidden Markov Model (HMM).

    Parameters
    ----------
    beats_per_bar : int or list
        Number of beats per bar to be modeled. Can be either a single number
        or a list or array with bar lengths (in beats).
    min_bpm : float or list, optional
        Minimum tempo used for beat tracking [bpm]. If a list is given, each
        item corresponds to the number of beats per bar at the same position.
    max_bpm : float or list, optional
        Maximum tempo used for beat tracking [bpm]. If a list is given, each
        item corresponds to the number of beats per bar at the same position.
    num_tempi : int or list, optional
        Number of tempi to model; if set, limit the number of tempi and use a
        log spacing, otherwise a linear spacing. If a list is given, each
        item corresponds to the number of beats per bar at the same position.
    transition_lambda : float or list, optional
        Lambda for the exponential tempo change distribution (higher values
        prefer a constant tempo from one beat to the next one).  If a list is
        given, each item corresponds to the number of beats per bar at the
        same position.
    observation_lambda : int, optional
        Split one (down-)beat period into `observation_lambda` parts, the first
        representing (down-)beat states and the remaining non-beat states.
    threshold : float, optional
        Threshold the RNN (down-)beat activations before Viterbi decoding.
    correct : bool, optional
        Correct the beats (i.e. align them to the nearest peak of the
        (down-)beat activation function).
    fps : float, optional
        Frames per second.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "Joint Beat and Downbeat Tracking with Recurrent Neural Networks"
           Proceedings of the 17th International Society for Music Information
           Retrieval Conference (ISMIR), 2016.

    Examples
    --------
    Create a DBNDownBeatTrackingProcessor. The returned array represents the
    positions of the beats and their position inside the bar. The position is
    given in seconds, thus the expected sampling rate is needed. The position
    inside the bar follows the natural counting and starts at 1.

    The number of beats per bar which should be modelled must be given, all
    other parameters (e.g. tempo range) are optional but must have the same
    length as `beats_per_bar`, i.e. must be given for each bar length.

    >>> proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom_infer.features.downbeats.DBNDownBeatTrackingProcessor object at 0x...>

    Call this DBNDownBeatTrackingProcessor with the beat activation function
    (shape (N, 2), columns 'beat'/'downbeat') to obtain the beat positions.

    >>> proc(act)  # doctest: +SKIP
    array([[0.09, 1. ],
           [0.45, 2. ],
           ...,
           [2.14, 3. ],
           [2.49, 4. ]])

    """

    MIN_BPM = 55.
    MAX_BPM = 215.
    NUM_TEMPI = 60
    TRANSITION_LAMBDA = 100
    OBSERVATION_LAMBDA = 16
    THRESHOLD = 0.05
    CORRECT = True

    def __init__(self, beats_per_bar, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 num_tempi=NUM_TEMPI, transition_lambda=TRANSITION_LAMBDA,
                 observation_lambda=OBSERVATION_LAMBDA, threshold=THRESHOLD,
                 correct=CORRECT, fps=None, **kwargs):
        # pylint: disable=unused-argument
        # expand arguments to arrays
        beats_per_bar = np.array(beats_per_bar, ndmin=1)
        min_bpm = np.array(min_bpm, ndmin=1)
        max_bpm = np.array(max_bpm, ndmin=1)
        num_tempi = np.array(num_tempi, ndmin=1)
        transition_lambda = np.array(transition_lambda, ndmin=1)
        # make sure the other arguments are long enough by repeating them
        if len(min_bpm) != len(beats_per_bar):
            min_bpm = np.repeat(min_bpm, len(beats_per_bar))
        if len(max_bpm) != len(beats_per_bar):
            max_bpm = np.repeat(max_bpm, len(beats_per_bar))
        if len(num_tempi) != len(beats_per_bar):
            num_tempi = np.repeat(num_tempi, len(beats_per_bar))
        if len(transition_lambda) != len(beats_per_bar):
            transition_lambda = np.repeat(transition_lambda,
                                          len(beats_per_bar))
        if not (len(min_bpm) == len(max_bpm) == len(num_tempi) ==
                len(beats_per_bar) == len(transition_lambda)):
            raise ValueError('`min_bpm`, `max_bpm`, `num_tempi`, `num_beats` '
                             'and `transition_lambda` must all have the same '
                             'length.')
        # Note: madmom supports a `num_threads` kwarg that swaps in a
        # multiprocessing.Pool(...).map for `self.map`; this port always
        # decodes the (2, for beats_per_bar=[3, 4]) bar-length HMMs
        # sequentially with the builtin `map` -- see this module's docstring
        # for why that already matches madmom's own default behavior.
        # convert timing information to construct a beat state space
        min_interval = 60. * fps / max_bpm
        max_interval = 60. * fps / min_bpm
        # model the different bar lengths
        self.hmms = []
        for b, beats in enumerate(beats_per_bar):
            st = BarStateSpace(beats, min_interval[b], max_interval[b],
                               num_tempi[b])
            tm = BarTransitionModel(st, transition_lambda[b])
            om = RNNDownBeatTrackingObservationModel(st, observation_lambda)
            self.hmms.append(HiddenMarkovModel(tm, om))
        # save variables
        self.beats_per_bar = beats_per_bar
        self.threshold = threshold
        self.correct = correct
        self.fps = fps

    def process(self, activations, **kwargs):
        """
        Detect the (down-)beats in the given activation function.

        Parameters
        ----------
        activations : numpy array, shape (num_frames, 2)
            Activation function with probabilities corresponding to beats
            and downbeats given in the first and second column, respectively.

        Returns
        -------
        beats : numpy array, shape (num_beats, 2)
            Detected (down-)beat positions [seconds] and beat numbers.

        """
        # pylint: disable=arguments-differ
        # use only the activations > threshold (init offset to be added later)
        first = 0
        if self.threshold:
            activations, first = threshold_activations(activations,
                                                       self.threshold)
        # return no beats if no activations given / remain after thresholding
        if not activations.any():
            return np.empty((0, 2))
        # (sequential) decoding of the activations with each bar-length HMM
        results = [hmm.viterbi(activations) for hmm in self.hmms]
        # choose the best HMM (highest log probability)
        best = int(np.argmax([r[1] for r in results]))
        # the best path through the state space
        path, _ = results[best]
        # the state space and observation model of the best HMM
        st = self.hmms[best].transition_model.state_space
        om = self.hmms[best].observation_model
        # the positions inside the pattern (0..num_beats)
        positions = st.state_positions[path]
        # corresponding beats (add 1 for natural counting)
        beat_numbers = positions.astype(int) + 1
        if self.correct:
            beats = np.empty(0, dtype=int)
            # for each detection determine the "beat range", i.e. states where
            # the pointers of the observation model are >= 1
            beat_range = om.pointers[path] >= 1
            # if there aren't any in the beat range, there are no beats
            if not beat_range.any():
                return np.empty((0, 2))
            # get all change points between True and False (cast to int before)
            idx = np.nonzero(np.diff(beat_range.astype(int)))[0] + 1
            # if the first frame is in the beat range, add a change at frame 0
            if beat_range[0]:
                idx = np.r_[0, idx]
            # if the last frame is in the beat range, append the length of the
            # array
            if beat_range[-1]:
                idx = np.r_[idx, beat_range.size]
            # iterate over all regions
            if idx.any():
                for left, right in idx.reshape((-1, 2)):
                    # pick the frame with the highest activations value
                    # Note: we look for both beats and down-beat activations;
                    #       since np.argmax works on the flattened array, we
                    #       need to divide by 2
                    peak = np.argmax(activations[left:right]) // 2 + left
                    beats = np.hstack((beats, peak))
        else:
            # transitions are the points where the beat numbers change
            beats = np.nonzero(np.diff(beat_numbers))[0] + 1
        # return the beat positions (converted to seconds) and beat numbers
        return np.vstack(((beats + first) / float(self.fps),
                          beat_numbers[beats])).T
