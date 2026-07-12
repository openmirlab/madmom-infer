"""Reimplementation of madmom.features.downbeats --
`DBNDownBeatTrackingProcessor`, the dynamic Bayesian network downbeat tracker
that wires together beats_hmm.py's bar-length state space with ml/hmm.py's
Viterbi decoder to turn a beat-activation function into downbeat positions,
PLUS (Phase 2) `RNNDownBeatProcessor`, the RNN-ensemble pre-processing +
inference chain that PRODUCES that beat-activation function from raw audio.
Chained together (`RNNDownBeatProcessor()(wav_path)` ->
`DBNDownBeatTrackingProcessor(...)`) these are the full audio-in,
beat/downbeat-times-out Phase-2 acceptance target.

Phase-1 ported only `DBNDownBeatTrackingProcessor` (plus the one helper it
needs, `threshold_activations` from madmom.features.beats) -- the RNN
activation-function classes needed the NN runtime (madmom.ml.nn), out of
scope until Phase 2. Phase 2 adds `RNNDownBeatProcessor`
(`madmom-upstream/madmom/features/downbeats.py:30-95`): it builds a
`ParallelProcessor` of 3 `SequentialProcessor`s (one per frame size
1024/2048/4096, each `frames -> stft -> filt -> spec -> diff`), `hstack`s
their outputs, feeds that through a `NeuralNetworkEnsemble` of madmom's 8
`downbeats_blstm_[1-8].pkl` models (`madmom_infer/models.py`), and drops the
"non-beat" column. `RNNBarProcessor`/`DOWNBEATS_BGRU` (madmom's alternative
BGRU-based downbeat model + separate rhythmic/harmonic feature split) is
explicitly OUT of Phase-2 scope -- `RNNDownBeatProcessor`/`DOWNBEATS_BLSTM`
is the one target this phase proves end-to-end.

madmom's `__init__` builds one `HiddenMarkovModel` per bar length and its
`process()` decodes each with `self.map` -- `map` (builtin, sequential) unless
`num_threads` > 1, in which case it swaps in `multiprocessing.Pool(...).map`
(downbeats.py:230-235). all-in-one-infer (the only phase-1 caller, see
all-in-one-fix/src/allin1_infer/postprocessing/metrical.py:26-30) never passes
`num_threads`, so madmom's own default is already sequential -- this port just
always uses a plain Python loop over `self.hmms` and drops the
`multiprocessing.Pool` branch entirely (documented here so a future reader
doesn't go looking for it and wonder if it was missed).

Wave 4c adds `SyncronizeFeaturesProcessor` (pure numpy, no NN weights --
average feature frames into per-beat-subdivision bins) and `RNNBarProcessor`
(madmom's alternative, GRU-based downbeat model:
`madmom-upstream/madmom/features/downbeats.py:915-1035`), completing the
`GRULayer`/`GRUCell`/`DOWNBEATS_BGRU` scope-addition the 4.0 audit flagged
(CLAUDE.md's corrections section). **`RNNBarProcessor` cannot be
INSTANTIATED end-to-end from raw audio in this wave** -- its `__init__`
needs `audio/chroma.py`'s `CLPChromaProcessor` for its harmonic feature
branch, which is TO-PORT(4d), not yet ported (confirmed by reading
`RNNBarProcessor.__init__` directly, `downbeats.py:965/980`; the audit
table's own `features/downbeats.py` row already flagged this exact
dependency). The class is ported VERBATIM regardless (matching upstream's
own structure exactly, including the `from ..audio.chroma import
CLPChromaProcessor` import) -- it simply raises `ImportError` on
construction until 4d lands `CLPChromaProcessor`, same as it would in any
partially-built environment. What CAN and IS proven bit-identical this wave
is the part that actually needed `GRULayer`/`GRUCell`: the `DOWNBEATS_BGRU`
`NeuralNetworkEnsemble` forward pass itself, fed real madmom's own captured
intermediate beat-synchronized features as a golden fixture (see
`tools/generate_beat_tempo_fixtures.py`/`tests/test_downbeats_rnn.py`) --
this is the genuine "does this port's GRU implementation match real
madmom's" question, answered directly, without requiring this project's own
(not-yet-existing) `CLPChromaProcessor` to reproduce the harmonic features
from scratch.

Reads: madmom_infer/features/beats_hmm.py (BarStateSpace, BarTransitionModel,
RNNDownBeatTrackingObservationModel), madmom_infer/ml/hmm.py
(HiddenMarkovModel), madmom_infer/processors.py (Processor, ParallelProcessor,
SequentialProcessor), madmom_infer/audio/{signal,stft,spectrogram}.py (the
pre-processing cascade), madmom_infer/ml/nn/__init__.py (NeuralNetworkEnsemble),
madmom_infer/models.py (DOWNBEATS_BLSTM/DOWNBEATS_BGRU download); read by:
madmom_infer/features/tempo.py does NOT read this file (only features/beats.py's
DBNBeatTrackingProcessor, see that module).
"""

import warnings

import numpy as np

from madmom_infer.audio.signal import FramedSignalProcessor, SignalProcessor
from madmom_infer.audio.spectrogram import (
    FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
    SpectrogramDifferenceProcessor,
)
from madmom_infer.audio.stft import ShortTimeFourierTransformProcessor
from madmom_infer.features.beats_hmm import (
    BarStateSpace, BarTransitionModel, RNNDownBeatTrackingObservationModel,
)
from madmom_infer.ml.hmm import HiddenMarkovModel
from madmom_infer.ml.nn import NeuralNetworkEnsemble
from madmom_infer.processors import (
    ParallelProcessor, Processor, SequentialProcessor,
)


class RNNDownBeatProcessor(SequentialProcessor):
    """Joint beat/downbeat activation function from an ensemble of RNNs.

    Port of `madmom.features.downbeats.RNNDownBeatProcessor`
    (`madmom-upstream/madmom/features/downbeats.py:30-95`). Builds:

    1. `SignalProcessor(num_channels=1, sample_rate=44100)` -- downmix and
       (if needed) resample to 44.1kHz mono. **Resampling a differently-
       rated input file is NOT implemented** (`madmom_infer.audio.signal`'s
       `Signal` has no ffmpeg-backed `resample()`, see that module's
       header) -- callers must supply 44.1kHz audio.
    2. A `ParallelProcessor` of 3 `SequentialProcessor`s, one per frame size
       in `[1024, 2048, 4096]` with `num_bands` in `[3, 6, 12]`: each is
       `FramedSignalProcessor(frame_size, fps=100)` ->
       `ShortTimeFourierTransformProcessor()` ->
       `FilteredSpectrogramProcessor(num_bands, fmin=30, fmax=17000,
       norm_filters=True)` -> `LogarithmicSpectrogramProcessor(mul=1, add=1)`
       -> `SpectrogramDifferenceProcessor(diff_ratio=0.5,
       positive_diffs=True, stack_diffs=np.hstack)`.
    3. `np.hstack` the 3 branches' outputs into one per-frame feature vector.
    4. `NeuralNetworkEnsemble.load(DOWNBEATS_BLSTM)` (`madmom_infer/models.py`)
       -- averages 8 BLSTM networks' 3-class (`[non-beat, beat, downbeat]`)
       softmax outputs.
    5. `np.delete(..., obj=0, axis=1)` -- drop the "non-beat" column, leaving
       the `(num_frames, 2)` `[beat, downbeat]` activation array
       `DBNDownBeatTrackingProcessor` (below) expects.
    """

    def __init__(self, **kwargs):
        from functools import partial

        from madmom_infer.models import downbeats_blstm

        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        multi = ParallelProcessor([])
        frame_sizes = [1024, 2048, 4096]
        num_bands = [3, 6, 12]
        for frame_size, bands in zip(frame_sizes, num_bands):
            frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
            stft = ShortTimeFourierTransformProcessor()
            filt = FilteredSpectrogramProcessor(
                num_bands=bands, fmin=30, fmax=17000, norm_filters=True)
            spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
            diff = SpectrogramDifferenceProcessor(
                diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
        pre_processor = SequentialProcessor((sig, multi, np.hstack))
        nn = NeuralNetworkEnsemble.load(downbeats_blstm(), **kwargs)
        act = partial(np.delete, obj=0, axis=1)
        super().__init__((pre_processor, nn, act))


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


class SyncronizeFeaturesProcessor(Processor):
    """Synchronize features to beats: divide a beat interval into
    `beat_subdivisions` divisions, then average all feature values that
    fall into each subdivision (0 if none do).

    Verbatim port of `madmom.features.downbeats.SyncronizeFeaturesProcessor`
    (`madmom-upstream/madmom/features/downbeats.py:824-912`). Wave 4c, part
    of the `RNNBarProcessor`/`GRULayer` scope addition (see this module's
    header). Pure numpy, no NN weights -- independently verifiable without
    `RNNBarProcessor`'s own blocked (see header) harmonic-feature branch.
    """

    def __init__(self, beat_subdivisions, fps, **kwargs):
        # pylint: disable=unused-argument
        self.beat_subdivisions = beat_subdivisions
        self.fps = fps

    def process(self, data, **kwargs):
        """Synchronize features to beats.

        Parameters
        ----------
        data : tuple (features, beats)
            Tuple of two numpy arrays, the first containing features to be
            synchronized and the second the beat times.

        Returns
        -------
        numpy array (num_beats - 1, beat_subdivisions, features_dim)
            Beat-synchronous features.
        """
        # pylint: disable=arguments-differ, unused-argument
        features, beats = data
        if beats.size == 0:
            return np.array([]), np.array([])
        if beats.ndim > 1:
            beats = beats[:, 0]
        while (float(len(features)) / self.fps) < beats[-1]:
            beats = beats[:-1]
            warnings.warn("Beat sequence too long compared to features.")
        num_beats = len(beats)
        features = np.array(features.T, copy=False, ndmin=2).T
        feat_dim = features.shape[-1]
        beat_features = np.zeros(
            (num_beats - 1, self.beat_subdivisions, feat_dim))
        beat_start = int(max(0, np.floor((beats[0] - 0.02) * self.fps)))
        for i in range(num_beats - 1):
            beat_duration = beats[i + 1] - beats[i]
            offset = 0.5 * beat_duration / self.beat_subdivisions
            offset = np.min([offset, 0.05])
            beat_end = int(np.floor((beats[i + 1] - offset) * self.fps))
            subdiv = np.floor(np.linspace(0, self.beat_subdivisions,
                                          beat_end - beat_start,
                                          endpoint=False))
            beat = features[beat_start:beat_end]
            subdiv_features = [beat[subdiv == div] for div in
                               range(self.beat_subdivisions)]
            beat_features[i, :, :] = np.array(
                [np.mean(x, axis=0) for x in subdiv_features])
            beat_start = beat_end
        return beat_features


class RNNBarProcessor(Processor):
    """Retrieve a downbeat activation function from a signal and
    pre-determined beat positions by obtaining beat-synchronous harmonic and
    percussive features which are processed with a GRU-RNN.

    Verbatim port of `madmom.features.downbeats.RNNBarProcessor`
    (`madmom-upstream/madmom/features/downbeats.py:915-1035`). Wave 4c --
    **see this module's header: cannot be instantiated end-to-end from raw
    audio until 4d ports `audio/chroma.py`'s `CLPChromaProcessor`** (its
    `__init__` needs it for `self.harm_feat`); the `GRULayer`/`GRUCell`
    forward pass this class exists to exercise is instead proven bit-exact
    via a golden intermediate-feature fixture, not a full audio-in run --
    see `tests/test_downbeats_rnn.py`.
    """

    def __init__(self, beat_subdivisions=(4, 2), fps=100, **kwargs):
        # pylint: disable=unused-argument
        from madmom_infer.audio.chroma import CLPChromaProcessor
        from madmom_infer.models import downbeats_bgru

        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        frames = FramedSignalProcessor(frame_size=2048, fps=fps)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        spec = FilteredSpectrogramProcessor(
            num_bands=6, fmin=30., fmax=17000., norm_filters=True)
        log_spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
        diff = SpectrogramDifferenceProcessor(
            diff_ratio=0.5, positive_diffs=True)
        self.perc_feat = SequentialProcessor(
            (sig, frames, stft, spec, log_spec, diff))
        self.harm_feat = CLPChromaProcessor(
            fps=fps, fmin=27.5, fmax=4200., compression_factor=100,
            norm=True, threshold=0.001)
        self.perc_beat_sync = SyncronizeFeaturesProcessor(
            beat_subdivisions[0], fps=fps, **kwargs)
        self.harm_beat_sync = SyncronizeFeaturesProcessor(
            beat_subdivisions[1], fps=fps, **kwargs)
        bgru = downbeats_bgru()
        self.perc_nn = NeuralNetworkEnsemble.load(bgru[0], **kwargs)
        self.harm_nn = NeuralNetworkEnsemble.load(bgru[1], **kwargs)

    def process(self, data, **kwargs):
        """Retrieve a downbeat activation function from a signal and beat
        positions.

        Parameters
        ----------
        data : tuple
            Tuple containing a signal or file (handle) and corresponding
            beat times [seconds].

        Returns
        -------
        numpy array, shape (num_beats, 2)
            Beat positions (first column) and the corresponding downbeat
            activations (second column).
        """
        # pylint: disable=arguments-differ, unused-argument
        signal, beats = data
        perc = self.perc_feat(signal)
        harm = self.harm_feat(signal)
        perc_synced = self.perc_beat_sync((perc, beats))
        harm_synced = self.harm_beat_sync((harm, beats))
        perc = self.perc_nn(perc_synced.reshape((len(perc_synced), -1)))
        harm = self.harm_nn(harm_synced.reshape((len(harm_synced), -1)))
        act = np.mean([perc, harm], axis=0)
        act = np.append(act, np.ones(1) * np.nan)
        return np.vstack((beats, act)).T
