"""Reimplementation of madmom.features.beats -- the beat-only counterpart to
`madmom_infer/features/downbeats.py`'s joint beat/downbeat machinery. Wave
4c of the complete-port campaign; see CLAUDE.md's audit table, `features/
beats.py` rows.

Ports exactly the 3 classes the 4.0 audit table's `features/beats.py` row
marks TO-PORT(4c) -- `RNNBeatProcessor`, `DBNBeatTrackingProcessor`,
`MultiModelSelectionProcessor` (pickle refs confirm no new NN layer classes
beyond Phase-2's LSTM/BLSTM set, per that row's own note).

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

**Also intentionally NOT ported this wave (see CLAUDE.md's 4.0 audit table
-- these rows are real, but land in OTHER waves, not 4c):**
`CRFBeatDetectionProcessor` (4f, needs a numpy port of `features/
beats_crf.pyx`), `TCNBeatProcessor`/`_tcn_beat_processor_pad`/the
TCN-specific parts of `detect_beats` (EXCLUDE, `BEATS_TCN` is not shipped by
a real madmom install -- see CLAUDE.md's 4.0 audit corrections).

**Found, not silently dropped -- an apparent audit-table gap.** Upstream's
`features/beats.py` also defines `BeatTrackingProcessor`,
`BeatDetectionProcessor`, and the free function `detect_beats` --
tempo-driven, look-aside/look-ahead beat alignment, NOT RNN/DBN-based, and
NOT actually TCN-specific despite the 4.0 audit table's `TCNBeatProcessor,
detect_beats, threshold_activations (TCN-specific parts)` EXCLUDE row
grouping `detect_beats` there (`detect_beats` is `BeatTrackingProcessor`'s
own helper in real madmom, unrelated to `TCNBeatProcessor` -- confirmed by
reading `beats.py:301-382`/`453-465` directly). Neither class appears in
ANY wave's TO-PORT row. `CRFBeatDetectionProcessor` (4f) actually subclasses
`BeatTrackingProcessor`, so 4f will need to port it anyway -- deferred there
rather than ported speculatively here, since no 4c target processor needs
it and this wave's own scope is `RNNBeatProcessor`/`DBNBeatTrackingProcessor`/
`MultiModelSelectionProcessor` per the audit table's explicit row.

Reads: madmom_infer/audio/{signal,stft,spectrogram}.py (the pre-processing
cascade), madmom_infer/features/beats_hmm.py (BeatStateSpace,
BeatTransitionModel, RNNBeatTrackingObservationModel), madmom_infer/
features/downbeats.py (threshold_activations, reused not duplicated),
madmom_infer/ml/hmm.py (HiddenMarkovModel), madmom_infer/ml/nn/__init__.py
(NeuralNetworkEnsemble, average_predictions), madmom_infer/models.py
(BEATS_LSTM/BEATS_BLSTM download), madmom_infer/processors.py (Processor,
ParallelProcessor, SequentialProcessor); read by: madmom_infer/features/
tempo.py (DBNTempoHistogramProcessor).
"""

import numpy as np
from scipy.signal import argrelmin

from madmom_infer.audio.signal import FramedSignalProcessor, SignalProcessor
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
