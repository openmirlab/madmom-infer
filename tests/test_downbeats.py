"""A/B and behavioral tests for madmom_infer.features.downbeats
(DBNDownBeatTrackingProcessor), the phase-1 top-level target.

Golden-fixture test uses tests/fixtures/dbn_downbeat.npz, recorded by
tests/generate_fixtures.py against real madmom's compiled
DBNDownBeatTrackingProcessor (all-in-one-infer's exact parameters:
beats_per_bar=[3, 4], fps=100 -- see all-in-one-fix/src/allin1_infer/
postprocessing/metrical.py:26-30). Also covers the pathological case flagged
by the fixtures workstream: one beats_per_bar hypothesis's HMM can decode to
an empty path / -inf log-probability (activation sequence doesn't fit that
meter at all), and DBNDownBeatTrackingProcessor.process() must tolerate this
(np.argmax over per-hypothesis log-probs naturally skips the -inf one) without
crashing, including the doubly-pathological case where *every* hypothesis
fails.

Reads: tests/fixtures/dbn_downbeat.npz; madmom_infer/features/downbeats.py
"""

import os
import time

import numpy as np

from madmom_infer.features.downbeats import (DBNDownBeatTrackingProcessor,
                                             threshold_activations)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def test_dbn_downbeat_exact_beat_times():
    d = np.load(os.path.join(FIXTURES_DIR, "dbn_downbeat.npz"))
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)

    result = proc(d["activations"])

    # derives from integer frame indices / fps, so exact equality is expected
    np.testing.assert_array_equal(result, d["result"])


def test_threshold_activations_basic():
    act = np.array([0.0, 0.02, 0.1, 0.2, 0.03, 0.0])
    thresholded, first = threshold_activations(act, 0.05)
    assert first == 2
    np.testing.assert_array_equal(thresholded, act[2:4])


def test_one_hypothesis_fails_other_succeeds_monkeypatched():
    # directly exercise the processor-level argmax-over-hypotheses skip: make
    # the beats_per_bar=3 HMM report total failure (empty path, -inf log_prob,
    # exactly what HiddenMarkovModel.viterbi returns on full failure -- see
    # test_hmm.py::test_viterbi_full_failure_returns_empty_path_and_neg_inf)
    # while the beats_per_bar=4 HMM decodes normally; process() must pick the
    # surviving (4/4) hypothesis and behave exactly as if only it existed.
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)

    d = np.load(os.path.join(FIXTURES_DIR, "dbn_downbeat.npz"))
    activations = d["activations"]

    baseline = DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)
    expected = baseline.process(activations)

    real_viterbi_4 = proc.hmms[1].viterbi

    def _always_fails(observations):
        return np.empty(0, dtype=np.uint32), -np.inf

    proc.hmms[0].viterbi = _always_fails
    proc.hmms[1].viterbi = real_viterbi_4

    result = proc.process(activations)

    np.testing.assert_array_equal(result, expected)


def test_all_hypotheses_fail_returns_empty_no_crash():
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)

    def _always_fails(observations):
        return np.empty(0, dtype=np.uint32), -np.inf

    for hmm in proc.hmms:
        hmm.viterbi = _always_fails

    activations = np.full((200, 2), 0.3)
    result = proc.process(activations)

    assert result.shape == (0, 2)


def test_no_activations_above_threshold_returns_empty():
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100,
                                        threshold=0.9)
    activations = np.full((100, 2), 0.01)
    result = proc.process(activations)
    assert result.shape == (0, 2)


def test_viterbi_perf_4beat_6000_frames(capsys):
    from madmom_infer.features.beats_hmm import (
        BarStateSpace, BarTransitionModel, RNNDownBeatTrackingObservationModel,
    )
    from madmom_infer.ml.hmm import HiddenMarkovModel

    d = np.load(os.path.join(FIXTURES_DIR, "perf_activation.npz"))
    activations = d["activations"]
    assert len(activations) == 6000

    min_interval = 60. * 100. / 215.
    max_interval = 60. * 100. / 55.
    st = BarStateSpace(4, min_interval, max_interval, 60)
    tm = BarTransitionModel(st, 100)
    om = RNNDownBeatTrackingObservationModel(st, observation_lambda=16)
    hmm = HiddenMarkovModel(tm, om)

    start = time.perf_counter()
    path, log_prob = hmm.viterbi(activations)
    elapsed = time.perf_counter() - start

    with capsys.disabled():
        print(f"\n[perf] viterbi() on 4/4 model ({st.num_states} states), "
             f"{len(activations)} frames: {elapsed:.3f}s")

    assert path.shape == (6000,)
    assert np.isfinite(log_prob)
