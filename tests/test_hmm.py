"""A/B tests for madmom_infer.ml.hmm against real madmom's compiled ml/hmm.pyx.

Fixtures are pre-recorded by tests/generate_fixtures.py, run once against the
compiled madmom install in the reference venv (madmom-reference/.venv --
see that script's docstring for exact provenance/regeneration instructions)
-- these tests never import
real madmom themselves, only the saved .npz arrays, so they run under
whatever numpy/Python this project's own environment uses.

Reads: tests/fixtures/toy_hmm.npz; madmom_infer/ml/hmm.py
"""

import os

import numpy as np
import pytest

from madmom_infer.ml.hmm import (DiscreteObservationModel, HiddenMarkovModel,
                                 TransitionModel)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture(scope="module")
def toy_hmm_fixture():
    return np.load(os.path.join(FIXTURES_DIR, "toy_hmm.npz"))


def test_docstring_example_exact():
    # madmom's own HiddenMarkovModel.viterbi/forward doctest example
    # (madmom/ml/hmm.pyx:407-433) -- hardcoded expected values, no dependency
    # on the madmom venv at all.
    tm = TransitionModel.from_dense([0, 1, 0, 1], [0, 0, 1, 1],
                                    [0.7, 0.3, 0.6, 0.4])
    om = DiscreteObservationModel(np.array([[0.2, 0.3, 0.5],
                                            [0.7, 0.1, 0.2]]))
    hmm = HiddenMarkovModel(tm, om)

    observations = [0, 0, 1, 1, 0, 0, 0, 2, 2]
    path, log_prob = hmm.viterbi(observations)

    expected_path = np.array([1, 1, 0, 0, 1, 1, 1, 0, 0], dtype=np.uint32)
    np.testing.assert_array_equal(path, expected_path)
    assert path.dtype == np.uint32
    assert log_prob == pytest.approx(-12.87489873725737, abs=1e-10)

    fwd = hmm.forward(observations)
    expected_fwd = np.array([
        [0.34667, 0.65333],
        [0.33171, 0.66829],
        [0.83814, 0.16186],
        [0.86645, 0.13355],
        [0.38502, 0.61498],
        [0.33539, 0.66461],
        [0.33063, 0.66937],
        [0.81179, 0.18821],
        [0.84231, 0.15769],
    ])
    np.testing.assert_allclose(fwd, expected_fwd, atol=5e-6)


def test_toy_hmm_viterbi_exact_path(toy_hmm_fixture):
    d = toy_hmm_fixture
    tm = TransitionModel(d["tm_states"], d["tm_pointers"], d["tm_probabilities"])
    om = DiscreteObservationModel(d["om_probs"])
    hmm = HiddenMarkovModel(tm, om)

    path, log_prob = hmm.viterbi(d["observations"])

    np.testing.assert_array_equal(path, d["viterbi_path"])
    assert path.dtype == np.uint32
    assert log_prob == pytest.approx(float(d["viterbi_log_prob"]), abs=1e-10)


def test_toy_hmm_forward_matches(toy_hmm_fixture):
    d = toy_hmm_fixture
    tm = TransitionModel(d["tm_states"], d["tm_pointers"], d["tm_probabilities"])
    om = DiscreteObservationModel(d["om_probs"])
    hmm = HiddenMarkovModel(tm, om)

    fwd = hmm.forward(d["observations"])

    np.testing.assert_allclose(fwd, d["forward"], rtol=1e-12, atol=1e-12)


def test_transition_model_csr_arrays_unchanged_by_roundtrip(toy_hmm_fixture):
    # from_dense(make_dense(...)) should round-trip to the same sparse CSR
    # representation (sanity check on make_dense/make_sparse, not just viterbi)
    d = toy_hmm_fixture
    tm = TransitionModel(d["tm_states"], d["tm_pointers"], d["tm_probabilities"])
    states, prev_states, probabilities = TransitionModel.make_dense(
        tm.states, tm.pointers, tm.probabilities)
    tm2 = TransitionModel.from_dense(states, prev_states, probabilities)
    np.testing.assert_array_equal(tm2.states, tm.states)
    np.testing.assert_array_equal(tm2.pointers, tm.pointers)
    np.testing.assert_allclose(tm2.probabilities, tm.probabilities, rtol=1e-12)


def test_viterbi_empty_segment_handling():
    # a state (state 2) with zero incoming transitions must decode to -inf
    # and never be selected, exercising the _segment_reduce empty-segment path
    states = np.array([0, 1], dtype=np.uint32)       # transitions into 0, 1
    pointers = np.array([0, 1, 2, 2], dtype=np.uint32)  # state 2: empty row
    probabilities = np.array([1.0, 1.0])
    # bincount-of-prev_states check requires outgoing prob sum to 1 for every
    # *origin* state referenced as a prev_state (0 and 1 both used once, each
    # with probability 1.0, so the check passes trivially)
    tm = TransitionModel(states, pointers, probabilities)

    class _TrivialOM:
        pointers = np.array([0, 1, 2], dtype=np.uint32)

        def log_densities(self, observations):
            return np.zeros((len(observations), 3))

    hmm = HiddenMarkovModel(tm, _TrivialOM(),
                            initial_distribution=np.array([0.5, 0.5, 0.0]))
    path, log_prob = hmm.viterbi([0, 0, 0])
    assert np.isfinite(log_prob)
    assert 2 not in path


def test_viterbi_full_failure_returns_empty_path_and_neg_inf():
    # If every state's density is -inf at some frame, the whole hypothesis
    # dies (previous_viterbi becomes all -inf and can never recover, since
    # -inf + anything stays -inf). madmom's hmm.pyx:566-571 special-cases
    # this: warn, and return (np.empty(0, dtype=uint32), -inf) instead of
    # backtracking through undefined pointers. This matters directly for
    # DBNDownBeatTrackingProcessor.process(), which runs one HMM per
    # beats_per_bar hypothesis and must tolerate one of them failing outright
    # (see this module's docstring / the coordinator's fixtures-workstream
    # finding) -- verified end-to-end in test_downbeats.py.
    states = np.array([0, 1, 1, 0], dtype=np.uint32)
    pointers = np.array([0, 2, 4], dtype=np.uint32)
    probabilities = np.array([0.5, 0.5, 0.5, 0.5])
    tm = TransitionModel(states, pointers, probabilities)

    class _KillFrameOM:
        pointers = np.array([0, 1], dtype=np.uint32)

        def log_densities(self, observations):
            # frame 1 (of 3) is -inf for every state -> whole path dies there
            # and can never recover in later frames
            dens = np.zeros((len(observations), 2))
            dens[1, :] = -np.inf
            return dens

    hmm = HiddenMarkovModel(tm, _KillFrameOM())
    with pytest.warns(RuntimeWarning, match="inf log probability"):
        path, log_prob = hmm.viterbi([0, 0, 0])

    assert np.isneginf(log_prob)
    assert path.shape == (0,)
    assert path.dtype == np.uint32


def _reference_viterbi(tm, om, initial_distribution, observations):
    """
    Unvectorized, literal transcription of madmom's Cython triple loop
    (hmm.pyx:526-585), including its exact `>` (not `>=`) tie-break and its
    "NaN never wins a comparison" behavior (native Python/IEEE754 float
    comparisons against NaN are always False, same as C). Used only as an
    independent correctness oracle in tests, not part of the shipped port.

    """
    num_states = tm.num_states
    num_observations = len(observations)
    om_densities = np.asarray(om.log_densities(observations), dtype=float)

    previous_viterbi = np.log(initial_distribution).astype(float)
    bt_pointers = np.zeros((num_observations, num_states), dtype=np.uint32)

    for frame in range(num_observations):
        current_viterbi = np.full(num_states, -np.inf)
        for state in range(num_states):
            density = om_densities[frame, om.pointers[state]]
            for pointer in range(tm.pointers[state], tm.pointers[state + 1]):
                prev_state = tm.states[pointer]
                transition_prob = (previous_viterbi[prev_state] +
                                  tm.log_probabilities[pointer] + density)
                if transition_prob > current_viterbi[state]:
                    current_viterbi[state] = transition_prob
                    bt_pointers[frame, state] = prev_state
        previous_viterbi = current_viterbi

    state = int(np.argmax(previous_viterbi))
    log_probability = float(previous_viterbi[state])
    if np.isinf(log_probability):
        return np.empty(0, dtype=np.uint32), log_probability
    path = np.empty(num_observations, dtype=np.uint32)
    for frame in range(num_observations - 1, -1, -1):
        path[frame] = state
        state = int(bt_pointers[frame, state])
    return path, log_probability


def test_nan_density_matches_reference_not_propagated(toy_hmm_fixture):
    # Regression test for a real fidelity gap found while porting: a NaN
    # transition/density score (e.g. from log() of an out-of-spec activation
    # summing >= 1) must decode to -inf for that state/frame, exactly like
    # pyx's `if transition_prob > current_viterbi[state]:` never lets NaN win
    # a comparison -- naive use of np.maximum (NaN-propagating, as opposed to
    # np.fmax) would instead let NaN leak into and poison the whole decode.
    # Cross-checked against _reference_viterbi (a literal, unvectorized
    # transcription of the pyx algorithm), not just re-derived by hand.
    d = toy_hmm_fixture
    tm = TransitionModel(d["tm_states"], d["tm_pointers"], d["tm_probabilities"])
    om_probs = d["om_probs"].copy()

    class _NanInjectingOM(DiscreteObservationModel):
        def log_densities(self, observations):
            dens = super().log_densities(observations)
            # force a NaN density for state 0 at one frame, and for state 0's
            # *destination* at a later frame too (poisoning a predecessor
            # instead of the state's own density), mimicking log(negative)
            dens[3, 0] = np.nan
            dens[10, 2] = np.nan
            return dens

    om = _NanInjectingOM(om_probs)
    observations = d["observations"]

    ref_path, ref_log_prob = _reference_viterbi(
        tm, om, np.ones(tm.num_states) / tm.num_states, observations)

    hmm = HiddenMarkovModel(tm, om)
    path, log_prob = hmm.viterbi(observations)

    assert not np.isnan(log_prob)
    np.testing.assert_array_equal(path, ref_path)
    if np.isfinite(log_prob) and np.isfinite(ref_log_prob):
        assert log_prob == pytest.approx(ref_log_prob, abs=1e-9)
    else:
        assert np.isinf(log_prob) and np.isinf(ref_log_prob)
