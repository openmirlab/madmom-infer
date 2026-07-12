"""A/B tests for madmom_infer.features.beats_hmm against real madmom's
pure-Python beats_hmm.py, using the golden fixtures recorded by
tests/generate_fixtures.py (run against the compiled madmom install in
all-in-one-fix/.venv -- see that script's docstring). Uses all-in-one-infer's
exact DBN parameters (beats_per_bar 3 and 4, min_bpm=55, max_bpm=215,
num_tempi=60, transition_lambda=100, observation_lambda=16), per
docs/DESIGN.md C.4 and this workstream's task brief.

Integer/structural arrays (state indices, CSR `states`/`pointers`, interval
counts) are compared with strict `assert_array_equal` -- these come from
`np.arange`/`np.cumsum`/`np.setdiff1d` etc. and are bit-identical on any
IEEE754-conformant platform. `BarTransitionModel.probabilities`, however, is
computed at runtime via `exponential_transition`'s `np.exp()` (see
madmom_infer/features/beats_hmm.py), and libm's `exp` implementation is not
guaranteed bit-identical across CPU/OS/libc builds. Observed on GitHub
Actions' py3.11 runner (run 29173379978): `test_bar_transition_model_csr_exact`
differed from the fixture by exactly 1 float64 ULP (max abs diff 1.11e-16) in
536/21648 elements, while passing bit-exact on the local dev machine -- same
class of env-dependence as the org constitution art.2 clause and test_stft.py/
test_spectrogram.py's ULP-tolerance precedent. That one assertion uses
`np.testing.assert_array_max_ulp(maxulp=4)` (4x the measured 1-ULP worst case,
matching the repo's margin convention); every other assertion in this file
stays exact.

Reads: tests/fixtures/bar_state_spaces.npz,
tests/fixtures/rnn_downbeat_observation_model.npz; madmom_infer/features/beats_hmm.py
"""

import os

import numpy as np
import pytest

from madmom_infer.features.beats_hmm import (
    BarStateSpace, BarTransitionModel, RNNDownBeatTrackingObservationModel,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

MIN_BPM = 55.
MAX_BPM = 215.
NUM_TEMPI = 60
TRANSITION_LAMBDA = 100
FPS = 100.

# measured worst case was exactly 1 ULP (see module header); 4x that, matching
# test_stft.py/test_spectrogram.py's own margin convention.
MAX_ULP = 4


@pytest.fixture(scope="module")
def bar_fixtures():
    return np.load(os.path.join(FIXTURES_DIR, "bar_state_spaces.npz"))


def _build_bar_transition_model(beats_per_bar):
    min_interval = 60. * FPS / MAX_BPM
    max_interval = 60. * FPS / MIN_BPM
    st = BarStateSpace(beats_per_bar, min_interval, max_interval, NUM_TEMPI)
    tm = BarTransitionModel(st, TRANSITION_LAMBDA)
    return st, tm


@pytest.mark.parametrize("beats_per_bar,expected_num_states", [(3, 11157),
                                                                (4, 14876)])
def test_bar_state_space_num_states_exact(beats_per_bar, expected_num_states):
    st, _ = _build_bar_transition_model(beats_per_bar)
    assert st.num_states == expected_num_states


@pytest.mark.parametrize("beats_per_bar", [3, 4])
def test_bar_transition_model_csr_exact(beats_per_bar, bar_fixtures):
    st, tm = _build_bar_transition_model(beats_per_bar)
    prefix = f"bar{beats_per_bar}_"

    assert st.num_states == int(bar_fixtures[prefix + "num_states"])
    np.testing.assert_array_equal(st.state_positions,
                                  bar_fixtures[prefix + "state_positions"])
    np.testing.assert_array_equal(st.state_intervals,
                                  bar_fixtures[prefix + "state_intervals"])

    np.testing.assert_array_equal(tm.states, bar_fixtures[prefix + "states"])
    np.testing.assert_array_equal(tm.pointers,
                                  bar_fixtures[prefix + "pointers"])
    # runtime np.exp() output (exponential_transition): libm's last bit can
    # differ across CPU/libc builds -- CI (py3.11) observed exactly 1 ULP
    # here vs. this file's fixture (see module header); not exact-equality-safe.
    np.testing.assert_array_max_ulp(tm.probabilities,
                                    bar_fixtures[prefix + "probabilities"],
                                    maxulp=MAX_ULP)


def test_rnn_downbeat_observation_model_exact(bar_fixtures=None):
    d = np.load(os.path.join(FIXTURES_DIR,
                             "rnn_downbeat_observation_model.npz"))
    min_interval = 60. * FPS / MAX_BPM
    max_interval = 60. * FPS / MIN_BPM
    st = BarStateSpace(4, min_interval, max_interval, NUM_TEMPI)
    om = RNNDownBeatTrackingObservationModel(st, observation_lambda=16)

    np.testing.assert_array_equal(om.pointers, d["pointers"])

    log_densities = om.log_densities(d["activations"])
    np.testing.assert_allclose(log_densities, d["log_densities"],
                               rtol=0, atol=1e-12)


def test_exponential_transition_diagonal_when_lambda_none():
    from madmom_infer.features.beats_hmm import exponential_transition
    prob = exponential_transition(np.array([4, 5, 6]), np.array([4, 5, 6]),
                                  transition_lambda=None)
    np.testing.assert_array_equal(prob, np.eye(3))
