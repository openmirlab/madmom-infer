"""Golden-fixture generator for the ml/hmm.py + features/beats_hmm.py +
features/downbeats.py A/B tests.

Standalone script, run against the *real*, compiled madmom (0.17.dev0) install
in all-in-one-fix/.venv -- must NOT import madmom_infer, only real `madmom`,
numpy and scipy. Dumps every reference input/output pair this workstream's
tests need as one .npz per test target under tests/fixtures/, so the actual
pytest suite (run under whatever numpy/Python the project's own environment
uses) never needs the madmom venv itself, just these recorded arrays.

Run with: /home/worzpro/Desktop/dev/openmirlab/all-in-one-fix/.venv/bin/python
tests/generate_fixtures.py

Reads: real `madmom` (ml.hmm, features.beats_hmm, features.downbeats), numpy,
scipy; writes: tests/fixtures/*.npz
"""

import os

import numpy as np

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def generate_toy_hmm():
    from madmom.ml.hmm import (DiscreteObservationModel, HiddenMarkovModel,
                                TransitionModel)

    rng = np.random.default_rng(42)
    num_states = 10
    num_symbols = 4
    num_frames = 200

    # build a random, row-stochastic (but not fully dense: ~3 destinations per
    # state) dense transition specification, then let TransitionModel.from_dense
    # turn it into the CSR form
    states_list = []
    prev_states_list = []
    prob_list = []
    for origin in range(num_states):
        num_dest = rng.integers(2, 5)
        dests = rng.choice(num_states, size=num_dest, replace=False)
        probs = rng.dirichlet(np.ones(num_dest))
        for dest, prob in zip(dests, probs):
            states_list.append(dest)
            prev_states_list.append(origin)
            prob_list.append(prob)
    states = np.array(states_list, dtype=np.uint32)
    prev_states = np.array(prev_states_list, dtype=np.uint32)
    probabilities = np.array(prob_list, dtype=float)

    tm = TransitionModel.from_dense(states, prev_states, probabilities)

    om_probs = rng.dirichlet(np.ones(num_symbols), size=num_states)
    om = DiscreteObservationModel(om_probs)

    hmm = HiddenMarkovModel(tm, om)

    observations = rng.integers(0, num_symbols, size=num_frames)

    path, log_prob = hmm.viterbi(observations)
    fwd = hmm.forward(observations)

    np.savez(
        os.path.join(FIXTURES_DIR, "toy_hmm.npz"),
        tm_states=tm.states, tm_pointers=tm.pointers,
        tm_probabilities=tm.probabilities,
        om_probs=om_probs,
        observations=observations,
        viterbi_path=path, viterbi_log_prob=log_prob,
        forward=fwd,
    )
    print("wrote toy_hmm.npz")


def generate_bar_state_spaces():
    from madmom.features.beats_hmm import BarStateSpace, BarTransitionModel

    min_bpm, max_bpm, num_tempi, transition_lambda, fps = 55., 215., 60, 100, 100.
    min_interval = 60. * fps / max_bpm
    max_interval = 60. * fps / min_bpm

    out = {}
    for beats in (3, 4):
        st = BarStateSpace(beats, min_interval, max_interval, num_tempi)
        tm = BarTransitionModel(st, transition_lambda)
        prefix = f"bar{beats}_"
        out[prefix + "num_states"] = np.array(st.num_states)
        out[prefix + "states"] = tm.states
        out[prefix + "pointers"] = tm.pointers
        out[prefix + "probabilities"] = tm.probabilities
        out[prefix + "state_positions"] = st.state_positions
        out[prefix + "state_intervals"] = st.state_intervals

    np.savez(os.path.join(FIXTURES_DIR, "bar_state_spaces.npz"), **out)
    print("wrote bar_state_spaces.npz",
         {k: v for k, v in out.items() if k.endswith("num_states")})


def generate_rnn_downbeat_observation_model():
    from madmom.features.beats_hmm import (BarStateSpace,
                                           RNNDownBeatTrackingObservationModel)

    min_bpm, max_bpm, num_tempi, fps = 55., 215., 60, 100.
    min_interval = 60. * fps / max_bpm
    max_interval = 60. * fps / min_bpm
    st = BarStateSpace(4, min_interval, max_interval, num_tempi)
    om = RNNDownBeatTrackingObservationModel(st, observation_lambda=16)

    rng = np.random.default_rng(7)
    num_frames = 500
    activations = rng.uniform(0.0, 0.5, size=(num_frames, 2))
    # make sure beat+downbeat sum stays < 1 (valid probability budget for the
    # "no beat" density) and occasionally spikes to look like real activations
    peak_idx = rng.choice(num_frames, size=num_frames // 20, replace=False)
    activations[peak_idx, 0] = rng.uniform(0.6, 0.95, size=len(peak_idx))
    downbeat_idx = rng.choice(peak_idx, size=len(peak_idx) // 3, replace=False)
    activations[downbeat_idx, 1] = rng.uniform(0.6, 0.9, size=len(downbeat_idx))
    # keep a "no beat" probability budget: rescale rows whose beat+downbeat
    # mass would leave log((1 - sum)/(lambda - 1)) undefined (sum >= 1)
    sums = activations.sum(axis=1)
    over = sums > 0.99
    activations[over] *= (0.99 / sums[over])[:, None]
    activations = np.clip(activations, 1e-6, 1 - 1e-6)

    log_densities = om.log_densities(activations)

    np.savez(
        os.path.join(FIXTURES_DIR, "rnn_downbeat_observation_model.npz"),
        pointers=om.pointers,
        activations=activations,
        log_densities=log_densities,
    )
    print("wrote rnn_downbeat_observation_model.npz")


def generate_dbn_downbeat():
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor

    fps = 100.
    duration_s = 45
    num_frames = int(duration_s * fps)

    rng = np.random.default_rng(123)
    # synthesize a plausible 4/4-ish beat/downbeat activation function: a
    # steady ~120 bpm grid of beat pulses, every 4th one also a downbeat pulse,
    # all corrupted with noise and occasional missed/extra pulses so the DBN
    # decoder has real work (not just reading pulses off a grid)
    bpm = 120.
    beat_period = 60. / bpm * fps
    activations = rng.uniform(0.0, 0.05, size=(num_frames, 2)).astype(float)
    beat_count = 0
    t = beat_period / 2.
    while t < num_frames:
        idx = int(round(t))
        width = 2
        for off in range(-width, width + 1):
            j = idx + off
            if 0 <= j < num_frames:
                bump = max(0.0, 1.0 - abs(off) / (width + 1))
                activations[j, 0] = max(activations[j, 0],
                                        bump * rng.uniform(0.75, 0.98))
                if beat_count % 4 == 0:
                    activations[j, 1] = max(activations[j, 1],
                                            bump * rng.uniform(0.7, 0.95))
        beat_count += 1
        # slight tempo jitter so it's not a perfectly synthetic grid
        t += beat_period * rng.uniform(0.97, 1.03)
    # keep a "no beat" probability budget (see generate_rnn_downbeat_observation_model)
    sums = activations.sum(axis=1)
    over = sums > 0.99
    activations[over] *= (0.99 / sums[over])[:, None]
    activations = np.clip(activations, 1e-6, 1 - 1e-6)

    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=fps)
    result = proc.process(activations)

    np.savez(
        os.path.join(FIXTURES_DIR, "dbn_downbeat.npz"),
        activations=activations,
        result=result,
    )
    print("wrote dbn_downbeat.npz, result shape", result.shape)


def generate_perf_activation():
    # separate, longer (6000-frame) 4/4-only activation array purely for the
    # viterbi wall-clock sanity check (task item: time viterbi on the 4/4
    # model with 6000 frames) -- no madmom reference output needed here,
    # just a representative activation array of the right size.
    fps = 100.
    num_frames = 6000
    rng = np.random.default_rng(999)
    bpm = 100.
    beat_period = 60. / bpm * fps
    activations = rng.uniform(0.0, 0.05, size=(num_frames, 2)).astype(float)
    t = beat_period / 2.
    beat_count = 0
    while t < num_frames:
        idx = int(round(t))
        if 0 <= idx < num_frames:
            activations[idx, 0] = rng.uniform(0.8, 0.98)
            if beat_count % 4 == 0:
                activations[idx, 1] = rng.uniform(0.7, 0.95)
        beat_count += 1
        t += beat_period
    # keep a "no beat" probability budget (see generate_rnn_downbeat_observation_model)
    sums = activations.sum(axis=1)
    over = sums > 0.99
    activations[over] *= (0.99 / sums[over])[:, None]
    activations = np.clip(activations, 1e-6, 1 - 1e-6)
    np.savez(os.path.join(FIXTURES_DIR, "perf_activation.npz"),
            activations=activations)
    print("wrote perf_activation.npz")


if __name__ == "__main__":
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    generate_toy_hmm()
    generate_bar_state_spaces()
    generate_rnn_downbeat_observation_model()
    generate_dbn_downbeat()
    generate_perf_activation()
