"""Numpy reimplementation of madmom's Cython Viterbi decoder (ml/hmm.pyx).

Phase-1 centerpiece of madmom-infer: replaces the compiled HiddenMarkovModel.viterbi()
with vectorized numpy. Feasible because the beat/downbeat state space is small
(~11k-15k states per bar-length HMM) and transitions are sparse (CSR degree ranges
from 1 for "same tempo" transitions up to a few dozen for beat-boundary tempo-change
transitions) -- confirmed by sizing the original madmom's BarStateSpace/
BarTransitionModel construction.

`viterbi()` keeps the frame loop in Python (a genuine sequential recursion) but
vectorizes the per-frame state/transition double loop with numpy: for each frame,
`candidate = previous_viterbi[tm.states] + tm.log_probabilities` computes every
transition's score at once, then a two-pass `np.fmax.reduceat`/`np.minimum.reduceat`
trick recovers, per destination state (a CSR row = "segment"), both the row's max
score and the *first* transition index attaining it -- replicating madmom's
`if transition_prob > current_viterbi[state]:` (hmm.pyx:552, strict `>`) tie-break,
which keeps the first-encountered predecessor on a tie. `np.argmax`/first-occurrence
semantics line up with this naturally, so no special-casing is needed once the
segment reduction is done correctly. `forward()` mirrors this with `np.add.reduceat`
in the linear (non-log) domain, per-frame renormalized, matching hmm.pyx:591-659.

Both `np.fmax.reduceat`/`np.add.reduceat` have a documented gotcha: consecutive
identical indices (a zero-length CSR segment, i.e. a state with no incoming
transitions) don't reduce to the reduction's identity element -- they silently copy
whatever value sits at that index instead. `_segment_reduce()` below detects and
overrides those positions explicitly (identity: -inf for max, 0.0 for sum) and pads
the input array with a sentinel so a *trailing* zero-length segment (pointer value
equal to the array length) doesn't raise `IndexError` from reduceat. Separately,
`viterbi()` uses `np.fmax` rather than `np.maximum` for the transition-score reduce
specifically because `np.maximum` propagates NaN (poisoning a whole CSR segment if
even one predecessor's score went NaN, e.g. from a malformed/out-of-range
observation), whereas pyx's sequential `>` comparison never lets a NaN win --
`np.fmax` (and an explicit NaN->-inf normalization on the final per-state score,
see the comment in `viterbi()`) reproduces that "NaN never wins" semantics exactly.

Reads: numpy, scipy.sparse (lazy, only inside TransitionModel.make_dense/make_sparse);
read by: madmom_infer/features/beats_hmm.py, madmom_infer/features/downbeats.py
"""

import warnings

import numpy as np


def _segment_reduce(ufunc, values, pointers, identity):
    """
    Segment-reduce `values` over the ragged CSR row ranges defined by `pointers`.

    Parameters
    ----------
    ufunc : numpy ufunc
        Reduction ufunc, e.g. ``np.maximum`` or ``np.add``.
    values : numpy array, shape (pointers[-1],)
        Flat array to reduce, indexed the same way as ``TransitionModel.states``/
        ``probabilities`` (i.e. row ``s``'s entries are ``values[pointers[s]:
        pointers[s + 1]]``).
    pointers : numpy array, shape (num_states + 1,)
        CSR row pointers (see :class:`TransitionModel`).
    identity : float
        Value to use for rows with zero entries (``-inf`` for max, ``0.0`` for sum).

    Returns
    -------
    numpy array, shape (num_states,)
        Per-row reduction, with `identity` substituted for empty rows.

    """
    num_states = len(pointers) - 1
    seg_start = pointers[:-1]
    if len(values) == 0:
        return np.full(num_states, identity, dtype=float)
    # pad with a sentinel so a trailing empty segment (seg_start == len(values))
    # doesn't raise IndexError inside reduceat; the sentinel's own value never
    # survives, since we overwrite every empty-segment result explicitly below
    padded = np.append(values, identity)
    result = ufunc.reduceat(padded, seg_start)
    # reduceat's "copy the element" shortcut for zero-length segments does not
    # give the reduction identity -- overwrite those rows explicitly
    empty = pointers[:-1] == pointers[1:]
    if empty.any():
        result = np.asarray(result, dtype=float).copy()
        result[empty] = identity
    return result


class TransitionModel(object):
    """
    Transition model class for a HMM.

    The transition model is defined similar to a scipy compressed sparse row
    matrix and holds all transition probabilities from one state to another.
    This allows an efficient Viterbi decoding of the HMM.

    Parameters
    ----------
    states : numpy array
        All states transitioning to state s are stored in:
        states[pointers[s]:pointers[s+1]]
    pointers : numpy array
        Pointers for the `states` array for state s.
    probabilities : numpy array
        The corresponding transition are stored in:
        probabilities[pointers[s]:pointers[s+1]].

    Notes
    -----
    This class should be either used for loading saved transition models or
    being sub-classed to define a specific transition model.

    See Also
    --------
    scipy.sparse.csr_matrix

    """

    def __init__(self, states, pointers, probabilities):
        self.states = states
        self.pointers = pointers
        self.probabilities = probabilities

    @property
    def num_states(self):
        """Number of states."""
        return len(self.pointers) - 1

    @property
    def num_transitions(self):
        """Number of transitions."""
        return len(self.probabilities)

    @property
    def log_probabilities(self):
        """Transition log probabilities."""
        return np.log(self.probabilities)

    @staticmethod
    def make_dense(states, pointers, probabilities):
        """
        Return a dense representation of sparse transitions.

        Parameters
        ----------
        states : numpy array
            All states transitioning to state s are returned in:
            states[pointers[s]:pointers[s+1]]
        pointers : numpy array
            Pointers for the `states` array for state s.
        probabilities : numpy array
            The corresponding transition are returned in:
            probabilities[pointers[s]:pointers[s+1]].

        Returns
        -------
        states : numpy array, shape (num_transitions,)
            Array with states (i.e. destination states).
        prev_states : numpy array, shape (num_transitions,)
            Array with previous states (i.e. origination states).
        probabilities : numpy array, shape (num_transitions,)
            Transition probabilities.

        See Also
        --------
        :class:`TransitionModel`

        """
        from scipy.sparse import csr_matrix
        transitions = csr_matrix((np.array(probabilities),
                                  np.array(states), np.array(pointers)))
        states, prev_states = transitions.nonzero()
        return states, prev_states, probabilities

    @staticmethod
    def make_sparse(states, prev_states, probabilities):
        """
        Return a sparse representation of dense transitions.

        This method removes all duplicate states and thus allows an efficient
        Viterbi decoding of the HMM.

        Parameters
        ----------
        states : numpy array, shape (num_transitions,)
            Array with states (i.e. destination states).
        prev_states : numpy array, shape (num_transitions,)
            Array with previous states (i.e. origination states).
        probabilities : numpy array, shape (num_transitions,)
            Transition probabilities.

        Returns
        -------
        states : numpy array
            All states transitioning to state s are returned in:
            states[pointers[s]:pointers[s+1]]
        pointers : numpy array
            Pointers for the `states` array for state s.
        probabilities : numpy array
            The corresponding transition are returned in:
            probabilities[pointers[s]:pointers[s+1]].

        See Also
        --------
        :class:`TransitionModel`

        """
        from scipy.sparse import csr_matrix
        states = np.asarray(states)
        prev_states = np.asarray(prev_states, dtype=int)
        probabilities = np.asarray(probabilities)
        if not np.allclose(np.bincount(prev_states, weights=probabilities), 1):
            raise ValueError('Not a probability distribution.')
        # convert everything into a sparse CSR matrix, make sure it is square.
        # looking through prev_states is enough, because there *must* be a
        # transition *from* every state
        num_states = max(prev_states) + 1
        transitions = csr_matrix((probabilities, (states, prev_states)),
                                 shape=(num_states, num_states))
        states = transitions.indices.astype(np.uint32)
        pointers = transitions.indptr.astype(np.uint32)
        probabilities = transitions.data.astype(dtype=float)
        return states, pointers, probabilities

    @classmethod
    def from_dense(cls, states, prev_states, probabilities):
        """
        Instantiate a TransitionModel from dense transitions.

        Parameters
        ----------
        states : numpy array, shape (num_transitions,)
            Array with states (i.e. destination states).
        prev_states : numpy array, shape (num_transitions,)
            Array with previous states (i.e. origination states).
        probabilities : numpy array, shape (num_transitions,)
            Transition probabilities.

        Returns
        -------
        :class:`TransitionModel` instance
            TransitionModel instance.

        """
        transitions = cls.make_sparse(states, prev_states, probabilities)
        return cls(*transitions)


class ObservationModel(object):
    """
    Observation model class for a HMM.

    The observation model is defined as a plain 1D numpy array `pointers` and
    the methods `log_densities()` and `densities()` which return 2D numpy
    arrays with the (log) densities of the observations.

    Parameters
    ----------
    pointers : numpy array (num_states,)
        Pointers from HMM states to the correct densities. The length of the
        array must be equal to the number of states of the HMM and pointing
        from each state to the corresponding column of the array returned
        by one of the `log_densities()` or `densities()` methods. The
        `pointers` type must be np.uint32.

    """

    def __init__(self, pointers):
        self.pointers = pointers

    def log_densities(self, observations):
        """
        Log densities (or probabilities) of the observations for each state.

        Parameters
        ----------
        observations : numpy array
            Observations.

        Returns
        -------
        numpy array
            Log densities as a 2D numpy array with the number of rows being
            equal to the number of observations and the columns representing
            the different observation log probability densities.

        """
        raise NotImplementedError('must be implemented by subclass')

    def densities(self, observations):
        """
        Densities (or probabilities) of the observations for each state.

        This defaults to computing the exp of the `log_densities`.
        You can provide a special implementation to speed-up everything.

        Parameters
        ----------
        observations : numpy array
            Observations.

        Returns
        -------
        numpy array
            Densities as a 2D numpy array with the number of rows being equal
            to the number of observations and the columns representing the
            different observation log probability densities.

        """
        return np.exp(self.log_densities(observations))


class DiscreteObservationModel(ObservationModel):
    """
    Simple discrete observation model that takes an observation matrix of the
    form (num_states x num_observations) containing P(observation | state).

    Parameters
    ----------
    observation_probabilities : numpy array
        Observation probabilities as a 2D array of shape (num_observations,
        num_states). Has to sum to 1 over the second axis, since it
        represents P(observation | state).

    """

    def __init__(self, observation_probabilities):
        if not np.allclose(observation_probabilities.sum(axis=1), 1):
            raise ValueError('Not a probability distribution.')
        super(DiscreteObservationModel, self).__init__(
            np.arange(observation_probabilities.shape[0], dtype=np.uint32))
        self.observation_probabilities = observation_probabilities

    def densities(self, observations):
        """Densities of the observations."""
        return self.observation_probabilities[:, observations].T

    def log_densities(self, observations):
        """Log densities of the observations."""
        return np.log(self.densities(observations))


class HiddenMarkovModel(object):
    """
    Hidden Markov Model

    To search for the best path through the state space with the Viterbi
    algorithm, the following parameters must be defined.

    Parameters
    ----------
    transition_model : :class:`TransitionModel` instance
        Transition model.
    observation_model : :class:`ObservationModel` instance
        Observation model.
    initial_distribution : numpy array, optional
        Initial state distribution; if 'None' a uniform distribution is
        assumed.

    """

    def __init__(self, transition_model, observation_model,
                 initial_distribution=None):
        self.transition_model = transition_model
        self.observation_model = observation_model
        if initial_distribution is None:
            initial_distribution = np.ones(transition_model.num_states,
                                           dtype=float) / \
                                   transition_model.num_states
        if not np.allclose(initial_distribution.sum(), 1):
            raise ValueError('Initial distribution is not a probability '
                             'distribution.')
        self.initial_distribution = initial_distribution
        # attributes needed for stateful processing (i.e. forward())
        self._prev = self.initial_distribution.copy()

    def reset(self, initial_distribution=None):
        """
        Reset the HMM to its initial state.

        Parameters
        ----------
        initial_distribution : numpy array, optional
            Reset to this initial state distribution.

        """
        self._prev = initial_distribution or self.initial_distribution.copy()

    def viterbi(self, observations):
        """
        Determine the best path with the Viterbi algorithm.

        Parameters
        ----------
        observations : numpy array
            Observations to decode the optimal path for.

        Returns
        -------
        path : numpy array
            Best state-space path sequence.
        log_prob : float
            Corresponding log probability.

        """
        tm = self.transition_model
        tm_states = np.asarray(tm.states, dtype=np.uint32)
        tm_pointers = np.asarray(tm.pointers, dtype=np.uint32)
        tm_log_probabilities = np.asarray(tm.log_probabilities, dtype=float)
        num_states = tm.num_states

        om = self.observation_model
        num_observations = len(observations)
        om_pointers = np.asarray(om.pointers, dtype=np.uint32)
        om_densities = np.asarray(om.log_densities(observations), dtype=float)
        # density_matrix[frame, state] = om_densities[frame, om_pointers[state]]
        density_matrix = om_densities[:, om_pointers]

        # back-tracking pointers, one row per frame
        bt_pointers = np.zeros((num_observations, num_states), dtype=np.uint32)

        # previous viterbi variables, init with the initial state distribution
        previous_viterbi = np.log(self.initial_distribution).astype(float)

        seg_start = tm_pointers[:-1]
        seg_end = tm_pointers[1:]
        empty_segment = seg_start == seg_end
        segment_lengths = seg_end - seg_start
        num_transitions = tm.num_transitions
        idx_full = np.arange(num_transitions)

        for frame in range(num_observations):
            # score of every transition (destination-state density added later,
            # since it is a constant offset per destination state and doesn't
            # change which candidate attains the row max)
            candidate = previous_viterbi[tm_states] + tm_log_probabilities

            # np.fmax (NaN-ignoring), not np.maximum (NaN-propagating): pyx's
            # sequential `if transition_prob > current_viterbi[state]:` scan
            # (hmm.pyx:552) never lets a NaN transition_prob win a comparison,
            # so a single poisoned predecessor (e.g. from a NaN density a few
            # frames back) doesn't stop a state from picking up its true max
            # among the *other*, still-valid predecessors. np.maximum would
            # propagate that NaN across the whole segment instead.
            seg_max = _segment_reduce(np.fmax, candidate, tm_pointers, -np.inf)

            if num_transitions > 0:
                # recover, per row, the index of the *first* transition
                # attaining the row max (madmom's hmm.pyx:552 uses strict `>`,
                # so ties keep the earliest-encountered predecessor -- this
                # matches np.argmax's own first-occurrence tie-break)
                seg_max_per_transition = np.repeat(seg_max, segment_lengths)
                # NaN != NaN, so a NaN candidate (poisoned predecessor) never
                # matches here either -- it can't become the backtrack target
                match = candidate == seg_max_per_transition
                candidate_pos = np.where(match, idx_full, num_transitions)
                seg_argmin = _segment_reduce(np.minimum,
                                             candidate_pos.astype(float),
                                             tm_pointers, float(num_transitions))
                seg_argmin = seg_argmin.astype(np.int64)
                # a segment is only safely indexable if it's non-empty AND a
                # match was actually found (a segment where every candidate is
                # NaN -- e.g. NaN density -- has no match, and seg_argmin
                # stays at the num_transitions sentinel)
                safe = (~empty_segment) & (seg_argmin < num_transitions)
                bt_pointers[frame, safe] = tm_states[seg_argmin[safe]]

            current_viterbi = seg_max + density_matrix[frame]
            # a NaN density (or an all-NaN candidate segment) must decode to
            # -inf, exactly like pyx's reset value survives when nothing ever
            # wins the `>` comparison -- not propagate as NaN
            nan_result = np.isnan(current_viterbi)
            if nan_result.any():
                current_viterbi[nan_result] = -np.inf
            previous_viterbi = current_viterbi

        state = int(np.argmax(previous_viterbi))
        log_probability = float(previous_viterbi[state])

        if np.isinf(log_probability):
            warnings.warn('-inf log probability during Viterbi decoding '
                          'cannot find a valid path', RuntimeWarning)
            return np.empty(0, dtype=np.uint32), log_probability

        path = np.empty(num_observations, dtype=np.uint32)
        for frame in range(num_observations - 1, -1, -1):
            path[frame] = state
            state = int(bt_pointers[frame, state])

        return path, log_probability

    def forward(self, observations, reset=True):
        """
        Compute the forward variables at each time step. Instead of computing
        in the log domain, we normalise at each step, which is faster for the
        forward algorithm.

        Parameters
        ----------
        observations : numpy array, shape (num_frames, num_densities)
            Observations to compute the forward variables for.
        reset : bool, optional
            Reset the HMM to its initial state before computing the forward
            variables.

        Returns
        -------
        numpy array, shape (num_observations, num_states)
            Forward variables.

        """
        tm = self.transition_model
        tm_states = np.asarray(tm.states, dtype=np.uint32)
        tm_pointers = np.asarray(tm.pointers, dtype=np.uint32)
        tm_probabilities = np.asarray(tm.probabilities, dtype=float)
        num_states = tm.num_states

        om = self.observation_model
        om_pointers = np.asarray(om.pointers, dtype=np.uint32)
        om_densities = np.asarray(om.densities(observations), dtype=float)
        num_observations = len(om_densities)
        density_matrix = om_densities[:, om_pointers]

        if reset:
            self.reset()

        fwd_prev = np.asarray(self._prev, dtype=float).copy()
        fwd = np.zeros((num_observations, num_states), dtype=float)

        for frame in range(num_observations):
            contributions = fwd_prev[tm_states] * tm_probabilities
            seg_sum = _segment_reduce(np.add, contributions, tm_pointers, 0.0)
            fwd[frame] = seg_sum * density_matrix[frame]
            prob_sum = fwd[frame].sum()
            norm_factor = 1. / prob_sum
            fwd[frame] *= norm_factor
            fwd_prev = fwd[frame]

        self._prev = fwd[-1].copy() if num_observations else fwd_prev
        return fwd


# alias
HMM = HiddenMarkovModel
