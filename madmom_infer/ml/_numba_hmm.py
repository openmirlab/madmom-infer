"""Optional Numba kernel for the exact CPU Viterbi recurrence.

This module is imported lazily by ``ml.hmm`` only when callers request
``fast_viterbi=True``.  It keeps the install-time package pure Python while
moving the frame/state/predecessor scan into native code at first use.  The
kernel deliberately retains the reference decoder's strict ``>`` comparison,
first-predecessor tie break, NaN handling, compact observation lookup, and
adaptive uint16/uint32 backpointer storage.

Reads: numba, numpy; read by: madmom_infer.ml.hmm.
"""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True, fastmath=False)
def _viterbi_core(states, pointers, log_probabilities,
                   observation_pointers, densities, initial_distribution,
                   backpointers):
    """Run the exact sparse Viterbi scan into caller-sized backpointers."""
    num_frames = densities.shape[0]
    num_states = pointers.shape[0] - 1
    previous = np.log(initial_distribution)
    current = np.empty(num_states, dtype=np.float64)

    for frame in range(num_frames):
        for destination in range(num_states):
            best = -np.inf
            source = 0
            for transition in range(pointers[destination],
                                    pointers[destination + 1]):
                candidate = (
                    previous[states[transition]]
                    + log_probabilities[transition]
                )
                # Strict > preserves madmom's first-predecessor tie break and
                # means a NaN candidate never replaces the current maximum.
                if candidate > best:
                    best = candidate
                    source = states[transition]
            value = best + densities[frame, observation_pointers[destination]]
            if np.isnan(value):
                value = -np.inf
            current[destination] = value
            backpointers[frame, destination] = source
        swap = previous
        previous = current
        current = swap

    state = int(np.argmax(previous))
    log_probability = previous[state]
    path = np.empty(num_frames, dtype=np.uint32)
    for frame in range(num_frames - 1, -1, -1):
        path[frame] = state
        state = backpointers[frame, state]
    return path, log_probability


def viterbi(states, pointers, log_probabilities, observation_pointers,
            densities, initial_distribution):
    """Allocate compact backpointers and invoke the compiled recurrence."""
    num_frames = densities.shape[0]
    num_states = pointers.shape[0] - 1
    backpointer_dtype = np.uint16 if num_states <= 65536 else np.uint32
    backpointers = np.zeros(
        (num_frames, num_states), dtype=backpointer_dtype
    )
    return _viterbi_core(
        states,
        pointers,
        log_probabilities,
        observation_pointers,
        densities,
        initial_distribution,
        backpointers,
    )
