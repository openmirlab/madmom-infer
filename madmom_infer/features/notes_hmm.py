"""Reimplementation of madmom.features.notes_hmm -- the ADSR (attack, decay,
sustain, release) HMM state-space and transition/observation-model classes
that `ADSRNoteTrackingProcessor` (`madmom_infer/features/notes.py`) decodes
per-pitch note segments from a CNN's [note, onset, offset] activations.
Wave 4e of the complete-port campaign -- see CLAUDE.md's audit table,
`features/notes_hmm.py` row.

Same shape of module as `features/beats_hmm.py`: this file is state-space
*construction* (building the small, per-pitch ADSR transition graph and its
sparse indices/probabilities plus the observation-density pointer table);
the actual Viterbi decoding recursion lives in `ml/hmm.py` and is reused
as-is, unmodified.

Near-mechanical, near-line-for-line port of the pure-Python original (class
names/attributes preserved) -- unlike `beats_hmm.py`'s
`RNNBeatTrackingObservationModel.log_densities`, nothing here hit a
numpy-2.x incompatibility: `ADSRObservationModel.log_densities` is plain
`np.ones`/`np.log` on an already-2D `(N, 3)` observations array (no
`np.array(..., copy=False, ...)`/`ndmin` call at all), confirmed by reading
`notes_hmm.py:152-178` directly.

`ADSRTransitionModel.__init__` builds its transition list `t` as
`(from_state, to_state, prob)` triples, then converts to this project's
`TransitionModel.make_sparse(states, prev_states, probabilities)` signature
exactly like `BeatTransitionModel`/`BarTransitionModel` do -- upstream's own
`self.make_sparse(t[:, 1].astype(int), t[:, 0].astype(int), t[:, 2])` already
passes `(to_state, from_state, prob)` in that argument order, which is
`(states, prev_states, probabilities)`, matching `ml/hmm.py`'s
`TransitionModel.make_sparse` signature with no reordering needed.

Reads: madmom_infer/ml/hmm.py (ObservationModel, TransitionModel);
read by: madmom_infer/features/notes.py (ADSRNoteTrackingProcessor)
"""

import numpy as np

from madmom_infer.ml.hmm import ObservationModel, TransitionModel


class ADSRStateSpace:
    """Map state numbers to actual ADSR states.

    State 0 refers to silence, the ADSR states (attack, decay, sustain,
    release) are numbered from 1 onwards.

    Port of `notes_hmm.ADSRStateSpace` (`madmom-upstream/madmom/features/
    notes_hmm.py:22-54`). The sustain phase has no specific minimum length
    -- self-transitions from this state model the note's (variable) length.

    Parameters
    ----------
    attack_length : int, optional
        Length of the attack phase.
    decay_length : int, optional
        Length of the decay phase.
    release_length : int, optional
        Length of the release phase.
    """

    def __init__(self, attack_length=1, decay_length=1, release_length=1):
        # define note with states which must be transitioned
        self.silence = 0
        self.attack = 1
        self.decay = self.attack + attack_length
        self.sustain = self.decay + decay_length
        self.release = self.sustain + release_length

    @property
    def num_states(self):
        """Number of states."""
        return self.release + 1


class ADSRTransitionModel(TransitionModel):
    """Transition model for note transcription with a HMM.

    Port of `notes_hmm.ADSRTransitionModel` (`madmom-upstream/madmom/
    features/notes_hmm.py:57-121`).

    Parameters
    ----------
    state_space : :class:`ADSRStateSpace` instance
        ADSRStateSpace which maps state numbers to states.
    onset_prob : float, optional
        Probability to enter/stay in the attack and decay phase. When
        entering this phase from a previously sounding note, this
        probability is divided by the sum of `onset_prob`, `note_prob`,
        and `offset_prob`.
    note_prob : float, optional
        Probability to enter the sustain phase. Notes can stay in the
        sustain phase given by this probability divided by the sum of
        `onset_prob`, `note_prob`, and `offset_prob`.
    offset_prob : float, optional
        Probability to enter/stay in the release phase.
    end_prob : float, optional
        Probability to go back from release to silence.
    """

    def __init__(self, state_space, onset_prob=0.8, note_prob=0.8,
                 offset_prob=0.2, end_prob=1.):
        # save attributes
        self.state_space = state_space
        # states
        silence = state_space.silence
        attack = state_space.attack
        decay = state_space.decay
        sustain = state_space.sustain
        release = state_space.release
        # transitions = [(from_state, to_state, prob), ...]
        # onset phase & min_onset_length
        t = [(silence, silence, 1. - onset_prob),
             (silence, attack, onset_prob)]
        for s in range(attack, decay):
            t.append((s, silence, 1. - onset_prob))
            t.append((s, s + 1, onset_prob))
        # transition to note & min_note_duration
        for s in range(decay, sustain):
            t.append((s, silence, 1. - note_prob))
            t.append((s, s + 1, note_prob))
        # 3 possibilities to continue note
        prob_sum = onset_prob + note_prob + offset_prob
        # 1) sustain note (keep sounding)
        t.append((sustain, sustain, note_prob / prob_sum))
        # 2) new note
        t.append((sustain, attack, onset_prob / prob_sum))
        # 3) release note (end note)
        t.append((sustain, sustain + 1, offset_prob / prob_sum))
        # release phase
        for s in range(sustain + 1, release):
            t.append((s, sustain, offset_prob))
            t.append((s, s + 1, 1. - offset_prob))
        # after releasing a note, go back to silence or start new note
        t.append((release, silence, end_prob))
        t.append((release, release, 1. - end_prob))
        t = np.array(t)
        # make the transitions sparse -- (states, prev_states, probabilities)
        # == (to_state, from_state, prob), matching ml/hmm.py's
        # TransitionModel.make_sparse signature directly.
        t = self.make_sparse(t[:, 1].astype(int), t[:, 0].astype(int),
                              t[:, 2])
        # instantiate a TransitionModel
        super().__init__(*t)


class ADSRObservationModel(ObservationModel):
    """Observation model for note transcription tracking with a HMM.

    Port of `notes_hmm.ADSRObservationModel` (`madmom-upstream/madmom/
    features/notes_hmm.py:124-178`). The observed probabilities for note
    onsets, sounding notes, and offsets are mapped to the states defined
    in the state space: 'silence' is `1 - p(onset)`, 'attack' is
    `p(onset)`, 'decay'/'sustain' are `p(note)` (both point at the same
    density column), and 'release' is `p(offset)`.

    Parameters
    ----------
    state_space : :class:`ADSRStateSpace` instance
        ADSRStateSpace instance.
    """

    def __init__(self, state_space):
        # define observation pointers
        pointers = np.zeros(state_space.num_states, dtype=np.uint32)
        # map from densities to states
        pointers[state_space.silence:] = 0
        pointers[state_space.attack:] = 1
        pointers[state_space.decay:] = 2
        # Note: sustain uses the same observations as decay
        pointers[state_space.release:] = 3
        # instantiate a ObservationModel with the pointers
        super().__init__(pointers)

    def log_densities(self, observations):
        """Compute the log densities of the observations.

        Parameters
        ----------
        observations : numpy array, shape (N, 3)
            Observations (i.e. 3D [note, onset, offset] activations of a
            CNN).

        Returns
        -------
        numpy array, shape (N, 4)
            Log densities of the observations -- the columns represent
            silence, attack (onset), decay+sustain (note), and release
            (offset).
        """
        # observations: notes, onsets, offsets
        densities = np.ones((len(observations), 4), dtype=float)
        # silence (not onset)
        densities[:, 0] = 1. - observations[:, 1]
        # attack: onset
        densities[:, 1] = observations[:, 1]
        # decay + sustain: note
        densities[:, 2] = observations[:, 0]
        # release: offset
        densities[:, 3] = observations[:, 2]
        # return the log densities
        return np.log(densities)
