"""Phase-1 target: reimplementation of madmom.features.beats_hmm -- the
bar-length state-space and transition-model classes (e.g. BarStateSpace,
BarTransitionModel) that define the ~11k-15k-state, sparsely-connected HMM
consumed by ml/hmm.py's Viterbi decoder. This is state-space *construction*
(building the transition graph and its sparse indices/probabilities); the
actual decoding recursion lives in ml/hmm.py.

Not yet implemented -- this is a Phase-1 stub. See README.md roadmap.

Reads: madmom_infer/ml/hmm.py (planned, consumes this module's state space);
read by: madmom_infer/features/downbeats.py (planned)
"""

raise NotImplementedError(
    "madmom_infer.features.beats_hmm is a Phase-1 stub: the bar-length state "
    "space/transition model classes are not yet ported from "
    "madmom.features.beats_hmm."
)
