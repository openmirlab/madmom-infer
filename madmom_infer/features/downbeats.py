"""Phase-1 target: reimplementation of madmom.features.downbeats --
`DBNDownBeatTrackingProcessor`, the dynamic Bayesian network downbeat tracker
that wires together beats_hmm.py's bar-length state space with ml/hmm.py's
Viterbi decoder to turn a beat-activation function into downbeat positions.
This is the top-level Phase-1 deliverable that the sibling all-in-one-infer
package needs; signal/stft/spectrogram/filters + beats_hmm + hmm are all
building blocks toward this one entry point.

Not yet implemented -- this is a Phase-1 stub. See README.md roadmap.

Reads: madmom_infer/features/beats_hmm.py (planned), madmom_infer/ml/hmm.py (planned)
"""

raise NotImplementedError(
    "madmom_infer.features.downbeats is a Phase-1 stub: "
    "DBNDownBeatTrackingProcessor is not yet ported from madmom.features.downbeats."
)
