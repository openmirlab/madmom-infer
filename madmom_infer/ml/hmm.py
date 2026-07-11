"""Numpy reimplementation of madmom's Cython Viterbi decoder (ml/hmm.pyx).

Phase-1 centerpiece of madmom-infer: replaces the compiled HiddenMarkovModel.viterbi()
with vectorized numpy. Feasible because the beat/downbeat state space is small
(~11k-15k states per bar-length HMM) and transitions are sparse (~1-2 incoming
edges/state) -- confirmed by sizing the original madmom's BarStateSpace/BarTransitionModel
construction. Log-domain recursion; must stay numerically verified against
madmom's own compiled output via golden fixtures before this is trusted.

Note: Viterbi is an inherently sequential (per-frame) recursion, so unlike the
STFT stage, an optional torch backend here is not expected to yield large GPU
speedups -- don't oversell this in downstream docs.

Not yet implemented -- this is a Phase-1 stub. See README.md roadmap.

Reads: numpy (planned); read by: madmom_infer/features/beats_hmm.py (planned),
downbeats.py (planned)
"""

raise NotImplementedError(
    "madmom_infer.ml.hmm is a Phase-1 stub: HiddenMarkovModel.viterbi() is "
    "not yet ported from madmom's compiled ml/hmm.pyx."
)
