"""Phase-1 machine-learning-adjacent decoding primitives: currently just the
numpy Viterbi rewrite in hmm.py. Named `ml` (not e.g. `decode`) to mirror
madmom's own module layout (madmom.ml.hmm), since features/beats_hmm.py and
features/downbeats.py both build their state spaces on top of the
HiddenMarkovModel class defined here.

Reads: madmom_infer/ml/hmm.py
"""
