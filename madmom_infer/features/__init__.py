"""Feature-extraction processors, ported incrementally by wave (see
CLAUDE.md's Phase-4 audit table for the ground truth): `beats_hmm.py`
(state spaces feeding `ml/hmm.py`'s Viterbi decoder) and `downbeats.py`
(`RNNDownBeatProcessor`/`DBNDownBeatTrackingProcessor`) are Phase 1/2;
`key.py` (`CNNKeyRecognitionProcessor`) is Wave 4a. The remaining
onset/tempo/chord/note processors are TO-PORT in later 4x waves -- see
CLAUDE.md, not this docstring, for the current status of each.

Reads: madmom_infer/features/beats_hmm.py, downbeats.py, key.py
"""
