"""Phase-1 (beat/downbeat decoding) and Phase-2 (onset/tempo/chord/key/note
feature extraction, gated) targets. Phase-1 here is limited to the HMM state
spaces and DBN downbeat tracker that consume ml/hmm.py's Viterbi decoder;
see beats_hmm.py and downbeats.py. Everything else in this subpackage
(onset/tempo/chord/key/note extraction, and any neural-net forward pass) is
Phase 2 and gated on the CC BY-NC-SA pretrained-weights question -- see
README.md.

Reads: madmom_infer/features/beats_hmm.py, downbeats.py
"""
