"""Phase-1 audio/DSP pipeline: framing, STFT, filterbank, and log-magnitude
spectrogram stages that mirror madmom.audio.{signal,stft,spectrogram,filters}.
All of madmom's original code here is pure numpy/scipy (no Cython), so this
is a near-mechanical port -- the main work is verifying bit-identical output
against madmom via golden fixtures, not redesigning the algorithms.

Reads: madmom_infer/audio/signal.py, stft.py, spectrogram.py, filters.py
"""
