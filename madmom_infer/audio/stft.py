"""Phase-1 target: reimplementation of madmom.audio.stft --
`ShortTimeFourierTransformProcessor`, applying a window function to each
frame from `FramedSignalProcessor` and taking the (real) FFT. Pure numpy/scipy
in the original. This is also the stage flagged as most likely to benefit
from the optional torch backend, since STFT batches trivially across frames
on a GPU -- unlike the sequential Viterbi decoder in ml/hmm.py.

Not yet implemented -- this is a Phase-1 stub. See README.md roadmap.

Reads: madmom_infer/audio/signal.py (planned, framed input); numpy/scipy (planned)
"""

raise NotImplementedError(
    "madmom_infer.audio.stft is a Phase-1 stub: "
    "ShortTimeFourierTransformProcessor is not yet ported from madmom.audio.stft."
)
