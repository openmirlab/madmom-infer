"""Phase-1 target: reimplementation of madmom.audio.signal -- `Signal` (audio
loading/resampling/normalization wrapper) and `FramedSignalProcessor` (fixed
size/hop frame slicing with configurable origin and padding). Pure numpy in
the original; the main design decision carried over is that framing is a
view/stride trick over the underlying array wherever possible, not a copy,
to keep long-file processing memory-cheap.

Not yet implemented -- this is a Phase-1 stub. See README.md roadmap.

Reads: numpy (planned); read by: madmom_infer/audio/stft.py (planned)
"""

raise NotImplementedError(
    "madmom_infer.audio.signal is a Phase-1 stub: Signal and "
    "FramedSignalProcessor are not yet ported from madmom.audio.signal."
)
