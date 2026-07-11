"""Phase-1 target: reimplementation of madmom.audio.spectrogram --
`FilteredSpectrogramProcessor` (applies a filterbank from filters.py to the
STFT magnitude) and `LogarithmicSpectrogramProcessor` (log-compression with
configurable multiplier/add-constant). Together with signal.py and stft.py,
these four stages compose into the standard madmom feature-extraction
pipeline via `SequentialProcessor` (see processors.py).

Not yet implemented -- this is a Phase-1 stub. See README.md roadmap.

Reads: madmom_infer/audio/stft.py (planned), filters.py (planned)
"""

raise NotImplementedError(
    "madmom_infer.audio.spectrogram is a Phase-1 stub: "
    "FilteredSpectrogramProcessor and LogarithmicSpectrogramProcessor are not "
    "yet ported from madmom.audio.spectrogram."
)
