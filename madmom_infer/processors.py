"""Phase-1 target: reimplementation of madmom.processors -- the `Processor`
base class (a callable with a uniform `process(data, **kwargs)` interface)
and `SequentialProcessor` (chains processors so e.g. signal -> framing ->
STFT -> filtering -> log-compression composes into one pipeline object).
This is infrastructure, not DSP math, but every audio/* stub in this package
is designed to slot into a SequentialProcessor the same way madmom's do.

Not yet implemented -- this is a Phase-1 stub. See README.md roadmap.

Reads: (planned) used by madmom_infer/audio/*.py to compose the DSP pipeline
"""

raise NotImplementedError(
    "madmom_infer.processors is a Phase-1 stub: Processor and "
    "SequentialProcessor are not yet ported from madmom.processors."
)
