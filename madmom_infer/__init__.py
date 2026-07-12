"""madmom-infer: a from-scratch, modernized reimplementation of CPJKU/madmom's
inference-relevant algorithms (signal processing, feature extraction, decoding)
for audio and music information retrieval (MIR).

This is an independent reimplementation, not an official fork -- see NOTICE.
madmom's original PyPI release is ~8 years stale and hard to install on
modern Python; this package re-derives the same published algorithms against
current numpy/scipy, verified against madmom's own output via golden
fixtures. The numpy backend is the default, required implementation; an
optional, differentiable torch spectrogram frontend (Phase 3a) is also
available via `import madmom_infer.torch` (requires the `torch` extra) --
this top-level package itself never imports torch. See README.md for the
phased roadmap and the numpy/torch backend split.

Reads: madmom_infer/__about__.py (version string)
"""

from madmom_infer.__about__ import __version__

__all__ = ["__version__"]
