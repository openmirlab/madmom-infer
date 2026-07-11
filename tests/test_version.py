"""Smoke test: madmom_infer must import cleanly and expose a non-empty
version string. This is deliberately the only test in the skeleton commit --
it exists so `pytest tests/` has something to run from commit one, before any
real DSP/decoding code lands.

Reads: madmom_infer/__init__.py (via madmom_infer.__version__)
"""

import madmom_infer


def test_version_is_non_empty_string():
    assert isinstance(madmom_infer.__version__, str)
    assert madmom_infer.__version__ != ""
