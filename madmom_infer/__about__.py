"""Single source of truth for the package version, read by hatchling's
`[tool.hatch.version]` at build time and re-exported by `madmom_infer/__init__.py`
at import time. Keeping the version in its own file (rather than in
pyproject.toml or __init__.py directly) means both build and runtime read the
same literal without a packaging tool having to parse Python source.

Reads: (leaf file, no imports); read by: pyproject.toml [tool.hatch.version],
madmom_infer/__init__.py
"""

__version__ = "0.1.1"
