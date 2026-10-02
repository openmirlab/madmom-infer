"""Optional-runtime-free guard for backend and decoder module imports.

Torch-free guard for `madmom_infer/backends.py` and the NN-backed
processors' `backend="numpy"` (default) path -- runs in a FRESH
interpreter (other test modules may import torch into this process) and
needs neither torch nor network access, so it stays in the default `uv
run pytest` suite (no `pytest.mark.network`, no `importorskip("torch")`),
unlike `tests/test_torch_backend.py`.

Mirrors `tests/test_api.py::test_top_level_import_stays_torch_free`'s
subprocess pattern.

Reads: madmom_infer.features.beats (RNNBeatProcessor, a representative
NN-backed processor -- the same dispatch shape is duplicated across all 10
NN-backed processors, see madmom_infer/backends.py's header for the full
list).
"""

import subprocess
import sys

import pytest


def test_numpy_backend_module_import_stays_optional_runtime_free():
    # Importing the processor module and madmom_infer.backends itself must
    # not import torch -- torch_pipeline_processor only imports
    # madmom_infer.torch lazily, inside its own body. Deliberately does NOT
    # construct RNNBeatProcessor() here: that downloads pretrained weights
    # on first use, which this (non-network-marked) test must not require.
    code = (
        "import sys\n"
        "import madmom_infer.backends\n"
        "import madmom_infer.features.beats\n"
        "import madmom_infer.ml.hmm\n"
        "assert 'torch' not in sys.modules, sorted(sys.modules)\n"
        "assert 'numba' not in sys.modules, sorted(sys.modules)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("entry", ["common", "bar"])
def test_explicit_auto_without_torch_preserves_install_hint(entry):
    code = '''
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("torch unavailable for test")
sys.meta_path.insert(0, NoTorch())
import madmom_infer.models as models
def forbidden(*args, **kwargs):
    raise AssertionError("must fail before checkpoint lookup")
models.downbeats_bgru = forbidden
from madmom_infer.backends import torch_pipeline_processor
from madmom_infer.features.downbeats import RNNBarProcessor
try:
    if ENTRY == "common":
        torch_pipeline_processor("beats", device="auto")
    else:
        RNNBarProcessor(backend="torch", device="auto")
except ImportError as exc:
    assert "madmom-infer[torch]" in str(exc), str(exc)
else:
    raise AssertionError("missing optional dependency did not raise")
assert "torch" not in sys.modules
assert "numba" not in sys.modules
'''
    subprocess.run([sys.executable, "-c", "ENTRY = " + repr(entry) + "\n" + code], check=True)
