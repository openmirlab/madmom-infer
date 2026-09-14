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
