"""Environment-guarded regression of original CPU device entry points.

Reads: tools/verify_device_baseline.py and its committed current-port fixture.
"""

import importlib.util
from pathlib import Path

import pytest


def test_original_cpu_device_dispatch_baseline():
    path = Path(__file__).resolve().parents[1] / "tools/verify_device_baseline.py"
    spec = importlib.util.spec_from_file_location("device_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        results = module.verify("cpu")
    except module.BaselineUnavailable as exc:
        pytest.skip(str(exc))
    assert len(results) == 6
