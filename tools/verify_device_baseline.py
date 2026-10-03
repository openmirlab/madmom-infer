"""Replay the original port's CPU device-dispatch baseline without downloads.

Exact floating outputs are guarded by their recording environment. This is
local port regression evidence, not an upstream madmom golden fixture.
Reads: tests/fixtures/device_dispatch; madmom_infer processor entry points.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/device_dispatch"
sys.path.insert(0, str(ROOT))


class BaselineUnavailable(RuntimeError):
    """The exact recording environment or cached weights are unavailable."""


def verify(device="cpu"):
    """Check all outputs exactly, or explain why this environment cannot."""
    try:
        import numpy as np
        import scipy
        import torch
    except ImportError as exc:
        raise BaselineUnavailable(str(exc)) from exc

    metadata = json.loads((FIXTURE / "metadata.json").read_text())
    runtime = metadata["runtime"]
    for name, actual in (("torch", torch.__version__), ("numpy", np.__version__),
                         ("scipy", scipy.__version__)):
        if actual != runtime[name]:
            raise BaselineUnavailable(f"{name} {actual}; fixture requires {runtime[name]}")
    if platform.system() != "Linux" or sys.version.split()[0] != runtime["python"].split()[0]:
        raise BaselineUnavailable("fixture requires its recorded Linux/Python environment")
    try:
        cpu = subprocess.check_output(["lscpu"], text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BaselineUnavailable("cannot identify recording CPU") from exc
    model = next(line for line in runtime["cpu"].splitlines() if "Model name:" in line)
    if model not in cpu or torch.__config__.show() != runtime["torch_config"]:
        raise BaselineUnavailable("fixture CPU or Torch/BLAS build differs")
    numpy_config = io.StringIO()
    with contextlib.redirect_stdout(numpy_config):
        np.show_config()
    if numpy_config.getvalue() != runtime["numpy_config"]:
        raise BaselineUnavailable("fixture NumPy/BLAS build differs")
    if any(os.environ.get(key) != "4" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise BaselineUnavailable("fixture requires OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4")
    if device == "auto" and torch.cuda.is_available():
        raise BaselineUnavailable("CPU auto replay requires CUDA_VISIBLE_DEVICES=''")
    from madmom_infer import models
    from madmom_infer.audio.signal import Signal
    from madmom_infer.features.beats import RNNBeatProcessor
    from madmom_infer.features.downbeats import RNNBarProcessor
    from madmom_infer.torch.features import TorchPipelineProcessor, build_pipeline

    # Check existing cache only. Never acquire missing fixture dependencies.
    for entry in models.validate_checkpoint_config()["models"].values():
        for checkpoint in entry:
            path = models._cache_root() / checkpoint.relpath
            if not path.is_file():
                raise BaselineUnavailable(f"cached checkpoint missing: {path}")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint.sha256
    for filename in ("outputs.npz", "inputs.npz"):
        assert hashlib.sha256((FIXTURE / filename).read_bytes()).hexdigest() == metadata["artifact_sha256"][filename]

    original_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    results = {}
    try:
        with patch("urllib.request.urlopen", side_effect=AssertionError("baseline must not download")), \
                patch("urllib.request.urlretrieve", side_effect=AssertionError("baseline must not download")), \
                np.load(FIXTURE / "inputs.npz") as inputs, \
                np.load(FIXTURE / "outputs.npz") as expected:
            processors = {
                "common_beats": RNNBeatProcessor(backend="torch", device=device),
                "bar": RNNBarProcessor(backend="torch", device=device),
                "direct_key": TorchPipelineProcessor(build_pipeline("key").to("cpu"), device=device),
            }
            for name, processor in processors.items():
                for case in ("music", "silence"):
                    signal = Signal(inputs[case].copy(), sample_rate=44100)
                    actual = np.asarray(processor((signal, inputs["beats"].copy()))
                                        if name == "bar" else processor(signal))
                    key = name + "__" + case
                    reference = expected[key]
                    assert actual.dtype == reference.dtype and actual.shape == reference.shape
                    assert np.array_equal(actual, reference, equal_nan=True), key
                    if name == "bar":
                        assert np.isnan(actual).sum() == 1 and np.isnan(actual[-1, 1])
                        assert np.array_equal(actual[:, 0], inputs["beats"])
                        actual = actual[:-1, 1]
                    assert np.isfinite(actual).all()
                    rms = float(np.sqrt(np.mean(actual.astype(np.float64) ** 2)))
                    results[key] = {"rms": rms, "relative_rms_error": 0.0,
                                    "relative_peak_error": 0.0, "equal_nan": True}
    finally:
        torch.set_num_threads(original_threads)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "auto"), default="cpu")
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.device), indent=2))
    except BaselineUnavailable as exc:
        parser.exit(2, f"Baseline unavailable: {exc}\n")
