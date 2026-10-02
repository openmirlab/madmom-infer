"""Environment gate and dependency-light replay for two legacy float goldens.

The original generators recorded version claims but no CPU/libm fingerprint.
The separate sidecar records a later verified replay, never invented historical
metadata. This script needs NumPy/SciPy, not pytest or compiled upstream madmom.
Reads: immutable gmm_scores/signal_leftovers fixtures and their replay sidecar.
"""

import argparse
import contextlib
import datetime
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures"
ENVIRONMENT = FIXTURES / "legacy_reference_replay.json"
DISPATCH_ENV = (
    "GLIBC_TUNABLES", "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_HWCAP_MASK",
    "NPY_DISABLE_CPU_FEATURES", "NPY_ENABLE_CPU_FEATURES", "OPENBLAS_CORETYPE",
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "MKL_CBWR",
    "BLIS_NUM_THREADS",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint():
    """Identify numerical builds, libc/libm, CPU features and dispatch knobs."""
    try:
        core = importlib.import_module("numpy._core._multiarray_umath")
    except ImportError:
        core = importlib.import_module("numpy.core._multiarray_umath")
    native = {"numpy_core": _sha(core.__file__)}
    for name in ("scipy.linalg._fblas", "scipy.linalg._flapack"):
        native[name] = _sha(importlib.import_module(name).__file__)
    for package in (np, scipy):
        libs = Path(package.__file__).parent.parent / (package.__name__ + ".libs")
        for path in sorted(libs.glob("*.so*")):
            native[package.__name__ + ".libs/" + path.name] = _sha(path)
    if Path("/proc/self/maps").exists():
        for line in Path("/proc/self/maps").read_text().splitlines():
            path = Path(line.split()[-1])
            if path.name in ("libc.so.6", "libm.so.6"):
                native[path.name] = _sha(path)
    cpu = {}
    if Path("/proc/cpuinfo").exists():
        first = Path("/proc/cpuinfo").read_text().split("\n\n")[0]
        for line in first.splitlines():
            key, _, value = line.partition(":")
            if key.strip() in ("vendor_id", "model name", "cpu family", "model", "stepping", "flags", "microcode"):
                cpu[key.strip()] = value.strip()
    configs = {}
    for name, package in (("numpy", np), ("scipy", scipy)):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            package.show_config()
        configs[name] = output.getvalue()
    return {
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "system": platform.system(), "machine": platform.machine(),
        "libc": list(platform.libc_ver()), "cpu": cpu,
        "numpy_cpu_features": core.__cpu_features__,
        "native_sha256": native, "build_configs": configs,
        "dispatch_environment": {name: os.environ.get(name) for name in DISPATCH_ENV},
    }


def mismatch(actual, expected):
    """Return differing fingerprint fields; never inspect outputs to skip."""
    return [key for key in sorted(set(actual) | set(expected))
            if actual.get(key) != expected.get(key)]


def reference_mismatch():
    record = json.loads(ENVIRONMENT.read_text())
    for filename, digest in record["fixture_sha256"].items():
        assert _sha(FIXTURES / filename) == digest, "immutable fixture changed: " + filename
    differences = mismatch(fingerprint(), record["verified_environment"])
    if differences:
        return "legacy float fixture requires verified reference environment; differs: " + ", ".join(differences)
    return None


def gmm_cases(fixture):
    """Reconstruct every original recorded model, without downloading weights."""
    from madmom_infer.ml.gmm import GMM

    for name in fixture.files:
        if not name.endswith("_means"):
            continue
        key = name[:-len("_means")]
        covariance_type = str(fixture[key.split("_gmm")[0] + "_covariance_type"])
        gmm = GMM(n_components=int(fixture[key + "_n_components"]),
                  covariance_type=covariance_type)
        for field in ("means", "covars", "weights"):
            setattr(gmm, field, fixture[key + "_" + field])
        yield key, gmm


def assert_gmm_exact(fixture):
    """Preserve the original exact assertions for every recorded output."""
    checked = 0
    for key, gmm in gmm_cases(fixture):
        x = fixture[f"{key}_x"]
        log_prob, responsibilities = gmm.score_samples(x)
        np.testing.assert_array_equal(log_prob, fixture[f"{key}_log_prob"])
        np.testing.assert_array_equal(responsibilities, fixture[f"{key}_responsibilities"])
        np.testing.assert_array_equal(gmm.score(x), fixture[f"{key}_log_prob"])
        checked += 1
    assert checked > 0
    return checked


def assert_signal_exact(fixture):
    from madmom_infer.audio.signal import (
        FramedSignalProcessor, Signal, energy, root_mean_square, sound_pressure_level,
    )

    sig = Signal(str(FIXTURES / "wavs/mono_44100.wav"), num_channels=1)
    frames = FramedSignalProcessor(frame_size=2048, fps=100)(sig)
    np.testing.assert_array_equal(energy(frames), fixture["energy_framed"])
    np.testing.assert_array_equal(root_mean_square(frames), fixture["root_mean_square_framed"])
    np.testing.assert_array_equal(sound_pressure_level(frames), fixture["sound_pressure_level_framed"])


def verify_exact():
    with np.load(FIXTURES / "gmm_scores.npz") as fixture:
        count = assert_gmm_exact(fixture)
    with np.load(FIXTURES / "signal_leftovers.npz") as fixture:
        assert_signal_exact(fixture)
    return {"gmm_models_exact": count, "framed_energy_rms_spl_exact": True}


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-environment", action="store_true")
    args = parser.parse_args()
    if args.record_environment:
        assert (platform.python_version(), np.__version__, scipy.__version__) == ("3.10.18", "1.23.5", "1.15.3")
        result = verify_exact()  # Record only after unchanged goldens pass.
        record = {
            "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scope": "Later verified replay environment; original fixture CPU/build metadata was not recorded",
            "original_version_evidence": ["tools/generate_crf_pattern_fixtures.py", "tools/generate_leftovers_fixtures.py"],
            "fixture_sha256": {name: _sha(FIXTURES / name) for name in (
                "gmm_scores.npz", "signal_leftovers.npz", "wavs/mono_44100.wav")},
            "verified_environment": fingerprint(), "verification": result,
        }
        ENVIRONMENT.write_text(json.dumps(record, indent=2) + "\n")
    else:
        reason = reference_mismatch()
        if reason:
            parser.exit(2, reason + "\n")
        result = verify_exact()
    print(json.dumps(result))
