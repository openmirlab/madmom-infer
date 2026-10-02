"""Integration checks for fixture integrity and the reference environment gate.

Reads: tests._legacy_fixture_reference and its verified replay sidecar.
"""

import copy
import json

import pytest

from tests import _legacy_fixture_reference as reference


@pytest.fixture
def controlled_reference(tmp_path, monkeypatch):
    record = json.loads(reference.ENVIRONMENT.read_text())
    for filename in record["fixture_sha256"]:
        target = tmp_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((reference.FIXTURES / filename).read_bytes())
    sidecar = tmp_path / "environment.json"
    sidecar.write_text(json.dumps(record))
    actual = copy.deepcopy(record["verified_environment"])
    monkeypatch.setattr(reference, "FIXTURES", tmp_path)
    monkeypatch.setattr(reference, "ENVIRONMENT", sidecar)
    monkeypatch.setattr(reference, "fingerprint", lambda: actual)
    return actual


def test_reference_gate_accepts_matching_environment_and_immutable_inputs(controlled_reference):
    assert reference.reference_mismatch() is None


def test_reference_gate_explains_dispatch_mismatch(controlled_reference):
    controlled_reference["dispatch_environment"]["GLIBC_TUNABLES"] = "glibc.cpu.hwcaps=-FMA"
    assert "dispatch_environment" in reference.reference_mismatch()


@pytest.mark.parametrize("filename", ["gmm_scores.npz", "signal_leftovers.npz", "wavs/mono_44100.wav"])
def test_reference_gate_fails_instead_of_skipping_changed_input(filename, controlled_reference):
    # Integrity is checked even when the environment also differs.
    controlled_reference["dispatch_environment"]["GLIBC_TUNABLES"] = "glibc.cpu.hwcaps=-FMA"
    path = reference.FIXTURES / filename
    path.write_bytes(path.read_bytes() + b"corrupted")
    with pytest.raises(AssertionError, match="immutable fixture changed"):
        reference.reference_mismatch()
