"""Lifecycle and release-pinned checkpoint metadata contract tests."""

import hashlib

import pytest

import madmom_infer as mm
from madmom_infer.models import cache_info


def test_analyzer_context_manager_loads_and_releases_without_network():
    with mm.MadmomAnalyzer(tasks=("mfcc",)) as analyzer:
        assert analyzer.status == "ready"
        result = analyzer.infer([0.0, 0.0], sample_rate=44100)
        assert "mfcc" in result.values
    assert analyzer.status == "released"
    with pytest.raises(RuntimeError):
        analyzer.infer([0.0], sample_rate=44100)


def test_checkpoint_catalog_is_pinned_and_cache_info_verifies(tmp_path):
    catalog = mm.checkpoint_catalog()
    assert "beats_blstm" in catalog
    spec = catalog["beats_blstm"]
    assert spec.url.startswith("https://")
    assert len(spec.files) == len(spec.sha256) == 8

    path = tmp_path / spec.files[0]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not-a-model")
    info = cache_info("beats_blstm", cache_root=tmp_path)
    assert info["beats_blstm"]["complete"] is False
    assert info["beats_blstm"]["files"][0]["verified"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() != spec.sha256[0]
