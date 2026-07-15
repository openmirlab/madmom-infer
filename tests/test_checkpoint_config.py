from pathlib import Path

import pytest

from madmom_infer.models import (
    checkpoint_catalog,
    checkpoint_config_path,
    validate_checkpoint_config,
)


def test_package_owned_config_is_installed_and_complete():
    path = checkpoint_config_path()
    assert path.name == "checkpoints.toml"
    config = validate_checkpoint_config()
    assert config["raw"]["package"]["name"] == "madmom-infer"
    assert set(config["models"]) == set(checkpoint_catalog())
    assert all(spec.files for spec in checkpoint_catalog().values())


def test_invalid_checkpoint_metadata_is_rejected(tmp_path):
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(
        '[schema]\nversion = 1\n[defaults]\nbase_url = "http://insecure"\n'
        '[models.bad]\n[[models.bad.files]]\npath = "/absolute.pkl"\n'
        'sha256 = "not-a-digest"\n', encoding="utf-8"
    )
    with pytest.raises(ValueError):
        validate_checkpoint_config(invalid)


def test_download_checkpoint_preserves_pinned_hash_on_url_override(tmp_path, monkeypatch):
    spec = checkpoint_catalog()["key_cnn"]
    target = tmp_path / spec.files[0]
    payload = b"checkpoint-fixture"
    import hashlib
    expected = hashlib.sha256(payload).hexdigest()
    # Use a temporary, valid catalog-style model through the low-level helper
    # contract; the package's pinned catalog remains unchanged.
    from madmom_infer.models import _ModelFile, download
    item = _ModelFile(spec.files[0], expected)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _Response(payload),
    )
    assert download(item, cache_root=tmp_path, url="https://example.test/model") == target
    assert target.read_bytes() == payload


def test_artifact_urls_are_preserved_and_used_by_default(tmp_path, monkeypatch):
    import hashlib
    from madmom_infer.models import _load_checkpoint_config, download

    payloads = {"one": b"first-artifact", "two": b"second-artifact"}
    urls = {"one": "https://mirror.example/one.pkl", "two": "https://cdn.example/two.pkl"}
    config = tmp_path / "checkpoints.toml"
    digest_one = hashlib.sha256(payloads["one"]).hexdigest()
    digest_two = hashlib.sha256(payloads["two"]).hexdigest()
    config.write_text(
        '[schema]\nversion = 1\n[defaults]\nbase_url = "https://legacy.example/models"\n'
        '[models.fixture]\n[[models.fixture.files]]\npath = "one.pkl"\n'
        f'url = "{urls["one"]}"\nsha256 = "{digest_one}"\n'
        '[[models.fixture.files]]\npath = "two.pkl"\n'
        f'url = "{urls["two"]}"\nsha256 = "{digest_two}"\n',
        encoding="utf-8",
    )
    files = _load_checkpoint_config(config)["models"]["fixture"]
    seen = []

    def open_url(url, **kwargs):
        seen.append(url)
        return _Response(payloads[Path(url).stem])

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    for item in files:
        download(item, cache_root=tmp_path / "cache")

    assert seen == [urls["one"], urls["two"]]


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload
