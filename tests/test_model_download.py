import hashlib

import pytest

from experiment.models.__main__ import download


def test_verified_download_can_be_reused_offline(tmp_path):
    source = tmp_path / "upstream.pt"
    target = tmp_path / "model.pt"
    source.write_bytes(b"model weights")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    download(source.as_uri(), target, expected)
    source.unlink()
    download(source.as_uri(), target, expected)
    assert target.read_bytes() == b"model weights"


def test_bad_download_leaves_no_checkpoint_or_partial_file(tmp_path):
    source = tmp_path / "upstream.pt"
    source.write_bytes(b"unexpected weights")
    target = tmp_path / "model.pt"
    with pytest.raises(ValueError, match="SHA-256"):
        download(source.as_uri(), target, "0" * 64)
    assert not target.exists()
    assert not list(tmp_path.glob("*.part"))


def test_existing_different_checkpoint_is_preserved(tmp_path):
    target = tmp_path / "model.pt"
    target.write_bytes(b"custom weights")
    with pytest.raises(ValueError, match="Move it aside"):
        download(target.as_uri(), target, "0" * 64)
    assert target.read_bytes() == b"custom weights"
