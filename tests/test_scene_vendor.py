import hashlib
import io
import stat
from zipfile import ZipFile, ZipInfo

import pytest

from experiment.scene import vendor


def digest(body):
    return hashlib.sha256(body).hexdigest()


def make_zip(entries):
    output = io.BytesIO()
    with ZipFile(output, "w") as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return output.getvalue()


def test_restaurant_installs_nested_dependencies_and_repairs_cached_material(tmp_path, monkeypatch):
    archive = make_zip([("Demos/restaurant.usd", b"scene"), ("Source/Materials/map.png", b"image")])
    monkeypatch.setattr(vendor, "RESTAURANT_SHA256", digest(archive))
    monkeypatch.setattr(vendor, "urlopen", lambda url, timeout: io.BytesIO(archive))
    root = vendor.restaurant_pack(tmp_path)
    material = root / "Source/Materials/map.png"
    assert material.read_bytes() == b"image"
    material.write_bytes(b"wrong")
    monkeypatch.setattr(vendor, "urlopen", lambda *a, **kw: pytest.fail("unexpected download"))
    assert vendor.restaurant_pack(tmp_path) == root
    assert material.read_bytes() == b"image"


def test_adult_cast_installs_shaders_and_notices_and_reuses_verified_cache(tmp_path, monkeypatch):
    live = tmp_path / "live"
    expected = {
        vendor.PACK_PREFIX + "rig.usd": digest(b"rig"),
        vendor.NATIVE_PREFIX + "textures/cloth.png": digest(b"cloth"),
        "restaurant_live_v1/assets/fork.usdc": digest(b"project fork"),
    }
    archive = make_zip([("LICENSE", b"notice"), ("rig.usd", b"rig"),
                        ("Materials/OmniReallusion_PBR.mdl", b"shader")])
    urls = []

    def download(url, timeout):
        urls.append(url)
        if url == vendor.CONSTRUCTION_URL + "textures/cloth.png":
            return io.BytesIO(b"cloth")
        assert url.startswith(vendor.PACK_URL)
        return io.BytesIO(archive)

    monkeypatch.setattr(vendor, "CHARACTERS_SHA256", digest(archive))
    monkeypatch.setattr(vendor, "urlopen", download)
    vendor.adult_characters(live, expected)
    assert (live / vendor.PACK_PREFIX / "LICENSE").read_bytes() == b"notice"
    assert (live / vendor.PACK_PREFIX / "rig.usd").read_bytes() == b"rig"
    assert (live / vendor.NATIVE_PREFIX / "textures/cloth.png").read_bytes() == b"cloth"
    assert (live / vendor.PACK_PREFIX / "Materials/OmniReallusion_PBR.mdl").read_bytes() == b"shader"
    assert not (live / "restaurant_live_v1/assets/fork.usdc").exists()
    vendor.adult_characters(live, expected)
    assert len(urls) == 2


def test_bad_download_preserves_existing_file_and_removes_partial(tmp_path, monkeypatch):
    target = tmp_path / "pack.zip"
    target.write_bytes(b"older")
    monkeypatch.setattr(vendor, "urlopen", lambda url, timeout: io.BytesIO(b"incomplete"))
    with pytest.raises(OSError, match="SHA-256 mismatch"):
        vendor._download("https://example.invalid/pack.zip", target, digest(b"complete"))
    assert target.read_bytes() == b"older"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("name", ["../outside", "C:/escape", "x\\escape"])
def test_rejects_unsafe_zip_member(tmp_path, name):
    with pytest.raises(ValueError, match="Invalid asset path"):
        vendor._target(tmp_path, name)


def test_rejects_zip_symlink(tmp_path, monkeypatch):
    link = ZipInfo("redirect")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = make_zip([(link, b"../elsewhere")])
    (tmp_path / vendor.RESTAURANT_ZIP).write_bytes(archive)
    monkeypatch.setattr(vendor, "RESTAURANT_SHA256", digest(archive))
    with pytest.raises(ValueError, match="special file"):
        vendor.restaurant_pack(tmp_path)
