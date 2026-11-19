"""Install the v2 restaurant and adult characters from NVIDIA's original assets."""

from contextlib import closing
import hashlib
from http.client import HTTPException
from pathlib import Path, PureWindowsPath
import stat
import tempfile
from urllib.parse import quote
from urllib.request import urlopen
from zipfile import ZipFile
import zlib


PACK_URL = "https://d4i3qtqj3r0z5.cloudfront.net/"
RESTAURANT_ZIP = "AECO_RestaurantDemoPack_NVD@10012.zip"
RESTAURANT_SHA256 = "7b66b0114766c493b2031cd95e0c034b6e795f4bc99253e9968ebab2d65bdad7"
CHARACTERS_ZIP = "Characters_NVD@10012.zip"
CHARACTERS_SHA256 = "09af4376dd42b21a4d18c31b1b639aa84391ed19fb6421a2b5778088ab9e62e7"
CONSTRUCTION_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.0/Isaac/People/Characters/original_male_adult_construction_05/"
)
PACK_PREFIX = "restaurant_live_v3/characters/official-pack/"
NATIVE_PREFIX = "restaurant_live_v3/characters/official-native/construction05/"


def _target(root, name):
    """Keep archive members and cached writes inside their designated directory."""
    parts = name.rstrip("/").split("/")
    if (
        "\\" in name or PureWindowsPath(name).drive
        or any(not part or part.rstrip(" .") != part or ":" in part for part in parts)
    ):
        raise ValueError(f"Invalid asset path: {name}")
    target = root.joinpath(*parts)
    if target.resolve() != target:
        raise ValueError(f"Linked asset path in cache: {target}")
    return target


def _matches(path, expected):
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == expected


def _install_file(stream, target, expected=None, show_progress=False):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(dir=target.parent, delete=False)
    path = Path(temporary.name)
    hasher = hashlib.sha256() if expected else None
    copied = 0
    next_update = 256 * 1024 * 1024
    try:
        with temporary as out:
            while block := stream.read(1024 * 1024):
                out.write(block)
                if hasher:
                    hasher.update(block)
                copied += len(block)
                if show_progress and copied >= next_update:
                    print(f"  {target.name}: {copied // (1024 * 1024)} MiB downloaded", flush=True)
                    next_update += 256 * 1024 * 1024
        if hasher and hasher.hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch for {target.name} after {copied:,} bytes")
        path.replace(target)
    finally:
        path.unlink(missing_ok=True)


def _download(url, target, expected):
    if not _matches(target, expected):
        print(f"Downloading {target.name} from NVIDIA...", flush=True)
        try:
            with closing(urlopen(url, timeout=120)) as response:
                _install_file(response, target, expected, show_progress=True)
        except (OSError, HTTPException, ValueError) as error:
            raise OSError(
                f"Could not install {url}: {error}. Retry setup, or save the original "
                f"file at {target} and run setup again."
            ) from error
    return target


def _archive(cache, name, sha256):
    return _download(PACK_URL + quote(name), _target(cache, name), sha256)


def _zip_member_matches(path, member):
    if not path.is_file() or path.stat().st_size != member.file_size:
        return False
    checksum = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            checksum = zlib.crc32(block, checksum)
    return checksum == member.CRC


def _regular_member(member):
    if stat.S_IFMT(member.external_attr >> 16) not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise ValueError(f"Unexpected special file in NVIDIA archive: {member.filename}")


def _extract_pack(archive, root):
    """Preserve every nested asset, shader and license, repairing interrupted installs."""
    print(f"Preparing {archive.name}...", flush=True)
    with ZipFile(archive) as source:
        for member in source.infolist():
            _regular_member(member)
            target = _target(root, member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not _zip_member_matches(target, member):
                with source.open(member) as stream:
                    _install_file(stream, target)


def restaurant_pack(cache: Path) -> Path:
    """Extract the entire original restaurant pack and return its root."""
    cache = cache.resolve()
    root = _target(cache, "restaurant-10012")
    _extract_pack(_archive(cache, RESTAURANT_ZIP, RESTAURANT_SHA256), root)
    return root


def adult_characters(live_root: Path, expected: dict) -> None:
    """Install original ActorCore dependency paths and the native construction actor."""
    live_root = live_root.resolve()
    archive = _archive(live_root.parent, CHARACTERS_ZIP, CHARACTERS_SHA256)
    _extract_pack(archive, _target(live_root, PACK_PREFIX))
    for name, digest in expected.items():
        if name.startswith(PACK_PREFIX) and not _matches(_target(live_root, name), digest):
            raise ValueError(f"Character pack differs from the recorded v2 asset: {name}")
        elif name.startswith(NATIVE_PREFIX):
            url = CONSTRUCTION_URL + quote(name.removeprefix(NATIVE_PREFIX))
            _download(url, _target(live_root, name), digest)
