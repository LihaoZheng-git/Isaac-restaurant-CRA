"""Stationary image/video ingestion for the hot-soup dataset.

Collected frames remain pending annotations. Collection alone never turns an
unlabeled frame into a negative example or runtime thermal evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from typing import Iterable

from robot.perception.hot_soup_dataset import (
    ALLOWED_DOMAINS,
    deterministic_capture_group_split,
    sha256_file,
)


HOT_SOUP_COLLECTION_SCHEMA_VERSION = 1
ALLOWED_SOURCE_KINDS = frozenset({"real_image", "real_video_frame", "isaac_render"})


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value.strip()


@dataclass(frozen=True)
class HotSoupCollectedImage:
    collection_id: str
    capture_group_id: str
    suggested_split: str
    image_path: str
    image_sha256: str
    width: int
    height: int
    domain: str
    source_kind: str
    source_reference: str
    usage_rights: str
    annotation_status: str = "pending"
    frame_index: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "collection_id",
            "capture_group_id",
            "image_path",
            "source_reference",
            "usage_rights",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.domain not in ALLOWED_DOMAINS:
            raise ValueError("unsupported collection domain")
        if self.source_kind not in ALLOWED_SOURCE_KINDS:
            raise ValueError("unsupported source_kind")
        if self.annotation_status != "pending":
            raise ValueError("collected images must remain pending annotation")
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.frame_index is not None and (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a nonnegative integer or None")


def _read_image_shape(path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    height, width = image.shape[:2]
    return width, height


def _collected_item(
    path: Path,
    *,
    dataset_root: Path,
    capture_group_id: str,
    domain: str,
    source_kind: str,
    source_reference: str,
    usage_rights: str,
    frame_index: int | None = None,
) -> HotSoupCollectedImage:
    digest = sha256_file(path)
    width, height = _read_image_shape(path)
    relative = path.relative_to(dataset_root).as_posix()
    suffix = f"-f{frame_index:08d}" if frame_index is not None else ""
    return HotSoupCollectedImage(
        collection_id=f"{capture_group_id}{suffix}-{digest[:12]}",
        capture_group_id=capture_group_id,
        suggested_split=deterministic_capture_group_split(capture_group_id),
        image_path=relative,
        image_sha256=digest,
        width=width,
        height=height,
        domain=domain,
        source_kind=source_kind,
        source_reference=source_reference,
        usage_rights=usage_rights,
        frame_index=frame_index,
    )


def collect_hot_soup_images(
    image_paths: Iterable[Path],
    dataset_root: Path,
    *,
    capture_group_id: str,
    domain: str,
    source_kind: str,
    source_reference: str,
    usage_rights: str,
) -> tuple[HotSoupCollectedImage, ...]:
    """Copy owned/authorized images into a content-addressed raw directory."""
    if source_kind == "real_video_frame":
        raise ValueError("use extract_hot_soup_video_frames for video frames")
    root = Path(dataset_root).resolve()
    destination = root / "raw" / capture_group_id
    destination.mkdir(parents=True, exist_ok=True)
    items = []
    seen = set()
    for source in (Path(item).resolve() for item in image_paths):
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = sha256_file(source)
        if digest in seen:
            continue
        seen.add(digest)
        suffix = source.suffix.lower() or ".png"
        target = destination / f"{digest}{suffix}"
        if not target.exists():
            shutil.copy2(source, target)
        elif sha256_file(target) != digest:
            raise FileExistsError(f"content-addressed target is inconsistent: {target}")
        items.append(
            _collected_item(
                target,
                dataset_root=root,
                capture_group_id=capture_group_id,
                domain=domain,
                source_kind=source_kind,
                source_reference=source_reference,
                usage_rights=usage_rights,
            )
        )
    if not items:
        raise ValueError("no unique images were collected")
    return tuple(items)


def extract_hot_soup_video_frames(
    video_path: Path,
    dataset_root: Path,
    *,
    capture_group_id: str,
    source_reference: str,
    usage_rights: str,
    every_n_frames: int = 30,
    max_frames: int | None = None,
) -> tuple[HotSoupCollectedImage, ...]:
    """Extract spaced frames from an owned clip without inferring labels."""
    if isinstance(every_n_frames, bool) or every_n_frames <= 0:
        raise ValueError("every_n_frames must be a positive integer")
    if max_frames is not None and (
        isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0
    ):
        raise ValueError("max_frames must be a positive integer or None")
    import cv2

    source = Path(video_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    root = Path(dataset_root).resolve()
    destination = root / "raw" / capture_group_id
    destination.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {source}")
    items = []
    seen = set()
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % every_n_frames == 0:
                temporary = destination / f"frame-{frame_index:08d}.png"
                if not cv2.imwrite(str(temporary), frame):
                    raise OSError(f"could not write frame: {temporary}")
                digest = sha256_file(temporary)
                target = destination / f"{digest}.png"
                if digest in seen:
                    temporary.unlink()
                else:
                    seen.add(digest)
                    if target.exists():
                        temporary.unlink()
                    else:
                        temporary.replace(target)
                    items.append(
                        _collected_item(
                            target,
                            dataset_root=root,
                            capture_group_id=capture_group_id,
                            domain="real_user_owned",
                            source_kind="real_video_frame",
                            source_reference=source_reference,
                            usage_rights=usage_rights,
                            frame_index=frame_index,
                        )
                    )
                    if max_frames is not None and len(items) >= max_frames:
                        break
            frame_index += 1
    finally:
        capture.release()
    if not items:
        raise ValueError("video produced no decodable sampled frames")
    return tuple(items)


def write_hot_soup_collection_manifest(
    items: Iterable[HotSoupCollectedImage], manifest_path: Path
) -> None:
    values = sorted(tuple(items), key=lambda item: item.collection_id)
    if len({item.collection_id for item in values}) != len(values):
        raise ValueError("collection_id values must be unique")
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in values:
            handle.write(
                json.dumps(
                    {
                        "schema_version": HOT_SOUP_COLLECTION_SCHEMA_VERSION,
                        **asdict(item),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
