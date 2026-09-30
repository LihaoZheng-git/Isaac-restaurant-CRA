"""Contracts for collecting controlled visual-mass data outside CRA runtime.

Ground-truth mass is an oracle training/evaluation label.  This module never
turns simulator truth into runtime evidence and has no Isaac Sim dependency.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Iterable

from robot.perception.physical_attributes import EstimateStatus


VISUAL_MASS_DATASET_SCHEMA_VERSION = 2
SUPPORTED_VISUAL_MASS_DATASET_SCHEMA_VERSIONS = frozenset({1, 2})
ALLOWED_SPLITS = frozenset({"train", "validation", "test"})
ALLOWED_MASS_SOURCES = frozenset(
    {"isaac_authored_mass", "isaac_density_volume_mass", "measured_scale_mass"}
)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value.strip()


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _relative_path(value: object, name: str) -> str:
    text = _text(value, name).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must stay inside the dataset root")
    return str(path)


@dataclass(frozen=True)
class VisualMassSample:
    """One image observation and its controlled mass label."""

    sample_id: str
    object_id: str
    semantic_class: str
    scene_id: str
    split: str
    rgb_path: str
    mass_kg: float
    mass_source: str
    width_m: float
    depth_m: float
    height_m: float
    camera_id: str
    simulation_seed: int
    domain: str = "isaac_sim"
    mask_path: str | None = None
    geometry_status: EstimateStatus = EstimateStatus.UNAVAILABLE
    geometry_source: str = "geometry_provenance_unavailable"

    def __post_init__(self) -> None:
        for name in (
            "sample_id",
            "object_id",
            "semantic_class",
            "scene_id",
            "camera_id",
            "domain",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.split not in ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {sorted(ALLOWED_SPLITS)}")
        if self.mass_source not in ALLOWED_MASS_SOURCES:
            raise ValueError("mass_source must identify auditable ground truth")
        try:
            geometry_status = EstimateStatus(self.geometry_status)
        except (TypeError, ValueError) as error:
            raise ValueError("geometry_status must be a supported EstimateStatus") from error
        geometry_source = _text(self.geometry_source, "geometry_source")
        object.__setattr__(self, "rgb_path", _relative_path(self.rgb_path, "rgb_path"))
        if self.mask_path is not None:
            object.__setattr__(
                self, "mask_path", _relative_path(self.mask_path, "mask_path")
            )
        for name in ("mass_kg", "width_m", "depth_m", "height_m"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if isinstance(self.simulation_seed, bool) or not isinstance(
            self.simulation_seed, int
        ):
            raise TypeError("simulation_seed must be an integer")
        if self.simulation_seed < 0:
            raise ValueError("simulation_seed cannot be negative")
        object.__setattr__(self, "geometry_status", geometry_status)
        object.__setattr__(self, "geometry_source", geometry_source)


@dataclass(frozen=True)
class VisualMassDatasetReport:
    sample_count: int
    object_count: int
    class_count: int
    split_counts: tuple[tuple[str, int], ...]
    missing_files: tuple[str, ...]
    object_split_leaks: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.missing_files and not self.object_split_leaks


def deterministic_object_split(
    object_id: str,
    *,
    salt: str = "cra_visual_mass_v1",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> str:
    """Assign all views of one physical object to the same stable split."""
    key = _text(object_id, "object_id")
    train = float(train_fraction)
    validation = float(validation_fraction)
    if not 0 < train < 1 or not 0 <= validation < 1 or train + validation >= 1:
        raise ValueError("split fractions must leave nonzero train and test portions")
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    if fraction < train:
        return "train"
    if fraction < train + validation:
        return "validation"
    return "test"


def write_visual_mass_manifest(
    samples: Iterable[VisualMassSample], manifest_path: Path
) -> None:
    """Write deterministic JSON Lines metadata; images are stored separately."""
    items = sorted(tuple(samples), key=lambda item: item.sample_id)
    if len({item.sample_id for item in items}) != len(items):
        raise ValueError("sample_id values must be unique")
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            payload = {
                "schema_version": VISUAL_MASS_DATASET_SCHEMA_VERSION,
                **asdict(item),
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_visual_mass_manifest(manifest_path: Path) -> tuple[VisualMassSample, ...]:
    items = []
    with Path(manifest_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            version = payload.pop("schema_version", None)
            if version not in SUPPORTED_VISUAL_MASS_DATASET_SCHEMA_VERSIONS:
                raise ValueError(f"line {line_number}: unsupported schema version")
            items.append(VisualMassSample(**payload))
    if len({item.sample_id for item in items}) != len(items):
        raise ValueError("sample_id values must be unique")
    return tuple(items)


def validate_visual_mass_dataset(
    samples: Iterable[VisualMassSample], dataset_root: Path
) -> VisualMassDatasetReport:
    """Check file presence and prevent view leakage across object splits."""
    items = tuple(samples)
    root = Path(dataset_root).resolve()
    missing = []
    splits_by_object: dict[str, set[str]] = {}
    split_counts = {name: 0 for name in sorted(ALLOWED_SPLITS)}
    for item in items:
        split_counts[item.split] += 1
        splits_by_object.setdefault(item.object_id, set()).add(item.split)
        for relative in (item.rgb_path, item.mask_path):
            if relative is not None and not (root / relative).is_file():
                missing.append(relative)
    leaks = tuple(
        sorted(key for key, value in splits_by_object.items() if len(value) > 1)
    )
    return VisualMassDatasetReport(
        len(items),
        len(splits_by_object),
        len({item.semantic_class for item in items}),
        tuple(split_counts.items()),
        tuple(sorted(set(missing))),
        leaks,
    )


@dataclass(frozen=True)
class VisualMassMetrics:
    count: int
    mean_absolute_error_kg: float
    median_absolute_error_kg: float
    mean_absolute_percentage_error: float


def evaluate_mass_predictions(
    ground_truth_kg: Iterable[float], predictions_kg: Iterable[float]
) -> VisualMassMetrics:
    """Evaluate positive mass predictions without importing an ML framework."""
    truth = tuple(_positive(item, "ground truth") for item in ground_truth_kg)
    predictions = tuple(_positive(item, "prediction") for item in predictions_kg)
    if not truth or len(truth) != len(predictions):
        raise ValueError("truth and predictions must have the same nonzero length")
    raw_errors = tuple(
        abs(actual - predicted) for actual, predicted in zip(truth, predictions)
    )
    errors = sorted(raw_errors)
    count = len(raw_errors)
    median = (
        errors[count // 2]
        if count % 2
        else 0.5 * (errors[count // 2 - 1] + errors[count // 2])
    )
    return VisualMassMetrics(
        count,
        sum(raw_errors) / count,
        median,
        sum(error / actual for error, actual in zip(raw_errors, truth)) / count,
    )
