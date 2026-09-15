"""Persist RGB/mask observations with explicit oracle mass labels."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot.perception.visual_mass_dataset import VisualMassSample


@dataclass(frozen=True)
class VisualMassCapture:
    sample_id: str
    object_id: str
    semantic_class: str
    scene_id: str
    split: str
    mass_kg: float
    mass_source: str
    dimensions_m: tuple[float, float, float]
    camera_id: str
    simulation_seed: int
    rgb: np.ndarray
    object_mask: np.ndarray


def persist_visual_mass_capture(
    capture: VisualMassCapture, dataset_root: Path
) -> VisualMassSample:
    """Save lossless arrays; caller owns Isaac capture and USD mass lookup."""
    if not isinstance(capture, VisualMassCapture):
        raise TypeError("capture must be VisualMassCapture")
    rgb = np.asarray(capture.rgb)
    mask = np.asarray(capture.object_mask)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8 (height, width, 3)")
    if mask.shape != rgb.shape[:2] or mask.dtype != np.bool_:
        raise ValueError("object_mask must be bool and align with rgb")
    if not mask.any():
        raise ValueError("object_mask must contain object pixels")
    root = Path(dataset_root)
    rgb_relative = Path("rgb") / f"{capture.sample_id}.npy"
    mask_relative = Path("masks") / f"{capture.sample_id}.npy"
    paths = ((rgb_relative, rgb), (mask_relative, mask))
    if any((root / relative).exists() for relative, _ in paths):
        raise FileExistsError("refusing to overwrite an existing capture")
    for relative, array in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array, allow_pickle=False)
    width, depth, height = capture.dimensions_m
    return VisualMassSample(
        sample_id=capture.sample_id,
        object_id=capture.object_id,
        semantic_class=capture.semantic_class,
        scene_id=capture.scene_id,
        split=capture.split,
        rgb_path=rgb_relative.as_posix(),
        mask_path=mask_relative.as_posix(),
        mass_kg=capture.mass_kg,
        mass_source=capture.mass_source,
        width_m=width,
        depth_m=depth,
        height_m=height,
        camera_id=capture.camera_id,
        simulation_seed=capture.simulation_seed,
    )
