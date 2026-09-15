"""Stationary fixed-label contact-cue diagnostic with synthetic masks."""

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.perception.contact_cues import (
    DangerousPartSegmentation,
    estimate_contact_cues,
)


def main():
    object_mask = np.ones((10, 20), dtype=np.bool_)
    blade_mask = np.zeros_like(object_mask)
    blade_mask[:, :12] = True
    result = estimate_contact_cues(
        DangerousPartSegmentation(
            object_mask,
            blade_mask,
            "knife_blade",
            0.8,
            1,
            "synthetic_fixed_label_fixture",
        ),
        timestamp_seconds=1,
    )
    print("STATIONARY CONTACT DIAGNOSTIC: no camera model or robot motion")
    print(f"part={result.dangerous_part.value}")
    print(f"pixel_exposure={result.dangerous_part_exposure.value:.3f}")
    print("sharpness=unavailable, hardness=unavailable, orientation=unavailable")
    assert not result.sharpness.is_available and not result.hardness.is_available
    print("PASS: segmentation evidence was not promoted to unsupported physics")


if __name__ == "__main__":
    main()
