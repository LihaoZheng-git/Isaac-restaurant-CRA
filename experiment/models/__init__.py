"""Checkpoint names used by the restaurant experiment."""

import hashlib
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent
MODEL_FILES = {
    "sam": ("SAM2_CHECKPOINT", "sam2.1_t.pt"),
    "mass": ("VISUAL_MASS_CHECKPOINT", "visual-mass.pt"),
    "contact": ("CONTACT_SEGMENTATION_CHECKPOINT", "contact-segmentation.pt"),
    "person": ("PERSON_SEGMENTATION_CHECKPOINT", "yolo11n-seg.pt"),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
