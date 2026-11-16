"""Calibrated pinhole view for registering the polynomial-fisheye plume cue.

This changes image coordinates only. Mass/contact/fusion retain native sensor
pixels. Source epochs, instance confidence and unknown class stay unchanged.
Unobserved remap pixels are explicitly invalid, never negative plume evidence.
"""

from dataclasses import replace
from functools import lru_cache
import math

import cv2
import numpy as np

from robot.perception.camera_lidar_fusion import _invert_radius_polynomial
from robot.perception.semantic_detection import PixelBoundingBox


@lru_cache(maxsize=4)
def pinhole_remap(calibration, horizontal_fov_degrees=140.0):
    if (
        not math.isfinite(horizontal_fov_degrees)
        or not 30 <= horizontal_fov_degrees < 170
    ):
        raise ValueError("rectification requires a bounded perspective field of view")
    height, width = calibration.output_height, calibration.output_width
    focal = (width / 2) / math.tan(math.radians(horizontal_fov_degrees / 2))
    rows, columns = np.indices((height, width))
    x = (columns - width / 2) / focal
    y = -(rows - height / 2) / focal
    radial = np.hypot(x, y)
    theta = np.arctan(radial)
    radius = _invert_radius_polynomial(theta, calibration.polynomial_coefficients)
    scale = np.divide(radius, radial, out=np.zeros_like(radius), where=radial > 1e-12)
    mx = (calibration.optical_center_x + x * scale) * width / calibration.nominal_width
    my = (
        (calibration.optical_center_y - y * scale) * height / calibration.nominal_height
    )
    valid = (
        (theta <= math.radians(calibration.maximum_fov_degrees / 2))
        & (mx >= 1)
        & (mx < width - 2)
        & (my >= 1)
        & (my < height - 2)
    )
    arrays = (mx.astype(np.float32), my.astype(np.float32), valid)
    for array in arrays:
        array.setflags(write=False)
    return arrays


def rectify_plume_observation(
    observation, calibration, *, horizontal_fov_degrees=140.0
):
    mx, my, valid = pinhole_remap(calibration, horizontal_fov_degrees)
    if observation.rgb.shape[:2] != valid.shape:
        raise ValueError("calibration does not match RGB resolution")
    rgb = cv2.remap(observation.rgb, mx, my, cv2.INTER_LINEAR)
    rgb[~valid] = 0
    detections = []
    for detection in observation.detections.detections:
        mask = (
            cv2.remap(
                detection.mask.astype(np.uint8), mx, my, cv2.INTER_NEAREST
            ).astype(bool)
            & valid
        )
        rows, columns = np.nonzero(mask)
        if not len(rows):
            # Keep the index occupied; an unresolved zero mask is rejected by
            # the plume gates, never attached to a different source.
            box = PixelBoundingBox(0.0, 0.0, 1.0, 1.0)
        else:
            box = PixelBoundingBox(
                float(columns.min()),
                float(rows.min()),
                float(columns.max() + 1),
                float(rows.max() + 1),
            )
        detections.append(replace(detection, mask=mask, bounding_box=box))
    return replace(
        observation,
        rgb=rgb,
        detections=replace(observation.detections, detections=tuple(detections)),
        valid_pixel_mask=valid,
        native_source_mask_pixels=observation.native_source_mask_pixels
        if observation.native_source_mask_pixels is not None
        else int(observation.detection.mask.sum()) if observation.detection is not None else 0,
    )
