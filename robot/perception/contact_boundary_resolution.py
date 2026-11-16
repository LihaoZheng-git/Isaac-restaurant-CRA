"""Opt-in one-pixel denoising for a visible contact contour, never occupancy.

The 3x3 cross is the smallest isotropic OpenCV ellipse. This declares a pixel
resolution assumption: one-pixel-wide protrusions are not resolved sharpness
evidence. Real very thin tips can also be removed. A specialist part cue still
enters the existing maximum separately; this is not proof of physical safety.
"""

import cv2
import numpy as np


def resolved_contact_boundary(component, original_contour):
    """Clean a copy only when <=2% of the accepted component is removed.

    Reject the entire correction on excessive removal, splitting, holes, or
    disappearance. Retain the original contour on rejection. The 98% retained
    area uses the existing dominant-component resolution budget, not object
    identity or a desired score. No largest-child selection or hole filling.
    """
    component = np.asarray(component)
    if component.ndim != 2 or not np.isin(component, (0, 1)).all():
        raise ValueError("contact component must be a binary 2D mask")
    original = component.astype(np.uint8, copy=True)
    count = int(original.sum())
    if count == 0:
        raise ValueError("contact component must not be empty")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(original, cv2.MORPH_OPEN, kernel)
    removed = count - int(cleaned.sum())
    evidence = (
        "contact_boundary_resolution=one_pixel_opening; kernel=3x3_cross; maximum_removed_fraction=0.02",
        f"contact_boundary_removed_pixels={removed}; original_component_pixels={count}",
        "contact-only resolution assumption; one-pixel-wide real tips may be unresolved; original mask/ownership/geometry unchanged",
    )
    contours = cv2.findContours(cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    components = cv2.connectedComponents(cleaned, connectivity=8)[0] - 1
    if (np.any(cleaned > original) or removed / count > .02
            or components != 1 or len(contours) != 1 or cv2.contourArea(contours[0]) <= 0):
        return original_contour, (*evidence, "contact_boundary_resolution_rejected; original contour retained")
    return contours[0], (*evidence, "contact_boundary_resolution_accepted")
