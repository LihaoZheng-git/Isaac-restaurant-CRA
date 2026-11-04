"""Contact-only boundary refinement of a currently verified object mask.

Use the same audited SAM backend on the original RGB crop. This does not
discover objects, change LiDAR ownership, or alter mass/occupancy geometry.
"""

from dataclasses import dataclass, replace
import hashlib
import math
import time

import cv2
import numpy as np

from robot.perception.semantic_detection import PixelBoundingBox, SemanticDetection


@dataclass(frozen=True)
class ContactMaskRefinementConfig:
    padding_fraction: float = .25
    minimum_quality: float = .85
    minimum_parent_iou: float = .85
    equivalent_candidate_iou: float = .85
    maximum_objects_per_frame: int = 16

    def __post_init__(self):
        for name in ("padding_fraction", "minimum_quality", "minimum_parent_iou", "equivalent_candidate_iou"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if (isinstance(self.maximum_objects_per_frame, bool)
                or not isinstance(self.maximum_objects_per_frame, int)
                or not 1 <= self.maximum_objects_per_frame <= 16):
            raise ValueError("refinement budget must be an integer in [1, 16]")


def refine_contact_mask(rgb, parent, backend, config=None):
    """Return a same-parent contact mask, or retain the original on rejection.

    Selection maximizes agreement with the parent, never a contact score.
    Every candidate keeps the original silhouette pixel/component/hole/border
    requirements; disagreement between plausible candidates rejects refinement.
    Always restore the shared backend's original image, even after an error.
    """
    from robot.perception.visual_shape_contact import SilhouetteContactConfig

    settings = ContactMaskRefinementConfig() if config is None else config
    if not isinstance(settings, ContactMaskRefinementConfig):
        raise TypeError("invalid contact mask refinement configuration")
    if not isinstance(parent, SemanticDetection):
        raise TypeError("parent must be a SemanticDetection")
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (*parent.mask.shape, 3):
        raise ValueError("RGB must align with the parent image")
    yy, xx = np.where(parent.mask)
    gates = SilhouetteContactConfig()
    audit = dict(accepted=False, candidates=[], reason="no acceptable refinement",
        scope="same RGB and audited SAM; contact boundary only, original object ownership and other factors unchanged")
    if len(xx) < gates.minimum_pixels or parent.confidence < gates.minimum_detection_confidence:
        audit["reason"] = "original parent lacks resolved pixels/confidence"
        return parent, audit
    pad = int(settings.padding_fraction * max(np.ptp(xx)+1, np.ptp(yy)+1)) + 1
    h,w = parent.mask.shape
    x0,x1 = max(0,int(xx.min())-pad), min(w,int(xx.max())+pad+1)
    y0,y1 = max(0,int(yy.min())-pad), min(h,int(yy.max())+pad+1)
    old = parent.mask[y0:y1,x0:x1]
    distance = cv2.distanceTransform(old.astype(np.uint8), cv2.DIST_L2, 5)
    py,px = np.unravel_index(distance.argmax(), distance.shape)
    box = (int(xx.min())-x0,int(yy.min())-y0,int(xx.max())+1-x0,int(yy.max())+1-y0)
    started = time.perf_counter()
    try:
        backend.set_image(rgb[y0:y1,x0:x1])
        masks, qualities = backend.predict((int(px),int(py)), context_box_xyxy=box)
    finally:
        backend.set_image(rgb)
    audit.update(crop_xyxy=(x0,y0,x1,y1), inference_and_restore_seconds=time.perf_counter()-started)
    masks, qualities = np.asarray(masks), np.asarray(qualities)
    if masks.ndim != 3 or masks.shape[1:] != old.shape or qualities.shape != (len(masks),):
        raise ValueError("refinement backend returned misaligned masks")
    eligible = []
    for index,(mask,quality) in enumerate(zip(masks.astype(bool), qualities)):
        iou = np.count_nonzero(mask & old) / max(1, np.count_nonzero(mask | old))
        row = dict(index=index, quality=float(quality), parent_iou=iou)
        audit["candidates"].append(row)
        if not np.isfinite(quality) or not settings.minimum_quality <= quality <= 1 or iou < settings.minimum_parent_iou:
            row["rejection"] = "quality or parent agreement"
            continue
        full = np.zeros((h,w),bool)
        full[y0:y1,x0:x1] = mask
        margin = gates.border_margin_pixels
        if (mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()
                or full[:margin].any() or full[-margin:].any() or full[:,:margin].any() or full[:,-margin:].any()):
            row["rejection"] = "crop/image truncation"
            continue
        count = int(mask.sum())
        _,labels,stats,_ = cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        largest = 1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        fraction = stats[largest,cv2.CC_STAT_AREA]/max(count,1)
        contours = cv2.findContours((labels==largest).astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
        if count < gates.minimum_pixels or fraction < gates.minimum_component_fraction or len(contours) != 1:
            row["rejection"] = "original silhouette pixel/component/interior-boundary gate"
            continue
        eligible.append((iou,index,full))
    if not eligible:
        return parent,audit
    chosen_iou, chosen, full = max(eligible,key=lambda item:(item[0],-item[1]))
    if any(np.count_nonzero(full & other)/max(1,np.count_nonzero(full | other)) < settings.equivalent_candidate_iou
           for _,_,other in eligible):
        audit["reason"] = "ambiguous plausible refinements"
        return parent,audit
    y,x = np.where(full)
    refined = replace(parent, mask=full, confidence=min(parent.confidence,float(qualities[chosen])),
        bounding_box=PixelBoundingBox(float(x.min()),float(y.min()),float(x.max()+1),float(y.max()+1)))
    contours = cv2.findContours(full.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    audit.update(accepted=True, reason="same-parent local boundary refinement", chosen_index=chosen,
        selected_parent_iou=chosen_iou,
        mask_sha256=hashlib.sha256(np.packbits(full).tobytes()).hexdigest(),
        mask_pixels=int(full.sum()), contour_xy=max(contours,key=cv2.contourArea).reshape(-1,2).tolist())
    return refined,audit
