import numpy as np
import pytest

from robot.perception.contact_mask_refinement import ContactMaskRefinementConfig, refine_contact_mask
from robot.perception.semantic_detection import PixelBoundingBox, SemanticDetection


def parent():
    mask = np.zeros((100,120),bool)
    mask[25:65,40:80] = True
    return SemanticDetection(0,"unknown",.75,PixelBoundingBox(40,25,80,65),mask)


class Backend:
    def __init__(self, mode="valid"):
        self.mode = mode
        self.images = []
    def set_image(self, rgb):
        self.images.append(rgb.copy())
    def predict(self, point, *, context_box_xyxy):
        if self.mode == "error":
            raise RuntimeError("controlled inference failure")
        shape = self.images[-1].shape[:2]
        x0,y0,x1,y1 = context_box_xyxy
        mask = np.zeros(shape,bool)
        mask[y0:y1,x0:x1] = True
        assert mask[point[1],point[0]]
        if self.mode == "hole":
            mask[y0+10:y0+15,x0+10:x0+15] = False
        elif self.mode == "part":
            mask[y0:y0+30] = False
        elif self.mode == "border":
            mask[:y0,x0:x0+2] = True
        elif self.mode == "filament":
            # Only eight extra pixels, so IoU still exceeds .99. This connected
            # background line must not create a new high-contact tip.
            mask[y1:y1+8,x0+20] = True
        elif self.mode == "empty":
            return np.empty((0,*shape),bool), np.empty(0)
        elif self.mode == "shape":
            return mask[None,:,:-1],np.array([.95])
        if self.mode == "ambiguous":
            a,b = mask.copy(),mask.copy()
            a[y0:y0+5] = False
            b[y1-5:y1] = False
            return np.array([a,b]),np.array([.95,.95])
        if self.mode == "competing":
            other = mask.copy()
            other[y0:y0+3,x0:x1] = False
            return np.array([other,mask]),np.array([.99,.95])
        return mask[None],np.array([.7 if self.mode == "quality" else .95])


def test_selection_uses_parent_agreement_not_score_or_highest_quality():
    item = parent()
    rgb = np.arange(100*120*3,dtype=np.uint8).reshape(100,120,3)
    backend = Backend("competing")
    refined,audit = refine_contact_mask(rgb,item,backend)
    assert audit["accepted"] and audit["chosen_index"] == 1
    assert refined.confidence == item.confidence
    np.testing.assert_array_equal(refined.mask,item.mask)
    np.testing.assert_array_equal(backend.images[-1],rgb)
    assert len(backend.images) == 2


@pytest.mark.parametrize("mode", ["hole","part","border","empty","quality","ambiguous","filament"])
def test_rejected_refinement_keeps_original_mask_and_restores_image(mode):
    item = parent()
    rgb = np.zeros((100,120,3),np.uint8)
    backend = Backend(mode)
    refined,audit = refine_contact_mask(rgb,item,backend)
    assert not audit["accepted"] and refined is item
    np.testing.assert_array_equal(backend.images[-1],rgb)
    if mode == "filament":
        assert audit["candidates"][0]["parent_iou"] > .99
        assert "boundary allowance" in audit["candidates"][0]["rejection"]


@pytest.mark.parametrize("mode,exception", [("error",RuntimeError),("shape",ValueError)])
def test_backend_failure_restores_original_image(mode,exception):
    rgb = np.zeros((100,120,3),np.uint8)
    backend = Backend(mode)
    with pytest.raises(exception):
        refine_contact_mask(rgb,parent(),backend)
    np.testing.assert_array_equal(backend.images[-1],rgb)


@pytest.mark.parametrize("kwargs", [{"padding_fraction": float('nan')}, {"minimum_parent_iou": 0},
    {"maximum_objects_per_frame": 17}, {"maximum_objects_per_frame": True},
    {"minimum_boundary_allowance_pixels": float('nan')}, {"maximum_boundary_excursion_fraction": 0}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ContactMaskRefinementConfig(**kwargs)
