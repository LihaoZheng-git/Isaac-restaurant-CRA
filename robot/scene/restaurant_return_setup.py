"""Session-only scene at the end of the completed left delivery."""

from pxr import Gf, Usd, UsdGeom

from robot.evaluation.restaurant_return_route import DINER_END
from robot.scene.restaurant_live_physical_handoff import LiveCarterPreloadedDelivery
from robot.scene.restaurant_right_crossing import PARKED_PREP_TROLLEY_PATH
from robot.scene.restaurant_server import SERVER_PRIM_PATH


def stage_return_diner(stage):
    """Keep the woman near her recorded departure endpoint for the return."""
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        diner = stage.GetPrimAtPath(SERVER_PRIM_PATH)
        diner.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*DINER_END))
        diner.GetAttribute("xformOp:rotateZ").Set(-90.)
        trolley = stage.GetPrimAtPath(PARKED_PREP_TROLLEY_PATH)
        if trolley and trolley.IsActive():
            trolley.SetActive(False)


def stage_delivered_meal(stage):
    """Reuse the same plate and tabletop as the successful left delivery."""
    handoff = LiveCarterPreloadedDelivery(stage)
    handoff.finish_delivery()
    if any(UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
           for prim in handoff.payload):
        raise RuntimeError("Return must start with the plate off Carter's tray")
    return handoff.events[-1]
