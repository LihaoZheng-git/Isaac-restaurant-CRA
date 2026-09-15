"""Read auditable USD mass labels for offline visual-dataset collection only."""

from dataclasses import dataclass
import math

from pxr import Usd, UsdPhysics


@dataclass(frozen=True)
class AuthoredMassGroundTruth:
    prim_path: str
    mass_kg: float
    source: str = "isaac_authored_mass"


def read_authored_mass_ground_truth(
    stage: Usd.Stage, prim_path: str
) -> AuthoredMassGroundTruth:
    """Require an explicit positive `physics:mass`; never guess from class."""
    if not isinstance(stage, Usd.Stage):
        raise TypeError("stage must be a Usd.Stage")
    if not isinstance(prim_path, str) or not prim_path.startswith("/"):
        raise ValueError("prim_path must be absolute")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"mass-label prim not found: {prim_path}")
    attribute = UsdPhysics.MassAPI(prim).GetMassAttr()
    if not attribute.IsValid() or not attribute.HasAuthoredValueOpinion():
        raise RuntimeError(
            f"explicit physics:mass is required for dataset label: {prim_path}"
        )
    mass = float(attribute.Get())
    if not math.isfinite(mass) or mass <= 0:
        raise RuntimeError(f"authored physics:mass must be positive: {prim_path}")
    return AuthoredMassGroundTruth(prim_path, mass)
