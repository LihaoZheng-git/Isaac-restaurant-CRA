import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from robot.scene.visual_mass_ground_truth import read_authored_mass_ground_truth


def test_reads_explicit_authored_mass():
    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Cube.Define(stage, "/Object").GetPrim()
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(3.5)
    result = read_authored_mass_ground_truth(stage, "/Object")
    assert result.mass_kg == pytest.approx(3.5)
    assert result.source == "isaac_authored_mass"


def test_missing_mass_is_rejected_instead_of_using_class_prior():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage, "/Object")
    with pytest.raises(RuntimeError, match="explicit physics:mass"):
        read_authored_mass_ground_truth(stage, "/Object")


@pytest.mark.parametrize("mass", [0, -1, float("inf")])
def test_invalid_authored_mass_is_rejected(mass):
    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Cube.Define(stage, "/Object").GetPrim()
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    with pytest.raises(RuntimeError, match="must be positive"):
        read_authored_mass_ground_truth(stage, "/Object")
