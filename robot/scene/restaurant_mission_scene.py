"""Session-layer staging for the continuous restaurant service sequence."""

import hashlib
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from robot.scene.restaurant_newspaper_reader import author_reader_visual
from robot.scene.restaurant_right_crossing import ASSETS, PARKED_PREP_TROLLEY_PATH
from robot.scene.restaurant_server import SERVER_PRIM_PATH


LEFT_DINER_NAVMESH_PATH = "/World/CRAServiceActors/HumanNavMeshVolume"
RIGHT_NEWSPAPER_READER_PATH = "/World/CRAServiceActors/RightNewspaperReader"
_ORIGINAL_SERVER_LAYER = "restaurant_service_isaac_server_v2_server.usda"
_ORIGINAL_SERVER_SHA256 = "f6fad2c0386b6bdf1de781eca32840b1667098e98dab42bf37efe817a01bfcbf"
_STANDING_XYZ = (29.8, 8.9, 0.)
_STANDING_YAW = -17.
_READER_SOURCE = ASSETS / (
    "restaurant_live_v3/characters/official-pack/Assets/Characters/Reallusion/ActorCore/"
    "Party_M_0001/Actor/party-m-0001/party-m-0001.usd"
)
_READER_SHA256 = "bdaa41b5798851d08e6bdf105681968c8fa1e33615cf45f135123ab5cbe9e9d8"


def remove_parked_prep_trolley(stage):
    """Hide the stationary pink prep trolley without affecting the moving cart."""
    prim = stage.GetPrimAtPath(PARKED_PREP_TROLLEY_PATH)
    was_active = bool(prim and prim.IsActive())
    if was_active:
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            if not prim.SetActive(False):
                raise RuntimeError("Could not deactivate the parked restaurant prep trolley")
    return dict(prim_path=PARKED_PREP_TROLLEY_PATH, active_before=was_active, active_after=False)


def verify_left_diner_parked(stage, *, left_diner_phase, prim_path=None):
    """Keep the initial NavMesh volumes registered after the diner sits."""
    if left_diner_phase != "away_endpoint":
        raise RuntimeError("The left diner must finish the native walk before the right phase")
    prim_path = prim_path or LEFT_DINER_NAVMESH_PATH
    volume = stage.GetPrimAtPath(prim_path)
    if not volume or not volume.IsActive():
        raise RuntimeError("The active left diner NavMesh volume is missing")
    return dict(prim_path=prim_path, active=True, staged_before_initial_bake=True)


def stage_right_newspaper_reader(stage):
    """Dress the tested stationary slot with a posed diner and the original collider."""
    source_path = Sdf.Path(SERVER_PRIM_PATH)
    target_path = Sdf.Path(RIGHT_NEWSPAPER_READER_PATH)
    source = stage.GetPrimAtPath(source_path)
    if not source or stage.GetPrimAtPath(target_path):
        raise RuntimeError("The original server or the distinct right standing slot is unavailable")
    layers = [spec.layer for spec in source.GetPrimStack() if spec.layer.realPath
              and Path(spec.layer.realPath).name == _ORIGINAL_SERVER_LAYER]
    if len(layers) != 1:
        raise RuntimeError("The verified original standing woman layer is missing or ambiguous")
    layer, = layers
    source_file = Path(layer.realPath)
    if hashlib.sha256(source_file.read_bytes()).hexdigest() != _ORIGINAL_SERVER_SHA256:
        raise RuntimeError("The verified original standing woman layer changed")
    if hashlib.sha256(_READER_SOURCE.read_bytes()).hexdigest() != _READER_SHA256:
        raise RuntimeError("The right newspaper reader asset changed")
    source_collider_path = source_path.AppendChild("StaticCollision")
    collider_path = target_path.AppendChild(source_collider_path.name)
    session = stage.GetSessionLayer()
    with Usd.EditContext(stage, session):
        standing = UsdGeom.Xform.Define(stage, target_path)
        standing.AddTranslateOp().Set(Gf.Vec3d(*_STANDING_XYZ))
        standing.AddRotateZOp().Set(_STANDING_YAW)
        if not Sdf.CopySpec(layer, source_collider_path, session, collider_path):
            raise RuntimeError("Could not reuse the tested stationary collider")
        collider = stage.GetPrimAtPath(collider_path)
        if not collider.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError("Right standing reader collider is missing")
        pose_path, paper_path = author_reader_visual(stage, target_path, _READER_SOURCE)
        UsdGeom.Imageable(source).MakeInvisible()
        old_collider = stage.GetPrimAtPath(source_collider_path)
        if old_collider and old_collider.IsActive() and not old_collider.SetActive(False):
            raise RuntimeError("Could not deactivate the superseded stationary collider")
    return dict(prim_path=str(target_path), source_prim_path=str(source_path),
                source_layer=str(source_file), source_layer_sha256=_ORIGINAL_SERVER_SHA256,
                world_xyz=_STANDING_XYZ, yaw_degrees=_STANDING_YAW,
                pose_path=str(pose_path), collider_path=str(collider_path), newspaper_path=str(paper_path),
                asset_path=str(_READER_SOURCE), asset_sha256=_READER_SHA256)
