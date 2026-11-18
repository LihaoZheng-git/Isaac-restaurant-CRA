"""Restore original seated background performers without adding locomotion."""

import hashlib
import json
import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdSkel

from experiment.scene.resources import DATA, live_assets
from robot.scene.restaurant_human_motion import (
    HUMAN_MOTION_LIBRARY_SHA256,
    HUMAN_MOTION_LIBRARY_URL,
)


_ROOT = "/World/RestaurantLiveV1/SeatedBackground"
_LIBRARY = _ROOT + "/MotionLibrary"
_SOURCE = DATA
_SETUP_HASH = "a07e5c357a729a3ed7cb4eb5edc779263a6743b9af2e5b87846e985ab5bc23be"
_CAST_HASH = "7951ebacbb1e3be034f552aa72479f6c47aa97069bdfe1424412f0598db64b9c"
_HOLD_TOLERANCE_M = .02
_SEAT_TOLERANCE_M = .08
_STABLE_FRAME_COUNT = 6
_STABLE_FRAME_M = .01
_DIAGNOSTIC_JOINTS = {"Hip": "Hips", "Head": "Head", "L_Foot": "LeftFoot",
                     "R_Foot": "RightFoot", "L_Hand": "LeftHand", "R_Hand": "RightHand"}
RECOMMENDED_SEATED_DINERS = ("LeftFarChair", "LeftFarBooth", "LeftOppositeChair",
                            "LeftOppositeBooth", "RightOppositeChair", "RightOppositeBooth",
                            "FamilyMother")
WINDOW_DINER_BACKGROUND = (
    "LeftNearChair", "LeftNearBooth", "LeftFarChair", "LeftFarBooth", "LeftOppositeBooth",
    "FrontChair", "FrontBooth", "RightChair", "RightBooth", "RightOppositeChair",
    "RightOppositeBooth", "FamilyFather", "FamilyMother", "SoloDiner",
)


def author_overview_food(stage):
    """Reuse the overview's original supported place settings in the session layer."""
    from experiment.scene.food import author_family_food

    return author_family_food(stage, asset_root=live_assets() / "restaurant_live_v1/assets")



def _verified_json(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Seated preview receipt changed: {path}")
    return json.loads(data)


def _verified_remote(url, expected):
    import omni.client

    status, _, data = omni.client.read_file(url)
    if status != omni.client.Result.OK or hashlib.sha256(bytes(data)).hexdigest() != expected:
        raise ValueError(f"Seated cast asset changed or unavailable: {url}")


def _archived_diners(names):
    setup = _verified_json(_SOURCE / "native-setup.json", _SETUP_HASH)
    entries = _verified_json(_SOURCE / "dining-cast.json", _CAST_HASH)["actors"]
    adults = {row["name"]: row for row in setup["actors"]
              if row.get("seat_path") and row["name"] != "FamilyChild"}
    names = tuple(names)
    if not names or len(names) != len(set(names)) or not set(names).issubset(adults):
        raise ValueError("Select distinct adult seats from the original overview cast")
    diners = [(adults[name], entries[name]) for name in names]
    for row, entry in diners:
        if "asset_path" in entry:
            base = live_assets()
            row["character"]["source"] = (base / row["character"]["source"]).as_posix()
            entry["asset_path"] = (base / entry["asset_path"]).as_posix()
    return diners


def _seat(stage, name, xyz, facing, behavior_schema):
    path = f"{_ROOT}/Seats/{name}"
    seat = UsdGeom.Xform.Define(stage, path)
    seat.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    basis = Gf.Matrix4d(0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1)
    yaw = math.degrees(math.atan2(facing[1], facing[0]))
    turn = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0., 0., 1.), yaw))
    hips = UsdGeom.Xform.Define(stage, path + "/Hips")
    hips.AddTransformOp().Set(basis * turn)
    effect = stage.DefinePrim(path + "/Sit", "BehaviorTaskEffectors")
    api = behavior_schema.BehaviorTaskEffectorAPI.Apply(effect)
    api.CreateBehaviorTaskAttr("Sit")
    api.CreateBehaviorTaskEffectorHipsRel().SetTargets([hips.GetPath()])
    prop = behavior_schema.BehaviorPropAPI.Apply(seat.GetPrim())
    prop.CreateBehaviorTaskEffectorsRel().SetTargets([effect.GetPath()])
    return path


def _author_diner(stage, row, entry, behavior_schema, nav_schema):
    import omni.kit.commands

    name = row["name"]
    provenance = row["character"]
    source = provenance["source"]
    expected = provenance["sha256"]
    if source != (entry.get("asset_path") or entry["asset_url"]) or expected != entry["sha256"]:
        raise ValueError(f"Overview setup and cast disagree for {name}")
    path = f"{_ROOT}/Actors/{name}"
    actor = UsdGeom.Xform.Define(stage, path)
    actor.AddTranslateOp().Set(Gf.Vec3d(*row["start"]))
    units = UsdGeom.Xform.Define(stage, path + "/Units")
    if provenance["source_up_axis"] == "Y":
        units.AddRotateXOp().Set(90.)
    units.AddScaleOp().Set(Gf.Vec3d(provenance["reference_uniform_scale"]))
    visual = stage.DefinePrim(path + "/Units/Visual")
    visual.GetReferences().AddReference(source)
    rig, = (prim for prim in Usd.PrimRange(visual) if prim.IsA(UsdSkel.Root))
    skeleton, = (prim for prim in Usd.PrimRange(visual) if prim.IsA(UsdSkel.Skeleton))
    UsdSkel.BindingAPI.Apply(skeleton).CreateAnimationSourceRel().SetTargets([])
    rig.ApplyAPI(nav_schema.NavMeshExcludeAPI)
    for hidden in provenance["hidden_costume_meshes"]:
        prim = stage.GetPrimAtPath(hidden.replace(row["root_path"], path, 1))
        if not prim or not prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Original costume mesh is missing for {name}")
        UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    for change in (row["appearance"] or {}).get("changes", ()):
        shader = stage.GetPrimAtPath(change["shader_path"].replace(row["root_path"], path, 1))
        tint = UsdShade.Shader(shader).GetInput("diffuse_tint") if shader else None
        if not tint:
            raise ValueError(f"Original tint is missing for {name}")
        tint.Set(Gf.Vec3f(*change["applied_tint"]))
    for prim in Usd.PrimRange(visual):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
    omni.kit.commands.execute(
        "ApplyBehaviorAgentAPICommand", skelroot_prim_paths=[rig.GetPath()],
        motion_library_prim_path=Sdf.Path(_LIBRARY), motion_library_skeleton_rig="Human",
    )
    if "BehaviorAgentAPI" not in rig.GetAppliedSchemas():
        raise RuntimeError(f"Native seating API is unavailable for {name}")
    tags = list(skeleton.GetAttribute("controlRig:retargetTags").Get() or ())
    joints = {key: row["joints"][key] for key in _DIAGNOSTIC_JOINTS}
    if any(index < 0 or index >= len(tags) or tags[index] != _DIAGNOSTIC_JOINTS[key]
           for key, index in joints.items()):
        raise RuntimeError(f"Native seating joints do not match archived rig for {name}")
    return dict(name=name, source=source, source_sha256=expected, world_seat_xyz=row["seat"],
                facing=row["facing"], rig_path=str(rig.GetPath()), root_path=path,
                seat_path=_seat(stage, name, row["seat"], row["facing"], behavior_schema), joints=joints)


class SeatedRestaurantCast:
    """Snap to each original seat once, then disable native agent updates."""

    def __init__(self, stage, names=RECOMMENDED_SEATED_DINERS):
        from pxr import BehaviorSchema, NavSchema

        if stage.GetPrimAtPath(_ROOT):
            raise ValueError("The seated background cast is already authored")
        diners = _archived_diners(names)
        sources = {row["character"]["source"]: row["character"]["sha256"] for row, _ in diners}
        for source, expected in sources.items():
            if source.startswith("https://"):
                _verified_remote(source, expected)
            elif hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Seated cast asset changed: {source}")
        _verified_remote(HUMAN_MOTION_LIBRARY_URL, HUMAN_MOTION_LIBRARY_SHA256)
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            stage.DefinePrim(_LIBRARY).GetReferences().AddReference(HUMAN_MOTION_LIBRARY_URL)
            self.actors = [_author_diner(stage, row, entry, BehaviorSchema, NavSchema)
                           for row, entry in diners]
        self.metadata = dict(setup_source=str(_SOURCE / "native-setup.json"), setup_sha256=_SETUP_HASH,
                             cast_source=str(_SOURCE / "dining-cast.json"), cast_sha256=_CAST_HASH,
                             actors=self.actors, has_collision=False,
                             scope="Seated appearance only. Visual sensor returns may change.")
        self._native = []
        self._frozen = []
        self._last_positions = []
        self._stable_frames = 0

    def start_seating(self):
        import omni.anim.behavior.core as behavior

        if self._native:
            raise RuntimeError("Seating already started")
        api = behavior.acquire_interface()
        for row in self.actors:
            agent = api.get_agent(row["rig_path"])
            if agent is None:
                raise RuntimeError(f"Native seated agent is unavailable: {row['name']}")
            agent.set_auto_avoidance_enabled(False)
            agent.set_auto_impact_enabled(False)
            task = agent.sit(row["seat_path"], snap_to_seat=True)
            if task == behavior.BEHAVIOR_TASK_ID_INVALID:
                raise RuntimeError(f"Original snap-to-seat rejected for {row['name']}")
            self._native.append((agent, task))

    @staticmethod
    def _positions(agent, row):
        import carb

        values = {}
        for key, index in row["joints"].items():
            pos, rotation = carb.Float3(), carb.Float4()
            if not agent.get_joint_world_transform(index, pos, rotation):
                raise RuntimeError(f"Native seated joint {key} unavailable for {row['name']}")
            xyz = tuple(float(pos[i]) for i in range(3))
            if not all(math.isfinite(value) for value in xyz):
                raise RuntimeError(f"Native seated joint {key} is invalid for {row['name']}")
            values[key] = xyz
        return values

    def freeze_when_seated(self):
        """Sample once per zero-wheel update and freeze only after a settled window."""
        import omni.anim.behavior.core as behavior

        if self._frozen:
            return True
        if len(self._native) != len(self.actors):
            raise RuntimeError("All native seating tasks must be initialized before freezing")
        positions = []
        for row, (agent, task) in zip(self.actors, self._native, strict=True):
            if agent.get_task_status(task) == behavior.BehaviorTaskStatus.FAILED:
                raise RuntimeError(f"Native seating failed for {row['name']}")
            xyz = self._positions(agent, row)
            if math.dist(xyz["Hip"], row["world_seat_xyz"]) > _SEAT_TOLERANCE_M:
                self._last_positions = []
                self._stable_frames = 0
                return False
            positions.append(xyz)
        if self._last_positions and all(
                math.dist(xyz[key], previous[key]) <= _STABLE_FRAME_M
                for xyz, previous in zip(positions, self._last_positions, strict=True) for key in xyz):
            self._stable_frames += 1
        else:
            self._stable_frames = 1
        self._last_positions = positions
        if self._stable_frames < _STABLE_FRAME_COUNT:
            return False
        for agent, _ in self._native:
            agent.set_enabled(False)
            if agent.is_enabled():
                raise RuntimeError("Native seated updates could not be disabled")
        self._frozen = positions
        return True

    def validate_hold(self):
        """Call after a further zero-wheel physics update before using the cast."""
        if not self._frozen:
            raise RuntimeError("The native seated pose was not captured")
        audit = []
        for row, (agent, _), frozen in zip(self.actors, self._native, self._frozen, strict=True):
            observed = self._positions(agent, row)
            drift = max(math.dist(observed[key], frozen[key]) for key in frozen)
            if agent.is_enabled() or drift > _HOLD_TOLERANCE_M:
                raise RuntimeError(f"Native seated pose did not stay fixed for {row['name']}")
            audit.append(dict(name=row["name"], maximum_joint_drift_m=drift, joints_world_xyz=observed))
        return dict(actors=audit, maximum_allowed_joint_drift_m=_HOLD_TOLERANCE_M,
                    capture_method="original native snap-to-seat with agent updates disabled")

    def frozen_seated_agents(self, names):
        """Expose the original seated pose readbacks for optional visual posing."""
        self.validate_hold()
        selected = frozenset(names)
        if not selected or not selected.issubset(row["name"] for row in self.actors):
            raise ValueError("Select gesturing diners from the seated cast")
        return [(row, agent, frozen)
                for row, (agent, _), frozen in zip(self.actors, self._native, self._frozen, strict=True)
                if row["name"] in selected]
