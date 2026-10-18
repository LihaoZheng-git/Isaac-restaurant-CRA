"""Native held-bowl authoring and independent attachment audit, not CRA inputs."""

import hashlib
import math

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from robot.evaluation.restaurant_service_validation import authored_raster_clearance
from robot.safety.safe_sage_configuration_space import RectangularRobotFootprint
from robot.scene.controlled_semantic_obstacle import read_world_bounds
from robot.scene.restaurant_service_build import ASSET_ROOT, audit_source_assets

BOWL_PATH = "/World/CRANativeCarriedBowl"


def author_native_carry_bowl(stage):
    """Reference the audited CC0 mesh; all overrides stay in the session layer."""
    audit_source_assets()
    source = ASSET_ROOT / "usd/bowl-soup.usda"
    if not source.is_file():
        raise FileNotFoundError(source)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        root = UsdGeom.Xform.Define(stage, BOWL_PATH)
        root.GetPrim().GetReferences().AddReference(source.as_posix())
        bounds = read_world_bounds(stage, BOWL_PATH)
        scale = 0.30 / bounds.size[0]
        root.AddTranslateOp().Set(Gf.Vec3d(29.8, 8.4, 1.0))
        root.AddScaleOp().Set(Gf.Vec3d(scale))
        rigid = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        rigid.CreateRigidBodyEnabledAttr(True)
        rigid.CreateKinematicEnabledAttr(True)
        for prim in Usd.PrimRange(root.GetPrim()):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                    "convexHull"
                )
    return {
        "path": BOWL_PATH,
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "license": "CC0-1.0 Kenney Food Kit; audited existing local source",
        "scale": scale,
        "local_center": (np.array(bounds.center) * scale).tolist(),
        "local_half_extents": (np.array(bounds.size) * scale / 2).tolist(),
        "spawn_world_xyz": [29.8, 8.4, 1.0],
        "scope": "native pickup rendering/physics authoring only; convex-hull collision proxy; no measured mass, fall height, heat or CRA scores; not fluid simulation",
    }


def summarize_carry_records(records):
    """A visual/physical attachment gate; intentionally no perception acceptance."""
    active = [r for r in records if r["phase"] != "pickup"]
    idle = [r for r in active if r["phase"] == "idle_before"]
    moving = [r for r in active if r["phase"] == "move"]
    finite = bool(active) and all(
        np.asarray(r["bowl_xyz"]).shape == (3,)
        and np.isfinite(r["bowl_xyz"]).all()
        and all(
            math.isfinite(r[k])
            for k in (
                "hand_distance_m",
                "relative_position_error_m",
                "relative_angle_error_degrees",
                "tilt_degrees",
                "bottom_world_z",
                "layout_clearance_m",
            )
        )
        for r in active
    )
    ordered = (
        len(records) >= 2
        and all(math.isfinite(r["time"]) for r in records)
        and all(b["time"] > a["time"] for a, b in zip(records, records[1:]))
    )
    checks = {
        "distinct_ordered_observations": ordered,
        "stationary_native_attachment": len(idle) >= 30,
        "held_through_walk_and_endpoint_observation": bool(active)
        and all(r["held"] for r in active),
        "rigid_hand_relative_position": finite
        and max(r["relative_position_error_m"] for r in active) <= 0.03,
        "rigid_hand_relative_orientation": finite
        and max(r["relative_angle_error_degrees"] for r in active) <= 10,
        "hand_proximity": finite and max(r["hand_distance_m"] for r in active) <= 0.4,
        "bounded_upright_posture": finite
        and max(r["tilt_degrees"] for r in active) <= 30,
        "not_ground_dragged": finite
        and min(r["bottom_world_z"] for r in active) >= 0.55,
        "independent_layout_clearance": finite
        and min(r["layout_clearance_m"] for r in active) > 0.02,
        "actual_bowl_motion": finite and len(moving) >= 10
        and np.linalg.norm(np.array(moving[-1]["bowl_xyz"]) - moving[0]["bowl_xyz"])
        >= 0.5,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": {k: bool(v) for k, v in checks.items()},
        "maximum_relative_position_error_m": max(
            (r["relative_position_error_m"] for r in active), default=None
        ),
        "maximum_tilt_degrees": max((r["tilt_degrees"] for r in active), default=None),
        "minimum_layout_clearance_m": min(
            (r["layout_clearance_m"] for r in active), default=None
        ),
        "separate_bowl_perception_validated": False,
        "scope": "native held-prop/zero-wheel component only; oracle joint/physics readbacks for evaluation; not grasp planning, mass, fall height, heat, four-factor perception or carrying navigation acceptance",
    }


class NativeBowlCarry:
    """Audit actual native hand state and PhysX prop pose, not stale USD poses."""

    def __init__(self, agent, setup, joint_names):
        from isaacsim.core.experimental.prims import RigidPrim

        self.agent, self.setup = agent, setup
        self.body = RigidPrim(BOWL_PATH)
        if not self.body.is_physics_tensor_entity_valid():
            raise RuntimeError("carried bowl has no physics tensor readback")
        self.joints = {
            side: next(
                i
                for i, n in enumerate(joint_names)
                if n.rsplit("/", 1)[-1] == side + "_Hand"
            )
            for side in ("L", "R")
        }
        self.records = []
        self.baseline = None
        self.consecutive_held = 0
        self.ready = False
        self.task = None

    def start(self):
        import omni.anim.behavior.core as bh

        self.task = self.agent.pickup_object(BOWL_PATH, snap_to_hand=True)
        if self.task == bh.BEHAVIOR_TASK_ID_INVALID:
            raise RuntimeError("native bowl pickup task rejected")

    def tick(self, now, phase, layout):
        import carb
        import omni.anim.behavior.core as bh
        from isaacsim.core.experimental.utils.backend import use_backend

        usage = self.agent.get_hand_usage_objects()
        side = (
            "R"
            if usage.right_hand == BOWL_PATH
            else "L"
            if usage.left_hand == BOWL_PATH
            else None
        )
        if side is None:
            self.consecutive_held = 0
            if (
                self.ready
                or self.agent.get_task_status(self.task) == bh.BehaviorTaskStatus.FAILED
            ):
                raise RuntimeError("native bowl attachment lost/rejected")
            return
        p, q = carb.Float3(), carb.Float4()
        if not self.agent.get_joint_world_transform(self.joints[side], p, q):
            raise RuntimeError("native hand transform unavailable")
        hand_r = Rotation.from_quat([q.x, q.y, q.z, q.w])
        with use_backend("tensor", raise_on_fallback=True):
            pos, rot = self.body.get_world_poses()
            xyz, wxyz = pos.numpy()[0], rot.numpy()[0]
        bowl_r = Rotation.from_quat([*wxyz[1:], wxyz[0]])
        center = xyz + bowl_r.apply(self.setup["local_center"])
        relative_p = hand_r.inv().apply(center - np.array(list(p)))
        relative_r = hand_r.inv() * bowl_r
        self.consecutive_held += 1
        # Native snap/IK may still settle during the initial pickup frames.
        if self.consecutive_held <= 15:
            self.baseline = (relative_p.copy(), relative_r)
        position_error = float(np.linalg.norm(relative_p - self.baseline[0]))
        angle_error = math.degrees((self.baseline[1].inv() * relative_r).magnitude())
        half = np.abs(bowl_r.as_matrix()) @ self.setup["local_half_extents"]
        clearance = authored_raster_clearance(
            layout[0],
            (*center[:2], 0.0),
            RectangularRobotFootprint(
                float(half[0]), float(half[0]), float(half[1]), float(half[1]), 0.0
            ),
            layout[1],
        )
        row = {
            "time": now,
            "phase": phase,
            "held": True,
            "hand": side,
            "bowl_xyz": xyz.tolist(),
            "hand_distance_m": float(np.linalg.norm(center - np.array(list(p)))),
            "relative_position_error_m": position_error,
            "relative_angle_error_degrees": angle_error,
            "tilt_degrees": math.degrees(
                math.acos(float(np.clip(bowl_r.as_matrix()[2, 2], -1, 1)))
            ),
            "bottom_world_z": float(center[2] - half[2]),
            "layout_clearance_m": clearance,
            "native_preferred_speed": self.agent.get_speed(),
            "native_linear_velocity": list(self.agent.get_linear_velocity()),
            "native_target_location": list(self.agent.get_target_location()),
            "native_target_object": self.agent.get_target_object(),
            "native_action_task_id": self.agent.get_action_task_id(),
        }
        self.records.append(row)
        if self.consecutive_held >= 45 and position_error <= 0.03 and angle_error <= 10:
            self.ready = True
        if self.ready and (clearance <= 0.02 or row["bottom_world_z"] < 0.55):
            raise RuntimeError("independent carried-bowl geometry guard failed")
