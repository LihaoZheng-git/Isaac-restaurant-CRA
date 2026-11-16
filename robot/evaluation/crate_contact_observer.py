"""Passive robot/prop PhysX contact reporting for the Isaac trials."""

from collections import Counter
import math


def _under(path, root):
    return path == root or path.startswith(root + "/")


class CrateContactObserver:
    def __init__(self, robot_root, crate_root, *, target_label="crate"):
        if target_label not in ("crate", "bin"):
            raise ValueError("unknown contact target label")
        self.target_label = target_label
        self.robot_root = str(robot_root).rstrip("/")
        self.crate_root = str(crate_root).rstrip("/")
        self.phase = "parked"
        self.rows = {}
        self.robot_bodies = []
        self.crate_colliders = []
        self.installed = False
        self.error = None
        self._subscription = None
        self.set_phase("parked")

    def set_phase(self, phase):
        if phase not in ("parked", "navigation", "braking"):
            raise ValueError("unknown contact reporting phase")
        self.phase = phase
        self.rows.setdefault(
            phase,
            {
                "robot_environment_headers": Counter(),
                "target_headers": Counter(),
                "target_contact_points": 0,
                "minimum_separation_m": None,
            },
        )

    def _on_contacts(self, headers, data):
        # A diagnostic callback must never interrupt a physics step. A decode
        # failure makes the reporting coverage indeterminate, even if earlier
        # callbacks supplied valid contacts.
        try:
            for header in headers:
                if header.stage_id != self._stage_id:
                    continue
                event = header.type.name
                if event not in ("CONTACT_FOUND", "CONTACT_PERSIST", "CONTACT_LOST"):
                    raise ValueError(f"unrecognized PhysX contact event {event}")
                sides = [
                    tuple(str(self._path(value)) if value else "" for value in pair)
                    for pair in (
                        (header.actor0, header.collider0),
                        (header.actor1, header.collider1),
                    )
                ]
                robots = [
                    any(_under(p, self.robot_root) for p in side) for side in sides
                ]
                if robots.count(True) != 1:
                    continue
                actor, collider = sides[robots.index(False)]
                if not (collider or actor):
                    raise ValueError("unresolved contact counterpart")
                row = self.rows[self.phase]
                if not any(_under(p, self.crate_root) for p in (actor, collider)):
                    if event != "CONTACT_LOST":
                        row["robot_environment_headers"][collider or actor] += 1
                    continue
                row["target_headers"][event] += 1
                if event == "CONTACT_LOST":
                    continue
                begin, count = header.contact_data_offset, header.num_contact_data
                if count < 0 or (count and not 0 <= begin <= len(data) - count):
                    raise ValueError("invalid PhysX contact point span")
                for index in range(begin, begin + count):
                    value = float(data[index].separation) * self._metres_per_unit
                    if not math.isfinite(value):
                        raise ValueError("nonfinite PhysX contact separation")
                    row["target_contact_points"] += 1
                    prior = row["minimum_separation_m"]
                    row["minimum_separation_m"] = (
                        value if prior is None else min(prior, value)
                    )
        except Exception as exc:
            self.error = self.error or f"{type(exc).__name__}: {exc}"

    def report(self):
        phases = {}
        target = self.target_label
        for phase, row in self.rows.items():
            found = (
                row["target_headers"]["CONTACT_FOUND"]
                + row["target_headers"]["CONTACT_PERSIST"]
            )
            healthy = self.installed and self.error is None
            # Wheel/floor or another identifiable external contact in this same
            # phase demonstrates that the callback actually delivered reports.
            observed = (
                found > 0
                if healthy and (found or row["robot_environment_headers"])
                else None
            )
            phases[phase] = {
                "robot_environment_headers": dict(row["robot_environment_headers"]),
                f"{target}_headers": dict(row["target_headers"]),
                f"{target}_contact_points": row["target_contact_points"],
                f"minimum_reported_{target}_separation_m": row["minimum_separation_m"],
                f"{target}_contact_reported": observed,
            }
        return {
            "source": "Isaac PhysX contact report events, session-layer instrumentation",
            "installed": self.installed,
            "error": self.error,
            "robot_reporter_bodies": self.robot_bodies,
            "target_prim_path": self.crate_root,
            f"{target}_colliders": self.crate_colliders,
            "phases": phases,
            "scope": f"Found/persist PhysX headers for this robot and {target}. "
            "Null denotes unmeasured. Contact separation uses the engine contact offset convention.",
        }

    def close(self):
        self._subscription = None


def install(stage, robot_root, crate_root, *, target_label="crate"):
    """Attach zero-force-threshold reporters to robot bodies in the session layer."""
    observer = CrateContactObserver(robot_root, crate_root, target_label=target_label)
    try:
        from omni.physx import get_physx_simulation_interface
        from pxr import (
            PhysxSchema,
            PhysicsSchemaTools,
            Usd,
            UsdGeom,
            UsdPhysics,
            UsdUtils,
        )

        observer._stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        if observer._stage_id < 0:
            raise ValueError("simulation stage has no cache id")
        observer._path = PhysicsSchemaTools.intToSdfPath
        observer._metres_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        robot = stage.GetPrimAtPath(robot_root)
        crate = stage.GetPrimAtPath(crate_root)
        if not robot or not crate:
            raise ValueError("robot or target root is unavailable")
        observer.crate_colliders = [
            str(p.GetPath())
            for p in Usd.PrimRange(crate, Usd.TraverseInstanceProxies())
            if p.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
        ]
        bodies = [
            p
            for p in Usd.PrimRange(robot, Usd.TraverseInstanceProxies())
            if p.HasAPI(UsdPhysics.RigidBodyAPI)
            and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get()
        ]
        if not bodies or not observer.crate_colliders:
            raise ValueError("enabled robot bodies and target colliders required")
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for body in bodies:
                reporter = PhysxSchema.PhysxContactReportAPI.Apply(body)
                if not reporter or not reporter.CreateThresholdAttr().Set(0.0):
                    raise RuntimeError(
                        f"could not enable reporting on {body.GetPath()}"
                    )
                observer.robot_bodies.append(str(body.GetPath()))
        observer._subscription = (
            get_physx_simulation_interface().subscribe_contact_report_events(
                observer._on_contacts
            )
        )
        if observer._subscription is None:
            raise RuntimeError("PhysX returned no contact subscription")
        observer.installed = True
    except Exception as exc:
        observer.error = f"{type(exc).__name__}: {exc}"
    return observer
