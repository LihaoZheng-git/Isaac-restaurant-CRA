"""Advisory travel preference from current observed approaching human motion.

Predicted centers and radius growth are hypotheses, never replacement geometry,
physical factor evidence, free-space evidence or execution permission.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from robot.perception.human_body_envelope import HUMAN_ENVELOPE_SOURCE


@dataclass(frozen=True)
class ObservedHumanMotionReference:
    track_id: int
    sensor_frame_id: int
    sensor_timestamp_seconds: float
    center_world_xy: tuple[float, float]
    velocity_world_xy: tuple[float, float]
    radius_prior_m: float
    centroid_uncertainty_prior_m: float
    current_support_count: int
    original_person_camera_timestamp: float
    original_person_camera_frame_id: int

    def __post_init__(self):
        for name in (
            "track_id",
            "sensor_frame_id",
            "current_support_count",
            "original_person_camera_frame_id",
        ):
            v = getattr(self, name)
            if (
                isinstance(v, (bool, np.bool_))
                or not isinstance(v, (int, np.integer))
                or v < 0
            ):
                raise ValueError("nonnegative integer identity/source/support required")
            object.__setattr__(self, name, int(v))
        for name in ("center_world_xy", "velocity_world_xy"):
            a = np.asarray(getattr(self, name), float)
            if a.shape != (2,) or not np.isfinite(a).all():
                raise ValueError("finite current observed vectors required")
            object.__setattr__(self, name, tuple(map(float, a)))
        for name in (
            "sensor_timestamp_seconds",
            "radius_prior_m",
            "centroid_uncertainty_prior_m",
            "original_person_camera_timestamp",
        ):
            v = getattr(self, name)
            if (
                isinstance(v, (bool, np.bool_))
                or not isinstance(v, (int, float, np.number))
                or not math.isfinite(v)
                or v < 0
            ):
                raise ValueError("finite nonnegative source/prior required")
            object.__setattr__(self, name, float(v))
        if (
            not 0 < self.centroid_uncertainty_prior_m < self.radius_prior_m
            or self.current_support_count < 20
            or not 0.2 <= np.linalg.norm(self.velocity_world_xy) <= 2.0
            or not -0.25
            <= self.sensor_timestamp_seconds - self.original_person_camera_timestamp
            <= 30.0
        ):
            raise ValueError(
                "bounded current moving human/body/identity evidence required"
            )

    @classmethod
    def from_current(cls, tracking, envelopes, selected_ids, *, frame):
        """No identity borrowing or zero-velocity substitution for missing input."""
        if len(selected_ids) != 1:
            return None
        key = selected_ids[0]
        track = next((t for t in tracking.tracked_obstacles if t.track_id == key), None)
        bodies = [e for e in envelopes if e["track_id"] == key]
        if (
            track is None
            or len(bodies) != 1
            or not track.matched_previous_track
            or track.observation_count < 3
            or track.point_count < 20
        ):
            return None
        e = bodies[0]
        if (
            e.get("source") != HUMAN_ENVELOPE_SOURCE
            or e.get("status") != "prior"
            or e.get("applied_to_occupancy") is not True
            or e["lidar_timestamp_seconds"] != tracking.timestamp_seconds
            or not np.allclose(
                e["center_world_xy"], track.centroid_world_xy, rtol=0, atol=1e-9
            )
        ):
            raise ValueError(
                "advisory context requires its original current applied body prior"
            )
        if not 0.2 <= np.linalg.norm(track.velocity_world_xy) <= 2.0:
            return None
        return cls(
            key,
            frame,
            tracking.timestamp_seconds,
            track.centroid_world_xy,
            track.velocity_world_xy,
            e["radius_m"],
            e["centroid_uncertainty_prior_m"],
            track.point_count,
            e["original_person_camera_timestamp"],
            e["original_person_camera_frame_id"],
        )

    def validate_current(self, tracking, frame, source):
        t = next(
            (t for t in tracking.tracked_obstacles if t.track_id == self.track_id), None
        )
        if (
            frame != self.sensor_frame_id
            or source != self.sensor_timestamp_seconds
            or tracking.timestamp_seconds != source
            or t is None
            or t.point_count != self.current_support_count
            or not np.allclose(
                t.centroid_world_xy, self.center_world_xy, rtol=0, atol=1e-9
            )
            or not np.allclose(
                t.velocity_world_xy, self.velocity_world_xy, rtol=0, atol=1e-9
            )
        ):
            raise ValueError(
                "advisory context cannot transfer identity, velocity or renew epochs"
            )

    def preference(
        self,
        states,
        times,
        *,
        anchor,
        anchor_yaw,
        footprint,
        lookahead,
        source_age,
        maximum_outward_fraction,
        growth_prior_mps,
        nominal_route_direction_grid_xy,
    ):
        """Score the unchanged nominal's 1.2s forecast, never constrain execution."""
        states, times = np.asarray(states, float), np.asarray(times, float)
        anchor = np.asarray(anchor, float)
        route = np.asarray(nominal_route_direction_grid_xy, float)
        from robot.safety.safe_sage_configuration_space import RectangularRobotFootprint

        if not isinstance(footprint, RectangularRobotFootprint):
            raise TypeError("original rectangular footprint required")
        if (
            anchor.shape != (2,)
            or route.shape != (2,)
            or not np.isfinite(route).all()
            or not np.isfinite(anchor).all()
            or any(
                isinstance(v, (bool, np.bool_))
                or not isinstance(v, (int, float, np.number))
                or not math.isfinite(v)
                for v in (
                    anchor_yaw,
                    lookahead,
                    source_age,
                    maximum_outward_fraction,
                    growth_prior_mps,
                )
            )
        ):
            raise ValueError(
                "finite original source/anchor and explicit numerical priors required"
            )
        if (
            states.shape != (len(times), 3)
            or not np.isfinite(states).all()
            or len(times) < 2
            or not np.isfinite(times).all()
            or times[0] != 0
            or np.any(np.diff(times) <= 0)
            or abs(times[-1] - 1.2) > 1e-9
            or not 0 <= source_age <= 0.35 + 1e-9
            or not 0 < maximum_outward_fraction <= 1
            or not 0.25 <= growth_prior_mps <= 0.5
        ):
            raise ValueError(
                "advisory forecast requires original1.2s coverage and actual bounded source age"
            )
        c, s = math.cos(anchor_yaw), math.sin(anchor_yaw)
        rotation = np.array([[c, -s], [s, c]])
        center = (np.asarray(self.center_world_xy) - anchor) @ rotation
        velocity = np.asarray(self.velocity_world_xy) @ rotation
        direction = states[-1, :2] - states[0, :2]
        length = np.linalg.norm(direction)
        delta = center - states[0, :2]
        nominal_velocity = direction / times[-1]
        route_length = float(np.linalg.norm(route))
        active = bool(
            length > 1e-9
            and route_length > 1e-9
            and velocity @ (route / max(route_length, 1e-12))
            <= -math.cos(math.pi / 6) * np.linalg.norm(velocity)
            and delta @ (velocity - nominal_velocity) < 0
        )
        elapsed = times + source_age
        offset = center + elapsed[:, None] * velocity - states[:, :2]
        cs, sn = np.cos(states[:, 2]), np.sin(states[:, 2])
        local = np.stack(
            (
                cs * offset[:, 0] + sn * offset[:, 1],
                -sn * offset[:, 0] + cs * offset[:, 1],
            ),
            axis=1,
        )
        front, rear = (
            footprint.front_extent + lookahead,
            footprint.rear_extent - lookahead,
        )
        if min(front, rear, footprint.left_extent, footprint.right_extent) < 0:
            raise ValueError("original footprint/lookahead required")
        half = (
            np.array(
                [
                    (front + rear) / 2,
                    (footprint.left_extent + footprint.right_extent) / 2,
                ]
            )
            + footprint.safety_margin
        )
        local -= [
            (front - rear) / 2,
            (footprint.left_extent - footprint.right_extent) / 2,
        ]
        q = np.abs(local) - half
        distance = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(
            np.max(q, axis=1), 0
        )
        margin = distance - (self.radius_prior_m + growth_prior_mps * elapsed)
        fraction = (
            maximum_outward_fraction
            * float(np.clip(-margin.min() / self.centroid_uncertainty_prior_m, 0, 1))
            if active
            else 0.0
        )
        return fraction, dict(
            **self.diagnostic(),
            active=active,
            outward_fraction=fraction,
            minimum_nominal_predicted_margin_m=float(margin.min()),
            source_age_seconds=source_age,
            horizon_seconds=float(times[-1]),
            growth_prior_mps=growth_prior_mps,
            hypothesis_basis="current_raw_LiDAR_velocity_constant_motion_with_configured_radius_growth_PRIOR",
            reason="observed_oncoming_motion"
            if active
            else "not_current_approaching_oncoming_motion",
        )

    def diagnostic(self):
        return dict(
            **asdict(self),
            status="prior",
            centroid_basis="inferred_current_positive_LiDAR_cluster_centroid",
            velocity_basis="inferred_current_filtered_LiDAR_track_velocity",
            radius_basis="configured_current_human_body_and_centroid_uncertainty_PRIORS",
            scope="advisory reference only; no hard future containment claim; original occupancy/CRA/1.2s exact horizon/analytical checks unchanged",
        )
