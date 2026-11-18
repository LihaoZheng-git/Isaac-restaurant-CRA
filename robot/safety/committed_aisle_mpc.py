"""Finite sensor-checked passing corridor for the static straight aisle trial."""

from dataclasses import asdict, dataclass, fields, replace
from functools import cache
import math

import numpy as np

from robot.safety.field_alignment_mpc import FieldAlignmentMPC
from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPCResult


def checked_exit_receipt(value):
    """Validate the immutable identity of one physically executed corridor exit."""
    if not isinstance(value, tuple) or len(value) != 5:
        raise ValueError("committed exit receipt requires five immutable fields")
    guide, serial, exit_x, side, phase = value
    if (
        type(guide) is not int
        or guide < 0
        or type(serial) is not int
        or serial < 0
        or type(exit_x) is not float
        or not math.isfinite(exit_x)
        or side not in ("left", "right")
        or phase != "verified_exit"
    ):
        raise ValueError("invalid committed exit receipt")
    return value


@dataclass(frozen=True)
class AisleGuideFrame:
    """A fixed route segment in the original field grid's coordinates."""

    start_grid_xy: tuple[float, float]
    end_grid_xy: tuple[float, float]

    def __post_init__(self):
        for name in ("start_grid_xy", "end_grid_xy"):
            point = np.asarray(getattr(self, name), dtype=float)
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError("aisle guide endpoints must be finite grid XY")
            object.__setattr__(self, name, tuple(map(float, point)))
        if math.dist(self.start_grid_xy, self.end_grid_xy) < 1e-6:
            raise ValueError("aisle guide endpoints must define a direction")

    @property
    def heading_grid(self):
        start_x, start_y = self.start_grid_xy
        end_x, end_y = self.end_grid_xy
        return math.atan2(end_y - start_y, end_x - start_x)

    def to_route(self, states):
        return self._transform(states, inverse=False)

    def to_grid(self, states):
        return self._transform(states, inverse=True)

    def _transform(self, states, *, inverse):
        states = np.asarray(states, dtype=float)
        yaw = self.heading_grid
        c, s = math.cos(yaw), math.sin(yaw)
        basis = np.array([[c, -s], [s, c]])
        if inverse:
            xy = states[..., :2] @ basis.T + self.start_grid_xy
            heading = states[..., 2:] + yaw
        else:
            xy = (states[..., :2] - self.start_grid_xy) @ basis
            heading = states[..., 2:] - yaw
        return np.concatenate((xy, heading), axis=-1)

    def preview_endpoint(self, grid, start, route_y, reserve):
        """Clip a forward route ray to the interior of the real grid."""
        point = self.to_grid((start, route_y))
        yaw = self.heading_grid
        direction = (math.cos(yaw), math.sin(yaw))
        lower = np.array((grid.minimum_x, grid.minimum_y)) + reserve
        upper = np.array((grid.maximum_x, grid.maximum_y)) - reserve
        if not np.all((lower <= point) & (point <= upper)):
            return start
        distances = [
            ((hi if d > 0 else lo) - xy) / d
            for xy, d, lo, hi in zip(point, direction, lower, upper)
            if abs(d) > 1e-12
        ]
        return float(start + min(distances))


@dataclass(frozen=True)
class AisleCorridor:
    """A fixed entry, passing lane, and exit in the straight route frame."""

    entry_x: float
    lane_start_x: float
    lane_end_x: float
    exit_x: float
    entry_y: float
    lane_y: float
    lane_end_y: float
    route_y: float
    half_width: float = 0.035

    def sample(self, x):
        def transition(start, end, y0, y1, slope0, slope1):
            width = end - start
            t = np.clip((x - start) / width, 0.0, 1.0)
            blend = t * t * (3.0 - 2.0 * t)
            y = y0 + (y1 - y0) * blend
            y += width * t * (1.0 - t) * (slope0 * (1.0 - t) - slope1 * t)
            slope = 6.0 * t * (1.0 - t) * (y1 - y0) / width
            slope += (1.0 - 4.0 * t + 3.0 * t * t) * slope0 + (
                3.0 * t * t - 2.0 * t
            ) * slope1
            return y, slope

        approach_slope = (self.lane_end_y - self.lane_y) / (
            self.lane_end_x - self.lane_start_x
        )
        enter = transition(
            self.entry_x,
            self.lane_start_x,
            self.entry_y,
            self.lane_y,
            0.0,
            approach_slope,
        )
        passing = transition(
            self.lane_start_x,
            self.lane_end_x,
            self.lane_y,
            self.lane_end_y,
            approach_slope,
            0.0,
        )
        leave = transition(
            self.lane_end_x, self.exit_x, self.lane_end_y, self.route_y, 0.0, 0.0
        )
        lateral, slope = (
            np.where(x < self.lane_start_x, a, np.where(x < self.lane_end_x, b, c))
            for a, b, c in zip(enter, passing, leave)
        )
        return lateral, np.arctan(slope)

    def contains(self, states):
        y, _ = self.sample(states[:, 0])
        return bool(np.all(np.abs(states[:, 1] - y) <= self.half_width + 1e-9))

    @property
    def pass_side(self):
        return "left" if self.lane_y > self.route_y else "right"


@dataclass(frozen=True)
class CommittedAisleResult(SafeSAGEUnicycleMPCResult):
    committed_bypass: dict
    corridor: AisleCorridor | None


class CommittedAisleMPC(FieldAlignmentMPC):
    """Keep the current N-step safety checks and select one static corridor.

    The corridor preview checks the current field only. All issued motion
    still requires exact short-horizon and analytical checks in live fields.
    Progress can be withdrawn on infeasibility. The preview gives no
    recursive-feasibility or moving-obstacle guarantee.
    """

    minimum_progress_mps = 0.045

    def __init__(self, *args, backup_filter, guide_frame=None, **kwargs):
        if guide_frame is not None and not isinstance(guide_frame, AisleGuideFrame):
            raise TypeError("guide_frame must be an AisleGuideFrame")
        super().__init__(*args, **kwargs)
        self.guide_frame = guide_frame
        self.backup_filter = backup_filter
        self.corridor = None
        self._guide_index = None
        self._commitment_serial = 0
        self._last_exit_receipt = None

    def _commitment_identity(self, corridor):
        return (
            self._guide_index,
            getattr(self, "_commitment_serial", 0),
            float(corridor.exit_x),
            corridor.pass_side,
        )

    def route_states(self, states):
        """Project chassis poses onto the frozen guide, if configured."""
        # Trusted static snapshots predate the optional constructor fields.
        frame = getattr(self, "guide_frame", None)
        return states if frame is None else frame.to_route(states)

    def _unsafe_poses(self, x, y, yaw, field, *, tube=0.0):
        # Same spatial and angular perturbations as the MPC's dense check,
        # plus lateral tube probes for the stationary preview.
        grid = field["configuration_space"].occupancy_grids[0].config
        p = self.config.configuration_position_padding_m
        a = self.config.configuration_yaw_padding_radians
        yaw_spacing = math.tau / field["h_full"].yaw_slice_count
        spatial = np.linspace(-p, p, 2 * max(1, math.ceil(2 * p / grid.resolution)) + 1)
        angular = np.linspace(-a, a, 2 * max(1, math.ceil(2 * a / yaw_spacing)) + 1)
        lookahead = self.config.lookahead_distance
        queries = np.column_stack(
            (x + lookahead * np.cos(yaw), y + lookahead * np.sin(yaw), yaw)
        )
        frame = getattr(self, "guide_frame", None)
        if frame is None:
            lateral = np.unique(
                (spatial[:, None] + np.linspace(-tube, tube, 5)).ravel()
            )
            offsets = np.stack(
                np.meshgrid(spatial, lateral, angular, indexing="ij"), axis=-1
            ).reshape(-1, 3)
        else:
            # The tube is lateral in the guide frame. Spatial padding remains
            # aligned to the original MPC grid, independent of guide heading.
            shifts = np.linspace(-tube, tube, 5) if tube else np.zeros(1)
            tube_queries = queries[:, None, :] + np.column_stack(
                (np.zeros(len(shifts)), shifts, np.zeros(len(shifts)))
            )
            queries = frame.to_grid(tube_queries).reshape(-1, 3)
            offsets = np.stack(
                np.meshgrid(spatial, spatial, angular, indexing="ij"), axis=-1
            ).reshape(-1, 3)
        try:
            return (
                self.paper_filter.unsafe_configuration_mask(
                    (queries[:, None, :] + offsets).reshape(-1, 3),
                    configuration_space=field["configuration_space"],
                    h_full=field["h_full"],
                )
                .reshape(len(x), -1)
                .any(axis=1)
            )
        except ValueError:
            return np.ones(len(x), dtype=bool)

    @staticmethod
    def _samples(start, end):
        return np.linspace(start, end, max(2, math.ceil((end - start) / 0.025) + 1))

    def _candidate_corridors(self, initial, goal, field):
        initial = self.route_states(initial)
        goal = self.route_states(goal)
        start = float(initial[0])
        route_y = float(goal[1])
        grid = field["configuration_space"].occupancy_grids[0].config
        frame = getattr(self, "guide_frame", None)
        if frame is None:
            grid_end = grid.maximum_x - self.config.lookahead_distance - 0.06
        else:
            reserve = self.config.lookahead_distance + max(
                0.06,
                self.config.configuration_position_padding_m
                + math.sqrt(2.0) * grid.resolution,
            )
            grid_end = frame.preview_endpoint(grid, start, route_y, reserve)
        endpoint = min(float(goal[0]) - 0.13, start + 4.5, grid_end)
        if endpoint - start < 1.5:
            return
        line = self._samples(start, endpoint)
        blocked = np.flatnonzero(
            self._unsafe_poses(
                line,
                np.full_like(line, route_y),
                np.zeros_like(line),
                field,
            )
        )
        if not len(blocked):
            return
        first, last = float(line[blocked[0]]), float(line[blocked[-1]])
        if (
            not 0.55 <= first - start <= 2.7
            or last - first > 2.2
            or endpoint - last < 0.65
        ):
            return
        # Check complete entry/lane/return paths against all occupied cells.
        # Try smaller deviations first, then the opposite side before locking.
        lane_end = last + 0.04
        exit_x = min(endpoint - 0.025, lane_end + 1.20)
        for width in (0.035, 0.02, 0.015):
            for shift in (0.26, -0.26, 0.32, -0.32, 0.38, -0.38, 0.44, -0.44):
                for lead in (0.67, 0.40):
                    lane_start = first - lead
                    if lane_start - start < 0.25:
                        continue
                    corridor = AisleCorridor(
                        start,
                        lane_start,
                        lane_end,
                        exit_x,
                        float(initial[1]),
                        route_y + 0.32 * shift,
                        route_y + shift,
                        route_y,
                        half_width=width,
                    )
                    x = self._samples(start, exit_x + 0.025)
                    y, yaw = corridor.sample(x)
                    if (
                        np.max(np.abs(yaw)) <= 0.65
                        and not self._unsafe_poses(
                            x,
                            y,
                            yaw,
                            field,
                            tube=corridor.half_width,
                        ).any()
                    ):
                        yield corridor

    def _corridor_commands(self, initial, corridor, speed):
        commands = np.empty((self.config.horizon_steps, 2))
        state = initial
        for index in range(len(commands)):
            station, lateral, route_yaw = self.route_states(state)
            y, yaw = corridor.sample(station)
            _, ahead_yaw = corridor.sample(station + speed * self.config.time_step)
            heading = yaw + math.atan2(1.8 * (y - lateral), 0.45)
            error = math.remainder(heading - route_yaw, math.tau)
            omega = (ahead_yaw - yaw) / self.config.time_step + 3.5 * error
            commands[index] = self._clip_commands(np.array([[speed, omega]]))[0]
            state = self._interval_dynamics(state, commands[index])[0]
        return commands

    def corridor_valid(self, initial, commands, corridor):
        states = self._rollout(initial, commands)
        swept, times = self._dense_chassis_queries(states, commands)
        swept = self.route_states(swept)
        progress = np.diff(swept[:, 0]) / np.diff(times)
        return corridor.contains(swept) and bool(
            np.all(progress >= self.minimum_progress_mps)
        )

    def executable_corridor_command(self, initial, command, plan):
        """Apply the side and progress checks after the final analytical filter."""
        if plan.corridor is None:
            return True
        sequence = plan.command_sequence.copy()
        sequence[0] = command
        return self.corridor_valid(initial, sequence, plan.corridor)

    def _filter_preserves_first(self, initial, commands, field):
        query = self._safety_states(initial[None, :])[0]
        filtered = self.backup_filter.filter_command(
            *commands[0],
            query_position_grid_xy=query[:2],
            query_yaw_radians=float(query[2]),
            **field,
        )
        return (
            filtered.constraint_feasible
            and filtered.constraint_satisfied
            and not filtered.emergency_stop_active
            and np.allclose(
                commands[0],
                (filtered.safe_linear_velocity, filtered.safe_angular_velocity),
                atol=1e-9,
                rtol=0.0,
            )
        )

    def _result(self, result, status, corridor, **details):
        frame = getattr(self, "guide_frame", None)
        if frame is not None:
            details["guide_frame"] = asdict(frame)
        return CommittedAisleResult(
            **{
                item.name: getattr(result, item.name)
                for item in fields(SafeSAGEUnicycleMPCResult)
            },
            corridor=corridor,
            committed_bypass=dict(
                mode=status,
                corridor=asdict(corridor) if corridor else None,
                pass_side=corridor.pass_side if corridor else None,
                commitment_identity=self._commitment_identity(corridor)
                if corridor
                else None,
                minimum_forward_progress_mps=self.minimum_progress_mps,
                **details,
            ),
        )

    def _checked_corridor_plan(self, initial, nominal, corridor, field, mode):
        speed = min(self.config.maximum_linear_velocity, float(np.max(nominal[:, 0])))
        for fraction in (1.0, 0.8, 0.6, 0.4):
            commands = self._corridor_commands(initial, corridor, speed * fraction)
            if not self.corridor_valid(initial, commands, corridor):
                continue
            trajectory = self._evaluate_trajectory(
                initial, commands, **field, require_feasible=True
            )
            if trajectory is None or not self._trajectory_is_safe(trajectory):
                continue
            if not self._filter_preserves_first(initial, commands, field):
                continue
            result = SafeSAGEUnicycleMPCResult(
                trajectory.commands,
                trajectory.chassis_states,
                trajectory.safety_states,
                trajectory.safety_values,
                trajectory.margins,
                tuple(nominal[0]),
                self._objective_value(commands, nominal),
                0,
                0,
                True,
                True,
                False,
                False,
                "exact_safe_committed_corridor_progress",
            )
            return self._result(result, mode, corridor, speed_fraction=fraction)
        return None

    def _plan_in_corridor(
        self, initial, nominal, corridor, field, mode, baseline, warm
    ):
        return self._checked_corridor_plan(initial, nominal, corridor, field, mode)

    def plan(
        self,
        initial_chassis_state_grid,
        nominal_command,
        *,
        configuration_space,
        h_full,
        guidance_fields,
        temporal_safety_derivative=0.0,
        nominal_goal_grid_xy=None,
        route_goal_grid_xy=None,
        active_guide_index=None,
        confirmed_committed_exit=None,
        selected_passing_tangents=(),
        preferred_lateral_direction_grid_y=None,
        initial_command_guess=None,
    ):
        initial = np.asarray(initial_chassis_state_grid, dtype=float)
        initial_route = self.route_states(initial)
        nominal = self._nominal_sequence(nominal_command)
        field = dict(
            configuration_space=configuration_space,
            h_full=h_full,
            guidance_fields=guidance_fields,
            temporal_safety_derivative=temporal_safety_derivative,
        )
        plan_baseline = super().plan

        @cache
        def baseline():
            return plan_baseline(
                initial,
                nominal,
                nominal_goal_grid_xy=nominal_goal_grid_xy,
                route_goal_grid_xy=route_goal_grid_xy,
                active_guide_index=active_guide_index,
                initial_command_guess=initial_command_guess,
                preferred_lateral_direction_grid_y=preferred_lateral_direction_grid_y,
                **field,
            )

        if selected_passing_tangents:
            raise ValueError(
                "the static committed corridor cannot consume a moving passing selection"
            )
        if confirmed_committed_exit is not None:
            receipt = checked_exit_receipt(confirmed_committed_exit)
            if receipt != getattr(self, "_last_exit_receipt", None):
                corridor = self.corridor
                if (
                    corridor is None
                    or receipt
                    != (*self._commitment_identity(corridor), "verified_exit")
                    or initial_route[0] < corridor.exit_x
                    or not corridor.contains(initial_route[None, :])
                ):
                    return self._result(
                        self._fallback_result(
                            initial, nominal, status="committed_exit_receipt_rejected"
                        ),
                        "progress_withdrawn",
                        corridor,
                    )
                self._last_exit_receipt = receipt
                self.corridor = None
        if active_guide_index != self._guide_index:
            self.corridor = None
            self._guide_index = active_guide_index
        if self.corridor is None:
            for candidate in self._candidate_corridors(
                initial, route_goal_grid_xy, field
            ):
                result = self._plan_in_corridor(
                    initial,
                    nominal,
                    candidate,
                    field,
                    "active_pass",
                    baseline,
                    initial_command_guess,
                )
                if result is not None:
                    self._commitment_serial = getattr(self, "_commitment_serial", 0) + 1
                    self.corridor = candidate
                    return replace(
                        result,
                        committed_bypass={
                            **result.committed_bypass,
                            "commitment_identity": self._commitment_identity(candidate),
                        },
                    )
            return self._result(baseline(), "cruise", None)

        corridor = self.corridor
        exited = initial_route[0] >= corridor.exit_x and corridor.contains(
            initial_route[None, :]
        )
        result = self._plan_in_corridor(
            initial,
            nominal,
            corridor,
            field,
            "verified_exit" if exited else "active_pass",
            baseline,
            initial_command_guess,
        )
        if result is not None:
            return result
        return self._result(
            self._fallback_result(
                initial, nominal, status="committed_progress_unavailable"
            ),
            "progress_withdrawn",
            corridor,
        )
