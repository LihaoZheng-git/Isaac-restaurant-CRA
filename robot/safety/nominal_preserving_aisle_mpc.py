"""Waypoint-reference MPC with a committed static passing region."""

from dataclasses import asdict, dataclass

import numpy as np

from robot.safety.committed_aisle_mpc import AisleCorridor, CommittedAisleMPC
from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPCResult


@dataclass(frozen=True)
class NominalAisleRegion(AisleCorridor):
    """Selected-side limits built around a checked geometric witness.

    The original footprint checks still determine free space inside the region.
    The witness is a candidate and a continuation probe, never a cost target.
    """

    def contains(self, states):
        x = states[:, 0]
        witness_y, _ = self.sample(x)
        side = 1.0 if self.pass_side == "left" else -1.0
        signed_y = side * (states[:, 1] - self.route_y)
        signed_witness = side * (witness_y - self.route_y)
        entry = np.clip(
            (x - self.entry_x) / (self.lane_start_x - self.entry_x), 0.0, 1.0
        )
        exit_fraction = np.clip(
            (self.exit_x - x) / (self.exit_x - self.lane_end_x), 0.0, 1.0
        )
        passing = np.minimum(entry, exit_fraction)
        entry_minimum = min(0.0, side * (self.entry_y - self.route_y)) - 0.065
        side_minimum = entry_minimum + (0.02 - entry_minimum) * passing
        lower = np.maximum(signed_witness - 0.065, side_minimum)
        upper = signed_witness + 0.12
        return bool(np.all((signed_y >= lower - 1e-9) & (signed_y <= upper + 1e-9)))


class NominalPreservingAisleMPC(CommittedAisleMPC):
    """Keep an admissible native horizon, otherwise rank checked alternatives.

    The search is over a finite candidate set under the original MPC cost.
    Positive progress is enforced after side acquisition; acquisition permits
    an initial checked turn with forward motion later in the horizon.
    """

    def _candidate_corridors(self, initial, goal, field):
        for witness in super()._candidate_corridors(initial, goal, field):
            yield NominalAisleRegion(**asdict(witness))

    def corridor_valid(self, initial, commands, corridor):
        states = self._rollout(initial, commands)
        swept, times = self._dense_chassis_queries(states, commands)
        swept = self.route_states(swept)
        dx = np.diff(swept[:, 0])
        passing = swept[:-1, 0] >= corridor.lane_start_x
        return (
            corridor.contains(swept)
            and np.all(dx >= -1e-9)
            and swept[-1, 0] - swept[0, 0] >= self.minimum_progress_mps * times[-1]
            and np.all(
                dx[passing] >= self.minimum_progress_mps * np.diff(times)[passing]
            )
        )

    def _continuation_available(self, terminal, corridor, field, speed):
        """Check a finite geometric continuation from the candidate's endpoint."""
        continuation = self._corridor_commands(terminal, corridor, speed)
        if not self.corridor_valid(terminal, continuation, corridor):
            return False
        return not self._inter_sample_unsafe_mask(
            self._rollout(terminal, continuation),
            np.zeros(self.config.horizon_steps + 1, dtype=bool),
            commands=continuation,
            configuration_space=field["configuration_space"],
            h_full=field["h_full"],
            stop_on_first_unsafe=True,
        ).any()

    def continuation_valid_at_receipt(self, initial, commands, plan, field):
        terminal = self._rollout(initial, commands)[-1]
        return self._continuation_available(
            terminal,
            plan.corridor,
            field,
            plan.committed_bypass["continuation_speed_mps"],
        )

    def _first_command_admissible(self, initial, commands, corridor, field, speed):
        query = self._safety_states(initial[None, :])[0]
        filtered = self.backup_filter.filter_command(
            *commands[0],
            query_position_grid_xy=query[:2],
            query_yaw_radians=float(query[2]),
            **field,
        )
        if not (
            filtered.constraint_feasible
            and filtered.constraint_satisfied
            and not filtered.emergency_stop_active
            and np.isfinite(filtered.cbf_margin)
        ):
            return False
        executable = (filtered.safe_linear_velocity, filtered.safe_angular_velocity)
        if np.array_equal(commands[0], executable):
            return True
        amended = commands.copy()
        amended[0] = executable
        return (
            self.corridor_valid(initial, amended, corridor)
            and self.revalidate_commands(initial, amended, **field)
            and self._continuation_available(
                self._rollout(initial, amended)[-1], corridor, field, speed
            )
        )

    def _alternative_sequences(self, initial, nominal, corridor, speed, native, warm):
        yield self._clip_commands(nominal)
        if warm is not None:
            yield self._clip_commands(
                self._command_sequence(warm, name="initial_command_guess")
            )
        reference = native.command_sequence if native.success else nominal
        for fraction in (1.0, 0.8, 0.6, 0.4):
            witness = self._corridor_commands(initial, corridor, speed * fraction)
            yield witness
            for weight in (0.25, 0.5, 0.75):
                yield self._clip_commands(reference + weight * (witness - reference))
        if self.route_states(initial)[0] < corridor.lane_start_x:
            side = 1.0 if corridor.pass_side == "left" else -1.0
            for omega in (0.4, 0.7):
                turn = self._clip_commands(np.array([[0.0, side * omega]]))[0]
                turned, _, _ = self._interval_dynamics(initial, turn)
                following = self._corridor_commands(turned, corridor, speed)
                yield np.vstack((turn, following[:-1]))

    def _plan_in_corridor(
        self, initial, nominal, corridor, field, mode, baseline, warm
    ):
        native = baseline()
        speed = min(
            self.config.maximum_linear_velocity, max(0.08, float(np.max(nominal[:, 0])))
        )
        station = self.route_states(initial)[0]
        phase = "acquire_side" if station < corridor.lane_start_x else mode
        details = dict(
            progress_phase=phase,
            native_first_command=native.first_command,
            native_objective_value=native.objective_value
            if np.isfinite(native.objective_value)
            else None,
            continuation_speed_mps=speed,
        )
        if (
            native.success
            and not native.fallback_active
            and self.corridor_valid(initial, native.command_sequence, corridor)
            and self._continuation_available(
                native.predicted_chassis_states[-1], corridor, field, speed
            )
            and self._first_command_admissible(
                initial, native.command_sequence, corridor, field, speed
            )
        ):
            return self._result(
                native, mode, corridor, nominal_selection="native_preserved", **details
            )

        nominal_states = self._rollout(initial, self._clip_commands(nominal))
        ranked = []
        seen = set()
        for sequence in self._alternative_sequences(
            initial, nominal, corridor, speed, native, warm
        ):
            key = sequence.tobytes()
            if key in seen or not self.corridor_valid(initial, sequence, corridor):
                continue
            seen.add(key)
            states = self._rollout(initial, sequence)
            cost = self._objective_value(
                sequence,
                nominal,
                chassis_states=states,
                target_chassis_states=nominal_states,
            )
            ranked.append((cost, sequence, states[-1]))

        for cost, sequence, terminal in sorted(
            ranked, key=lambda candidate: candidate[0]
        ):
            trajectory = self._evaluate_trajectory(
                initial, sequence, **field, require_feasible=True
            )
            if trajectory is None or not self._trajectory_is_safe(trajectory):
                continue
            if not self._continuation_available(terminal, corridor, field, speed):
                continue
            if not self._first_command_admissible(
                initial, sequence, corridor, field, speed
            ):
                continue
            result = SafeSAGEUnicycleMPCResult(
                trajectory.commands,
                trajectory.chassis_states,
                trajectory.safety_states,
                trajectory.safety_values,
                trajectory.margins,
                tuple(nominal[0]),
                cost,
                0,
                0,
                True,
                True,
                False,
                False,
                "checked_minimum_nominal_cost_candidate",
            )
            return self._result(
                result,
                mode,
                corridor,
                nominal_selection="minimum_cost_checked_candidate",
                ranked_candidate_count=len(ranked),
                **details,
            )
        return None
