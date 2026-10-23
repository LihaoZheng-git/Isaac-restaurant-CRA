"""ORACLE-only rear-pass evaluation adapter; deliberately not a native runtime mode."""

from dataclasses import dataclass, fields, replace
import numpy as np

from robot.evaluation.encounter_guidance import apply_encounter_guidance
from robot.safety.checked_rear_pass_reference import (
    CheckedRearPassReferenceConfig,
    rear_pass_context,
    search_checked_rear_pass_reference,
)
from robot.safety.crossing_maneuver import apply_crossing_maneuver
from robot.safety.safe_sage_runtime import SafeSAGEControlRuntime
from robot.safety.safe_sage_unicycle_mpc import SafeSAGEUnicycleMPCResult


def rear_pass_policy_adapter(
    guidance,
    audit,
    classification,
    passing,
    *,
    source_time,
    context,
    configuration=None,
):
    """Only this fixed world=grid ORACLE fixture supplies these qualified tracks."""
    changed, rows, ids = apply_encounter_guidance(
        guidance, audit, classification, passing, source_time=source_time
    )
    tracks = {t.track_id: t for t in context["tracking"].tracked_obstacles}
    decisions = {
        key: rear_pass_context(
            context["state"],
            context["goal"],
            tracks[key].centroid_world_xy,
            tracks[key].velocity_world_xy,
            (1.0, 0.0),
            configuration,
        )
        for key in ids
    }
    active = tuple(key for key in ids if decisions[key]["mode"] != "none")
    for row in rows:
        cue = row["encounter_guidance_ablation"]
        owner = cue["parent_track_id"]
        if owner not in decisions:
            continue
        decision = decisions[owner]
        if decision["mode"] != "wait":
            key = row["track_or_group_id"]
            changed[key] = guidance[key]
            row["lambda"] = 1.25
            cue.update(
                lambda_suppressed=False,
                basis="rear_reference_or_released_maneuver_normal_guidance",
            )
        row["rear_pass_behavior"] = dict(
            **decision,
            source_timestamp_seconds=source_time,
            original_selection_timestamp_seconds=passing[
                "encounter_selected_timestamp"
            ],
        )
    return changed, rows, active


@dataclass(frozen=True)
class RearPassBenchmarkMPCResult(SafeSAGEUnicycleMPCResult):
    rear_pass_reference: dict
    crossing_maneuver: dict


class CheckedRearPassBenchmarkRuntime(SafeSAGEControlRuntime):
    """Experimental selection utility; not the primary MPC objective or sensors."""

    requires_nominal_goal_context = True

    def __init__(
        self, *, mpc, backup_filter, crossing_strategy=None, rear_pass_config=None
    ):
        super().__init__(
            mpc=mpc, backup_filter=backup_filter, crossing_strategy="legacy"
        )
        self.rear_pass_config = rear_pass_config or CheckedRearPassReferenceConfig()

    def plan(self, initial, nominal, field, *, nominal_goal_grid_xy):
        result = super().plan(initial, nominal, field)
        ids = field.crossing_yield_track_ids or ()
        audit = dict(
            selected=False,
            reason="no_unique_current_crossing",
            source_timestamp_seconds=field.sensor_timestamp_seconds,
            original_selection_timestamp_seconds=field.motion_passing_reference_source_timestamp_seconds,
            scope="ORACLE-only complete checked reference selected for a plan; actual filtered execution is recorded separately",
        )
        if len(ids) == 1:
            track = next(
                t
                for t in field.tracking_result.tracked_obstacles
                if t.track_id == ids[0]
            )
            if (
                track.point_count >= 20
                and track.matched_previous_track
                and track.observation_count >= 3
            ):
                options = dict(
                    configuration_space=field.pipeline.configuration_space,
                    h_full=field.pipeline.h_full,
                    guidance_fields=field.pipeline.guidance_fields,
                    temporal_safety_derivative=field.temporal_safety_derivative,
                )
                search = search_checked_rear_pass_reference(
                    self.mpc,
                    initial,
                    nominal_goal_grid_xy,
                    track.centroid_world_xy,
                    track.velocity_world_xy,
                    (1.0, 0.0),
                    options,
                    lambda command: self.filter_command(command, initial, field),
                    config=self.rear_pass_config,
                )
                audit.update(search.diagnostics, track_id=ids[0])
                if search.trajectory is not None:
                    value = search.trajectory

                    def frozen(array):
                        array = np.array(array, copy=True)
                        array.setflags(write=False)
                        return array

                    result = replace(
                        result,
                        command_sequence=frozen(value.commands),
                        predicted_chassis_states=frozen(value.chassis_states),
                        predicted_safety_states=frozen(value.safety_states),
                        predicted_safety_values=frozen(value.safety_values),
                        dtcbf_margins=frozen(value.margins),
                        objective_value=self.mpc._objective_value(
                            value.commands, self.mpc._nominal_sequence(nominal)
                        ),
                        success=True,
                        fallback_active=False,
                        intervention_active=True,
                        status="experimental_checked_rear_pass_reference",
                    )
                elif search.diagnostics["context"]["mode"] in {"wait", "rear"}:
                    result = apply_crossing_maneuver(
                        self.mpc,
                        result,
                        initial,
                        nominal,
                        field,
                        "yield-then-go",
                        lambda: self.filter_command((0.0, 0.0), initial, field),
                    )
        return RearPassBenchmarkMPCResult(
            **{
                f.name: getattr(result, f.name)
                for f in fields(SafeSAGEUnicycleMPCResult)
            },
            rear_pass_reference=audit,
            crossing_maneuver=getattr(result, "crossing_maneuver", {}),
        )
