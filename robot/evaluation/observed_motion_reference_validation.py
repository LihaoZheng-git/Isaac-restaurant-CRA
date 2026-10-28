"""Current-source advisory dataflow gate, not future-containment acceptance."""

import math

import numpy as np


def _current_selection(selection, source, key):
    """Audit both existing fresh and held-current modes, without new leases."""
    if selection.get("decision") not in {
        "selected_parallel_away_tangent",
        "held_current_track_policy_preference",
    }:
        return False
    try:
        original = selection["encounter_selected_timestamp"]
        lease = selection["configuration"]["maximum_encounter_seconds"]
        preferences = selection["preferences"]
        return bool(
            selection["original_selection_visual_person_verified"] is True
            and selection["encounter_geometry"] == "oncoming_parallel"
            and selection["source_timestamp"] == source
            and isinstance(original, (int, float))
            and not isinstance(original, bool)
            and math.isfinite(original)
            and original >= 0
            and 0 <= source - original <= lease + 1e-9
            and isinstance(lease, (int, float))
            and not isinstance(lease, bool)
            and 0 < lease <= 8.0
            and preferences.get(key, preferences.get(str(key))) in {"left", "right"}
        )
    except (KeyError, TypeError, ValueError):
        return False


def validate_observed_motion_reference(records):
    sources, failures, quiet = set(), [], set()
    origins = {}
    for row in records:
        d = row.get("diagnostics", row)
        source = row.get("sensor_timestamp", row.get("time"))
        context = d.get("observed_motion_reference")
        selection = d.get("motion_aware_passing") or {}
        selected = selection.get("decision") in {
            "selected_parallel_away_tangent",
            "held_current_track_policy_preference",
        }
        if context is None:
            if not selected:
                quiet.add((source, row["sensor_frame_id"]))
            continue
        key = (source, row["sensor_frame_id"])
        duplicate = key in sources
        sources.add(key)
        bodies = [
            b
            for b in d.get("human_body_envelope_priors", ())
            if b["track_id"] == context["track_id"]
        ]
        ok = (
            not duplicate
            and _current_selection(selection, source, context["track_id"])
            and len(bodies) == 1
            and context["sensor_frame_id"] == row["sensor_frame_id"]
            and math.isclose(source, context["sensor_timestamp_seconds"], abs_tol=1e-9)
            and context["status"] == "prior"
        )
        origin = selection.get("encounter_selected_timestamp")
        if (
            selected
            and context["track_id"] in origins
            and selection["decision"] == "held_current_track_policy_preference"
        ):
            ok = ok and origin == origins[context["track_id"]]
        if ok:
            origins[context["track_id"]] = origin
        if ok:
            b = bodies[0]
            ok = (
                b.get("applied_to_occupancy") is True
                and b["lidar_timestamp_seconds"] == source
                and b["current_support_count"] == context["current_support_count"] >= 20
                and np.allclose(
                    b["center_world_xy"], context["center_world_xy"], rtol=0, atol=1e-9
                )
                and np.allclose(
                    b["velocity_world_xy"],
                    context["velocity_world_xy"],
                    rtol=0,
                    atol=1e-9,
                )
                and b["radius_m"] == context["radius_prior_m"]
                and b["centroid_uncertainty_prior_m"]
                == context["centroid_uncertainty_prior_m"]
                and b["original_person_camera_timestamp"]
                == context["original_person_camera_timestamp"]
                and b["original_person_camera_frame_id"]
                == context["original_person_camera_frame_id"]
            )
        if not ok:
            failures.append(dict(source=source, track_id=context["track_id"]))
    checks = dict(
        twenty_selected_current_sources=len(sources) >= 20,
        no_source_or_identity_mismatch=not failures,
        quiet_or_unselected_without_context=len(quiet) >= 10,
    )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        selected_sources=len(sources),
        quiet_or_unselected_sources=len(quiet),
        failures=failures,
        scope="Current-source advisory dataflow only; predictions are uncalibrated hypotheses, not hard future geometry or complete physical perception. Original1.2s hard safety/execution/arrival gates remain.",
    )
