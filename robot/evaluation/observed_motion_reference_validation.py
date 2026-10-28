"""Current-source advisory dataflow gate, not future-containment acceptance."""

import math

import numpy as np


def validate_observed_motion_reference(records):
    sources, failures, quiet = set(), [], set()
    for row in records:
        d = row.get("diagnostics", row)
        source = row.get("sensor_timestamp", row.get("time"))
        context = d.get("observed_motion_reference")
        selection = d.get("motion_aware_passing") or {}
        selected = selection.get("decision") == "selected_parallel_away_tangent"
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
            and selected
            and len(bodies) == 1
            and context["sensor_frame_id"] == row["sensor_frame_id"]
            and math.isclose(source, context["sensor_timestamp_seconds"], abs_tol=1e-9)
            and context["status"] == "prior"
        )
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
