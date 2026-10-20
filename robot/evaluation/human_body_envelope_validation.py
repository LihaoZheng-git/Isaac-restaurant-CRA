"""Independent actor-envelope audit; oracle records never supply runtime inputs."""

import math
import numpy as np


def validate_human_body_envelopes(records, actor_records, *, reference_radius_m):
    if not math.isfinite(reference_radius_m) or reference_radius_m <= 0:
        raise ValueError("reference radius must be positive and finite")
    rows = []
    for record in records:
        envelopes = record.get("diagnostics", {}).get("human_body_envelope_priors", ())
        if not envelopes:
            continue
        timestamp = record["sensor_timestamp"]
        reference = (
            min(actor_records, key=lambda item: abs(item["time"] - timestamp))
            if actor_records
            else None
        )
        for item in envelopes:
            aligned = (
                reference is not None and abs(reference["time"] - timestamp) <= 0.05
            )
            error = (
                float(
                    np.linalg.norm(
                        np.asarray(item["center_world_xy"]) - reference["actor_xyz"][:2]
                    )
                )
                if aligned
                else None
            )
            margin = (
                item["radius_m"] - reference_radius_m - error
                if error is not None
                else None
            )
            rows.append(
                dict(
                    timestamp=timestamp,
                    track_id=item["track_id"],
                    aligned=aligned,
                    status=item["status"],
                    center_error_m=error,
                    containment_margin_m=margin,
                    current_source=abs(item["lidar_timestamp_seconds"] - timestamp)
                    <= 1e-9,
                )
            )
    checks = dict(
        three_current_envelopes=len(rows) >= 3,
        original_prior_provenance=bool(rows)
        and all(row["status"] == "prior" for row in rows),
        current_lidar_sources=bool(rows) and all(row["current_source"] for row in rows),
        bounded_independent_actor_envelope=bool(rows)
        and all(
            row["aligned"] and row["containment_margin_m"] >= -1e-9 for row in rows
        ),
    )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        rows=rows,
        scope="independent configured actor-envelope containment audit; no actor truth in runtime; not calibrated human anatomy or universal coverage",
    )
