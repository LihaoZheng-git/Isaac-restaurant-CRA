"""Independent actor-envelope audit; oracle records never supply runtime inputs."""

import math
import numpy as np


def validate_tracking_uncertainty(records, actor_records, *, reference_radius_m):
    """Independent current/available-future containment, never estimator input."""
    rows = []
    for record in records:
        diagnostics = record.get('diagnostics', record)
        for region in diagnostics.get('tracking_uncertainty_regions', ()):
            t = region['evaluation_timestamp_seconds']
            nearest = min(actor_records, key=lambda r: abs(r['time']-t)) if actor_records else None
            future = [r for r in actor_records if t <= r['time'] <= t+region['checked_horizon_seconds']]
            aligned = nearest is not None and abs(nearest['time']-t) <= .05
            samples = ([nearest] if aligned else []) + future
            margins = [region['radius_m']-reference_radius_m-np.linalg.norm(
                np.array(region['center_world_xy'])-r['actor_xyz'][:2]) for r in samples]
            expected = region['body_radius_prior_m'] + region['maximum_speed_prior_mps'] * (
                t-region['original_lidar_timestamp_seconds']+region['checked_horizon_seconds'])
            rows.append(dict(source_time=t, track_id=region['track_id'],
                aligned=aligned, source_age=region['source_age_seconds'],
                minimum_containment_margin_m=float(min(margins)) if margins else None,
                available_future_queries=len(future),
                full_future_available=bool(actor_records and actor_records[-1]['time'] >= t+region['checked_horizon_seconds']),
                explicit_original_source=region['status']=='prior' and not region['current_identity_verified']
                    and not region['current_motion_measured'] and region['source_age_seconds'] >= 0
                    and math.isclose(region['radius_m'], expected, abs_tol=1e-9),
                original_risk_floor=region['applied_cra_phi'] >= region['original_applied_risk']))
    return dict(passed=all(r['aligned'] and r['explicit_original_source'] and r['original_risk_floor']
        and r['minimum_containment_margin_m'] >= -1e-9 for r in rows),
        exercised=bool(rows), rows=rows,
        scope='independent source/available-future actor containment; truncated final horizons explicitly recorded; no truth in control')


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
