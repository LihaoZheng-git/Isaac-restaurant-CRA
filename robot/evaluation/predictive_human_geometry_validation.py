"""Independent future-envelope scoring; actor truth never enters prediction/control."""

from bisect import bisect_left
import math

import numpy as np

from robot.safety.predictive_human_geometry import (
    HumanEnvelopePrediction, HumanGeometryForecastConfig, PredictiveHumanGeometry,
)


def validate_predictive_human_geometry(records, actor_records, *, reference_radius_m):
    if not math.isfinite(reference_radius_m) or reference_radius_m <= 0:
        raise ValueError('positive independent actor radius required')
    epochs = [float(r['time']) for r in actor_records]
    if len(epochs) < 2 or not np.isfinite(epochs).all() or np.any(np.diff(epochs) <= 0):
        raise ValueError('ordered independent actor epochs required')
    rows, rejected, skipped = [], [], []
    accepted_sources = set()
    for record in records:
        stamp = record.get('sensor_timestamp', record.get('time'))
        frame = record.get('sensor_frame_id')
        diagnostic = record.get('diagnostics', record).get('predictive_human_geometry')
        if diagnostic is None:
            rejected.append(dict(source_timestamp=stamp, reason='missing_prediction_diagnostic'))
            continue
        if not diagnostic['ready'] or not diagnostic['forecasts']:
            skipped.append(dict(source_timestamp=stamp, reason='no_ready_current_human_forecast',
                fits=diagnostic.get('fits')))
            continue
        try:
            config = HumanGeometryForecastConfig(**diagnostic['configuration'])
            forecasts = tuple(HumanEnvelopePrediction(**{key: value for key, value in f.items()
                if key in HumanEnvelopePrediction.__dataclass_fields__ and key != 'config'},
                config=HumanGeometryForecastConfig(**f['config'])) for f in diagnostic['forecasts'])
            geometry = PredictiveHumanGeometry(frame, stamp,
                tuple(diagnostic['anchor_position_world_xy']), diagnostic['anchor_yaw'],
                tuple(diagnostic['required_track_ids']), forecasts, config)
            geometry.validate_source(diagnostic['sensor_frame_id'], diagnostic['sensor_timestamp_seconds'],
                geometry.anchor_position_world_xy, geometry.anchor_yaw)
            if config.velocity_uncertainty_floor_prior_mps is not None and any(
                not math.isclose(f['effective_uncertainty_growth_prior_mps'], forecast.uncertainty_growth_prior_mps,
                    rel_tol=0, abs_tol=1e-12)
                or f['uncertainty_growth_basis'] != 'bounded_prior_from_current_raw_minus_window_fit_velocity'
                for f, forecast in zip(diagnostic['forecasts'], forecasts)):
                raise ValueError('adaptive growth must preserve its current source-derived diagnostic')
            if diagnostic['status'] != 'prior':
                raise ValueError('forecast must remain an explicit prior')
        except (KeyError, TypeError, ValueError) as error:
            rejected.append(dict(source_timestamp=stamp, reason=str(error)))
            continue
        horizon = config.horizon_seconds+config.maximum_source_age_seconds
        offsets = sorted(set([0., horizon]+np.arange(.2, horizon, .2).tolist()))
        for forecast in forecasts:
            center = np.asarray(forecast.centroids_world_xy[-1])
            for offset in offsets:
                target = stamp+offset
                if target > epochs[-1]+1e-9:
                    skipped.append(dict(source_timestamp=stamp, offset_seconds=offset, reason='future_beyond_recorded_actor_window'))
                    continue
                target = min(target, epochs[-1])  # numerical roundoff only, <=1ns
                index = bisect_left(epochs, target)
                if index == 0:
                    index = 1
                if index >= len(epochs) or epochs[index]-epochs[index-1] > .05+1e-9 or target < epochs[0]:
                    rejected.append(dict(source_timestamp=stamp, offset_seconds=offset, reason='unaligned_independent_future_truth'))
                    continue
                a, b = actor_records[index-1], actor_records[index]
                fraction = (target-epochs[index-1])/(epochs[index]-epochs[index-1])
                truth = (1-fraction)*np.asarray(a['actor_xyz'][:2])+fraction*np.asarray(b['actor_xyz'][:2])
                predicted = center+offset*forecast.velocity_world_xy
                error = float(np.linalg.norm(predicted-truth))
                current_error = float(np.linalg.norm(center-truth))
                current_margin = forecast.radius_prior_m-reference_radius_m-current_error
                predicted_margin = (forecast.radius_prior_m+offset*forecast.uncertainty_growth_prior_mps
                    -reference_radius_m-error)
                # Sufficient disk containment by either member of the union.
                # The unchanged original current field remains enforced too.
                margin = max(current_margin, predicted_margin)
                walking = float(np.linalg.norm(np.asarray(b['actor_xyz'][:2])-a['actor_xyz'][:2])
                    /(epochs[index]-epochs[index-1])) >= .3
                rows.append(dict(source_timestamp=stamp, source_frame_id=frame, track_id=forecast.track_id,
                    uncertainty_growth_prior_mps=forecast.uncertainty_growth_prior_mps,
                    offset_seconds=offset, predicted_center_error_m=error, current_center_error_m=current_error,
                    current_containment_margin_m=current_margin, predictive_containment_margin_m=predicted_margin,
                    union_containment_margin_m=margin, walking=walking, passed=margin >= -1e-9))
                accepted_sources.add((frame, stamp))
    checks = dict(at_least_twenty_forecast_sources=len(accepted_sources) >= 20,
        at_least_twenty_walking_future_queries=sum(r['walking'] and r['offset_seconds'] > 0 for r in rows) >= 20,
        original_typed_source_provenance=not rejected,
        all_tested_future_envelopes_contain_reference=bool(rows) and all(r['passed'] for r in rows))
    return dict(passed=all(checks.values()), checks=checks, forecast_source_count=len(accepted_sources),
        query_count=len(rows), failed_query_count=sum(not r['passed'] for r in rows),
        minimum_union_containment_margin_m=min((r['union_containment_margin_m'] for r in rows), default=None),
        rows=rows, rejected=rejected, skipped=skipped,
        scope='independent bounded future actor-envelope audit; truth scoring only; sufficient containment by current-or-predicted disk union; original static field remains required; not a general motion guarantee')
