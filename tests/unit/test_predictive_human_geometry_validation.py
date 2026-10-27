from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from test_predictive_human_geometry import prediction

from robot.evaluation.predictive_human_geometry_validation import validate_predictive_human_geometry
from robot.safety.predictive_human_geometry import PredictiveHumanGeometry


def fixture():
    records = []
    for i, t in enumerate(np.arange(1., 5.1, .1)):
        f = prediction(center=(2.-.5*t, 1.), velocity=(-.5, 0.))
        times = np.array(f.source_timestamps_seconds)+t-1.
        times[-1] = t
        f = replace(f, source_timestamps_seconds=tuple(times))
        g = PredictiveHumanGeometry(60+i, float(t), (0., 0.), 0., (f.track_id,), (f,), f.config)
        records.append(dict(time=float(t), sensor_frame_id=60+i, predictive_human_geometry=g.diagnostic()))
    truth = [dict(time=float(t), actor_xyz=[2.-.5*t, 1., 0.]) for t in np.arange(0, 7.01, 1/60)]
    return records, truth


def test_independent_future_audit_preserves_source_epochs_and_scores_all_available_horizons():
    records, truth = fixture()
    before = deepcopy(records)
    result = validate_predictive_human_geometry(records, truth, reference_radius_m=.4)
    assert result['passed'] and result['forecast_source_count'] == 41
    assert result['failed_query_count'] == 0
    assert max(r['offset_seconds'] for r in result['rows']) == pytest.approx(1.55)
    assert records == before


def test_wrong_forecast_and_missing_frame_provenance_cannot_pass():
    records, truth = fixture()
    truth = [dict(r, actor_xyz=[r['actor_xyz'][0], 3., 0.]) for r in truth]
    result = validate_predictive_human_geometry(records, truth, reference_radius_m=.4)
    assert not result['passed'] and result['failed_query_count'] > 0
    records, truth = fixture()
    del records[0]['sensor_frame_id']
    result = validate_predictive_human_geometry(records, truth, reference_radius_m=.4)
    assert not result['passed'] and result['rejected']


def test_warmup_and_end_of_record_are_reported_without_future_extrapolation():
    records, truth = fixture()
    records[0]['predictive_human_geometry']['ready'] = False
    truth = [r for r in truth if r['time'] <= 5.5]
    result = validate_predictive_human_geometry(records, truth, reference_radius_m=.4)
    assert result['passed']
    reasons = {r['reason'] for r in result['skipped']}
    assert reasons == {'no_ready_current_human_forecast', 'future_beyond_recorded_actor_window'}


@pytest.mark.parametrize('script,flags', [
    ('cra_live_object_isaac_demo.py', ['--predictive-human-geometry-ablation']),
    ('restaurant_human_locomotion_isaac.py', ['--predictive-human-geometry']),
    ('cra_live_object_isaac_demo.py', ['--human-prediction-uncertainty-rate', '.30']),
    ('restaurant_human_locomotion_isaac.py', ['--human-prediction-uncertainty-rate', '.30'])])
def test_cli_rejects_unlabeled_prediction_before_native_imports(script, flags, tmp_path):
    source = Path(__file__).resolve().parents[2]/'scripts/demos'/script
    command = [sys.executable, str(source), '--output', str(tmp_path/'unused'), *flags]
    if script == 'restaurant_human_locomotion_isaac.py':
        command += ['--scene', 'unused.usda', '--layout-survey', 'unused.json']
    process = subprocess.run(command, text=True, capture_output=True)
    assert process.returncode == 2 and 'requires' in process.stderr
    assert not (tmp_path/'unused').exists()
