"""Candidate visuals preserve original geometry, pixels, identity and epochs."""

from dataclasses import replace
from copy import deepcopy
import json

import numpy as np
import pytest
from PIL import Image

from robot.evaluation.calibration_replay import CalibrationReplay, candidate_geometry, digest
from robot.evaluation.risk_palette import PALETTES, risk_color_rgb, palette_description
from robot.evaluation.risk_camera_overlay import overlay_rgb
from robot.evaluation.lidar_risk_map import map_pixels, render_lidar_risk_map
from test_lidar_risk_map import geometry, frame
from test_risk_camera_overlay import capture, fixture


def test_legacy_palette_exact_and_new_palette_fixed_without_input_changes():
    stops, colors = PALETTES["legacy"]
    for value in np.linspace(0, 1, 501):
        expected = tuple(int(round(np.interp(value, stops, np.asarray(colors)[:, i]))) for i in range(3))
        assert risk_color_rgb(value) == expected
    for value, color in zip(*PALETTES["restaurant_v1"]):
        assert risk_color_rgb(value, palette="restaurant_v1") == color
    camera = capture(*fixture(phi=.35))
    original = camera.rgb.copy()
    other = replace(camera.masks[0], track_id=9, phi=.9, mask=np.zeros_like(camera.masks[0].mask))
    # Adding a different-risk object elsewhere never renormalizes the first.
    other.mask[85:90, 85:90] = True
    a = overlay_rgb(camera, palette="restaurant_v1")
    b = overlay_rgb(replace(camera, masks=(*camera.masks, other)), palette="restaurant_v1")
    np.testing.assert_array_equal(a[50, 50], b[50, 50])
    np.testing.assert_array_equal(camera.rgb, original)
    assert palette_description("restaurant_v1")["score_knots"][0] == 0
    for value in (float("nan"), -1, 2):
        with pytest.raises(ValueError):
            risk_color_rgb(value)
    with pytest.raises(ValueError):
        risk_color_rgb(.5, palette="automatic")


def candidate(data):
    return dict(policies=[dict(track_id=int(key), original_applied_phi=float(data['point_phi'][i]),
                              candidate_current_phi=.35 + i*.1)
                          for i, key in enumerate(data['point_track_ids'])])


def test_current_scores_change_but_prior_geometry_and_sources_remain_exact():
    data = geometry()
    before = deepcopy(data)
    result = candidate_geometry(data, candidate(data))
    assert not np.array_equal(result['point_phi'], data['point_phi'])
    for key in data:
        np.testing.assert_array_equal(data[key], before[key])
        if key != 'point_phi':
            assert result[key] is data[key]
    image, audit = render_lidar_risk_map(result, frame(), [], [], palette="restaurant_v1",
        score_title="CANDIDATE phi | REPLAY", replay_scores=True)
    x, y = np.round(map_pixels([[1, 1]], [0, 0], (10, 10, 524, 380))[0][0]).astype(int)
    assert image.getpixel((x, y)) == risk_color_rgb(.45, palette="restaurant_v1")
    assert audit['map_score'] == 'candidate_current_phi_original_prior_geometry'


@pytest.mark.parametrize('fault', ['missing_id', 'duplicate', 'original_score', 'new_score'])
def test_replay_rejects_unbound_or_corrupt_policies(fault):
    data = geometry()
    row = candidate(data)
    if fault == 'missing_id':
        row['policies'].pop()
    elif fault == 'duplicate':
        row['policies'].append(row['policies'][0])
    elif fault == 'original_score':
        row['policies'][0]['original_applied_phi'] = .7
    else:
        row['policies'][0]['candidate_current_phi'] = float('nan')
    with pytest.raises(ValueError):
        candidate_geometry(data, row)


def test_manifest_sources_and_exact_field_epoch_are_checked(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    evidence = source/'evidence.json'
    evidence.write_text('{}')
    fields = tmp_path/'fields.jsonl'
    fields.write_text(json.dumps(dict(sensor_frame_id=4, sensor_timestamp=2.5,
                                     commit_simulation_time=3.))+'\n')
    manifest = tmp_path/'summary.json'
    manifest.write_text(json.dumps(dict(source_directory=str(source), fields_file=fields.name,
        fields_sha256=digest(fields), source_hashes={str(evidence):digest(evidence)})))
    replay = CalibrationReplay(manifest, source)
    epoch = dict(field_id=4, source_time=2.5, commit_time=3.)
    assert replay.field(epoch)['sensor_frame_id'] == 4
    with pytest.raises(ValueError, match='source/commit'):
        replay.field(dict(epoch, commit_time=3.1))
    evidence.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='source changed'):
        replay.verify()


def test_no_mask_rgb_requires_original_camera_and_lidar_pair(tmp_path):
    replay = CalibrationReplay.__new__(CalibrationReplay)
    replay.source, replay.hashes = tmp_path, {}
    folder = tmp_path/'sensor-snapshots'
    folder.mkdir()
    Image.fromarray(np.full((3, 4, 3), 80, np.uint8)).save(folder/'000004.png')
    metadata = folder/'000004.json'
    metadata.write_text(json.dumps(dict(lidar_id=4, lidar_time=1., camera_id=9, camera_time=1.1)))
    epoch = dict(status='latest_assessed_frame', sensor_frame_id=4, sensor_timestamp_seconds=1.,
                 camera_frame_id=9, camera_timestamp_seconds=1.1, masks=[], rejected=[])
    row = dict(sensor_frame_id=4, objects=[], policies=[])
    camera = replay.camera(epoch, row)
    assert not camera.masks and camera.rgb.shape == (3, 4, 3)
    assert camera.camera_timestamp_seconds == 1.1
    with pytest.raises(ValueError, match='different source epoch'):
        replay.camera(dict(epoch, camera_frame_id=10), row)
