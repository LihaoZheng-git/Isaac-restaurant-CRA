"""Resolution controls and original-clock retention, independent of prop names."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from robot.perception.contact_boundary_resolution import resolved_contact_boundary
from robot.perception.local_contact_geometry import LocalContactGeometryConfig
from robot.perception.visual_shape_contact import SilhouetteContactConfig, estimate_silhouette_contact_proxy
from test_local_contact_geometry import detection, mask_fixture


CONFIG = SilhouetteContactConfig(local_geometry=LocalContactGeometryConfig(), pixel_denoise=True)


def cue(mask, config=CONFIG):
    return estimate_silhouette_contact_proxy(detection(mask), timestamp_seconds=4., config=config)


@pytest.mark.parametrize('rotation', [0, 27, 90, 143])
@pytest.mark.parametrize('scale', [.5, 1, 2])
def test_resolved_tips_survive_and_smooth_negatives_remain_low(rotation, scale):
    scores = {}
    for name in ('circle', 'ellipse', 'capsule', 'triangle', 'nail', 'star'):
        mask = cv2.warpAffine(mask_fixture(name).astype(np.uint8),
            cv2.getRotationMatrix2D((128, 128), rotation, 1), (256, 256), flags=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST).astype(bool)
        scores[name] = cue(mask).value
    assert max(scores[n] for n in ('circle', 'ellipse', 'capsule')) < .2
    assert min(scores[n] for n in ('triangle', 'nail', 'star')) > .5


def test_thin_spur_is_resolution_limited_without_changing_source_mask_or_clocks():
    mask = np.zeros((100, 100), np.uint8)
    cv2.circle(mask, (50, 50), 20, 1, -1)
    mask = mask.astype(bool)
    mask[50, 70:75] = True
    original = mask.copy()
    before = cue(mask, replace(CONFIG, pixel_denoise=False))
    after = cue(mask)
    assert before.value > .5 and after.value < .1
    assert after.timestamp_seconds == before.timestamp_seconds == 4.
    assert after.confidence == before.confidence and after.status == before.status
    np.testing.assert_array_equal(mask, original)
    assert any('one-pixel-wide real tips may be unresolved' in e for e in after.evidence)
    # A name is never an input to the correction or the scalar value.
    assert after == estimate_silhouette_contact_proxy(detection(mask, 'knife'),
        timestamp_seconds=4., config=CONFIG)


@pytest.mark.parametrize('case', ['excess_removal', 'split', 'hole'])
def test_rejects_whole_correction_without_selecting_a_convenient_component(case):
    mask = np.zeros((100, 100), np.uint8)
    mask[20:80, 20:80] = 1
    if case == 'excess_removal':
        mask[10:20, 20:80:2] = 1
    elif case == 'split':
        mask[20:80, 47:53] = 0
        mask[50, 47:53] = 1
    else:
        mask[40:60, 40:60] = 0
    original = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    result, evidence = resolved_contact_boundary(mask, original)
    assert result is original
    assert 'contact_boundary_resolution_rejected; original contour retained' in evidence


def test_specialist_cue_is_still_maximum_and_bad_configs_reject():
    from robot.perception.physical_attributes import ScalarEstimate, EstimateStatus
    from robot.perception.visual_shape_contact import combine_shape_and_part_contact
    part = ScalarEstimate(.9, .8, 1., .75, '1', 'specialist', 3., EstimateStatus.INFERRED, ())
    combined = combine_shape_and_part_contact(cue(mask_fixture('circle')), part)
    assert combined.value == .9 and combined.timestamp_seconds == 3.
    with pytest.raises(ValueError):
        SilhouetteContactConfig(pixel_denoise=True)
    with pytest.raises(TypeError):
        replace(CONFIG, pixel_denoise=1)


def test_retention_keeps_corrected_intrinsic_cue_but_motion_stays_live():
    from robot.perception.static_prop_evidence import StaticPropEvidenceConfig
    from test_image_object_runtime import setup, observed
    from test_image_object_association import proposal
    from test_static_prop_evidence import retained
    results = []
    for enabled in (False, True):
        worker = setup()
        worker.shape_contact_config = replace(CONFIG, pixel_denoise=enabled)
        worker.static_prop_evidence_config = StaticPropEvidenceConfig()
        worker.image_object_property_persistence = True
        mask = np.zeros((100, 100), bool)
        mask[35:45, 34:67] = True
        worker.segmenter.image_proposals = lambda *a: (proposal(mask),)
        for index, timestamp in enumerate((1., 1.1, 1.2), 1):
            result = worker(observed(index, timestamp))
        fresh = next(o for o in result.objects if o.diagnostics.get('image_object_association'))
        worker.persistent_properties.freeze_properties()
        worker.segmenter.image_proposals = lambda *a: ()
        result = worker(replace(observed(4, 1.3), measured_robot_velocity_world_xy=(.2, 0.)))
        old = fresh.diagnostics['contact_proxy']
        current = retained(result)
        prop = current.diagnostics['contact_proxy']
        for name in ('value', 'lower_bound', 'upper_bound', 'timestamp_seconds'):
            assert prop[name] == old[name]
        assert current.attributes.motion.relative_speed_mps.timestamp_seconds == 1.3
        assert current.diagnostics['factor_scores']['kinetic'] != fresh.diagnostics['factor_scores']['kinetic']
        assert result.field.all_obstacle_point_count == 16
        assert result.diagnostics['fixed_lambda'] == 1.25
        results.append((fresh, current, result))
    assert results[0][0].attributes.mass == results[1][0].attributes.mass
    for factor in ('potential', 'kinetic', 'thermal'):
        assert results[0][1].diagnostics['factor_scores'][factor] == results[1][1].diagnostics['factor_scores'][factor]
    np.testing.assert_array_equal(results[0][2].field.tracking_result.point_track_ids,
                                  results[1][2].field.tracking_result.point_track_ids)


def test_native_opt_in_requires_stationary_or_explicit_bounded_approach(monkeypatch):
    from test_restaurant_container_bench import _parse_mount, BASE_ARGS
    assert not _parse_mount(BASE_ARGS, monkeypatch).contact_pixel_denoise
    args = [*BASE_ARGS, '--stationary-image-objects', '--local-contact-geometry', '--contact-pixel-denoise']
    assert _parse_mount(args, monkeypatch).contact_pixel_denoise
    for bad in ([*BASE_ARGS, '--contact-pixel-denoise'], [*args, '--motion']):
        with pytest.raises(SystemExit):
            _parse_mount(bad, monkeypatch)


def test_corrected_short_motion_keeps_all_original_prop_guards(monkeypatch):
    import json
    from pathlib import Path
    from test_restaurant_container_bench import _parse_mount
    recipes = json.loads(Path('configs/restaurant_contact_short_live.json').read_text())['commands']
    for command in recipes.values():
        args = command[1:] + ['--motion']
        parsed = _parse_mount(args, monkeypatch)
        assert parsed.contact_pixel_denoise and parsed.image_object_approach
        assert parsed.static_prop_evidence and parsed.stationary_frames == 60
        for missing in ('--image-object-approach', '--static-prop-evidence', '--moving-view-image-objects', '--contact-pixel-denoise'):
            bad = args.copy()
            bad.remove(missing)
            with pytest.raises(SystemExit):
                _parse_mount(bad, monkeypatch)
        for extra in (['--restaurant-human', 'standing'], ['--restaurant-server'],
                      ['--stationary-frames', '18'], ['--restaurant-route', 'long-service']):
            with pytest.raises(SystemExit):
                _parse_mount([*args, *extra], monkeypatch)


@pytest.mark.parametrize('case', ['long-parallel', 'long-parallel-near'])
def test_corrected_service_motion_keeps_mixed_scene_guards(monkeypatch, case):
    import json
    from pathlib import Path
    from test_restaurant_container_bench import _parse_mount
    recipes = json.loads(Path('configs/restaurant_service_demo_v3_corrected_contact.json').read_text())['commands']
    args = recipes[case][1:] + ['--motion']
    verified = []
    if case == 'long-parallel':
        with pytest.raises(SystemExit):
            _parse_mount(args, monkeypatch)
        from robot.evaluation import restaurant_long_service_validation
        monkeypatch.setattr(restaurant_long_service_validation, 'verify_near_human_gates',
                            lambda *a, **kw: verified.append((a, kw)))
        args += ['--long-service-human-gates', 'component', 'near', 'actor']
    parsed = _parse_mount(args, monkeypatch)
    assert parsed.contact_pixel_denoise and parsed.service_static_props
    assert parsed.restaurant_human == 'parallel' and parsed.static_prop_evidence
    if case == 'long-parallel':
        assert len(verified) == 1
        assert verified[0][0][0] == [Path('component'), Path('near'), Path('actor')]
        assert verified[0][1]['require_straight_walk']
    for missing in ('--service-static-props', '--static-prop-evidence',
                    '--moving-view-image-objects', '--image-object-property-persistence',
                    '--image-object-overlap-hold', '--local-contact-geometry',
                    '--stationary-image-objects', '--restaurant-server'):
        bad = args.copy()
        bad.remove(missing)
        with pytest.raises(SystemExit):
            _parse_mount(bad, monkeypatch)
    for extra in (['--restaurant-human', 'standing'], ['--human-camera-raise-metres', '0'],
                  ['--disable-lidar-deskew'], ['--contact-mask-refinement'], ['--image-object-approach']):
        with pytest.raises(SystemExit):
            _parse_mount([*args, *extra], monkeypatch)


def test_corrected_service_recipe_only_enables_the_validated_contact_option():
    import json
    from pathlib import Path
    old = json.loads(Path('configs/restaurant_service_demo_v3_open_aisle.json').read_text())['commands']
    new = json.loads(Path('configs/restaurant_service_demo_v3_corrected_contact.json').read_text())['commands']
    assert old.keys() == new.keys()
    changed = {'long-parallel', 'long-parallel-near', 'bin-static', 'crate-static'}
    for case, command in old.items():
        assert new[case] == command + (['--contact-pixel-denoise'] if case in changed else [])
