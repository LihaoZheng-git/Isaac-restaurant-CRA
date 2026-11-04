from dataclasses import replace
import copy

import numpy as np
import pytest

from robot.perception.static_prop_evidence import StaticPropEvidence, StaticPropEvidenceConfig
from robot.perception.persistent_object_evidence import PersistentObjectEvidenceConfig
from robot.perception.physical_attributes import EstimateStatus
from test_image_object_runtime import setup, observed
from test_image_object_association import proposal
from test_object_risk_map import scalar


def prepared():
    runtime = setup()
    runtime.static_prop_evidence_config = StaticPropEvidenceConfig()
    runtime.image_object_property_persistence = True
    mask = np.zeros((100,100), bool)
    mask[35:45,34:67] = True
    runtime.segmenter.image_proposals = lambda *a: (proposal(mask),)
    for i,t in enumerate((1.,1.1,1.2),1):
        result = runtime(observed(i,t))
    original = next(o for o in result.objects if o.diagnostics.get('image_object_association'))
    runtime.persistent_properties.freeze_properties()
    return runtime, original


def retained(result):
    return next(o for o in result.objects if o.diagnostics.get('static_prop_policy'))


def test_visual_gap_retains_original_mass_contact_and_live_kinetic():
    runtime, original = prepared()
    runtime.capture_risk_masks = True
    runtime.segmenter.image_proposals = lambda *a: ()
    source = replace(observed(4,1.3), measured_robot_velocity_world_xy=(.2,0.))
    result = runtime(source)
    obj = retained(result)
    assert obj.attributes.track_id == original.attributes.track_id
    assert obj.attributes.mass.mass_kg.value == original.attributes.mass.mass_kg.value
    assert obj.attributes.mass.mass_kg.timestamp_seconds == original.attributes.mass.mass_kg.timestamp_seconds
    assert obj.attributes.mass.mass_kg.status is EstimateStatus.PRIOR
    assert obj.attributes.mass.mass_kg.confidence < original.attributes.mass.mass_kg.confidence
    assert obj.diagnostics['contact_proxy']['value'] == original.diagnostics['contact_proxy']['value']
    assert obj.attributes.motion.relative_speed_mps.timestamp_seconds == 1.3
    assert obj.diagnostics['factor_scores']['kinetic'] != original.diagnostics['factor_scores']['kinetic']
    assert obj.diagnostics['policy_return_count'] == 16
    assert not obj.diagnostics['accepted_instance']  # no invented visual measurement
    assert result.field.all_obstacle_point_count == 16
    assert result.diagnostics['fixed_lambda'] == 1.25
    view = result.risk_camera_frame
    assert any(m.retained_lidar_points and 'retained' in m.basis for m in view.masks)


def test_new_current_samples_inside_static_region_keep_original_source():
    runtime, original = prepared()
    runtime.segmenter.image_proposals = lambda *a: ()
    source = observed(4,1.3)
    points = np.vstack((source.obstacle_xyz, [[2.,.1,.2],[2.,.11,.2],[2.01,.1,.2],[2.01,.11,.2]]))
    result = runtime(replace(source, obstacle_xyz=points, scene_xyz=points))
    obj = retained(result)
    assert obj.attributes.mass.mass_kg.value == original.attributes.mass.mass_kg.value
    assert obj.diagnostics['static_prop_association']['current_support_fraction'] < 1
    assert result.field.all_obstacle_point_count == len(points)


def test_static_memory_is_transactional_on_field_failure(monkeypatch):
    import robot.evaluation.multi_object_perception as mapping
    runtime, _ = prepared()
    state = runtime.image_object_state
    def fail(*a,**kw):
        raise RuntimeError('controlled field failure')
    monkeypatch.setattr(mapping, 'build_safe_sage_field_pipeline', fail)
    with pytest.raises(RuntimeError, match='controlled'):
        runtime(observed(4,1.3))
    assert runtime.image_object_state is state


def memory():
    m = StaticPropEvidence(StaticPropEvidenceConfig(), PersistentObjectEvidenceConfig(scope='unit static props'))
    points = observed(1,1.).obstacle_xyz
    assert m.enroll(5, points, scalar(2,'kg',1.), scalar(.3,'1',1.),
        frame_id=1, source_time=1., robot_xy=(0.,0.), frozen=False)
    return m, points


def test_missing_support_keeps_private_evidence_but_publishes_nothing():
    m, points = memory()
    original = copy.deepcopy(m.records[5])
    matched,_ = m.match(points+[2,0,0], np.ones(len(points),int), 2.)
    assert not matched and 5 in m.records
    matched,_ = m.match(points, np.ones(len(points),int), 2.1)
    assert 5 in matched
    np.testing.assert_array_equal(m.records[5]['points'],original['points'])
    assert m.retained(5,2.1)[0]['mass'].timestamp_seconds == 1.


def test_expired_properties_cannot_be_renewed_by_lidar():
    m, points = memory()
    matched,_ = m.match(points, np.ones(len(points),int), 122.)
    assert not matched and not m.records


def test_static_props_keep_their_clock_in_a_mixed_human_scene():
    human = PersistentObjectEvidenceConfig(scope='human only',
        maximum_property_age_seconds=3., confidence_half_life_seconds=1.5,
        retained_properties=('mass',))
    m = StaticPropEvidence(StaticPropEvidenceConfig(), human)
    points = observed(1,1.).obstacle_xyz
    mass, contact = scalar(2,'kg',1.), scalar(.3,'1',1.)
    assert m.enroll(5,points,mass,contact,frame_id=1,source_time=1.,robot_xy=(0,0),frozen=True)
    assert m.match(points,np.ones(len(points),int),11.)[0]
    values, details = m.retained(5,11.)
    assert values['mass'].value == 2 and values['contact'].value == .3
    assert values['mass'].confidence == pytest.approx(mass.confidence*2**(-10/60))
    assert details['mass']['source_timestamp'] == 1.
    assert human.maximum_property_age_seconds == 3.
    assert human.confidence_half_life_seconds == 1.5
    assert human.retained_properties == ('mass',)
    assert not m.match(points,np.ones(len(points),int),122.)[0]


def test_static_property_clock_is_explicitly_configurable():
    cfg = StaticPropEvidenceConfig(maximum_property_age_seconds=10.,confidence_half_life_seconds=5.)
    m = StaticPropEvidence(cfg,PersistentObjectEvidenceConfig(scope='unit'))
    assert m.property_config.maximum_property_age_seconds == 10.
    assert m.property_config.confidence_half_life_seconds == 5.


def test_competing_locations_and_excluded_people_withhold_policy():
    m, points = memory()
    m.records[6] = copy.deepcopy(m.records[5])
    assert not m.match(points,np.ones(len(points),int),2.)[0]
    del m.records[6]
    assert not m.match(points,np.ones(len(points),int),2.,excluded=np.ones(len(points),bool))[0]


def test_large_foreign_fragment_is_not_absorbed():
    m, points = memory()
    points = np.vstack((points,points+[1,0,0]))
    assert not m.match(points,np.ones(len(points),int),2.)[0]


def test_retained_and_oracle_properties_cannot_seed():
    m, points = memory()
    for status in (EstimateStatus.ORACLE, EstimateStatus.PRIOR):
        assert not m.enroll(9,points+[1,0,0],replace(scalar(2,'kg',1.),status=status),
            scalar(.3,'1',1.),frame_id=1,source_time=1.,robot_xy=(0,0),frozen=True)


@pytest.mark.parametrize('kw',[{'position_allowance_m':0},{'minimum_distinct_support':3},
                             {'minimum_current_support_fraction':1.1}])
def test_invalid_config(kw):
    with pytest.raises(ValueError):
        StaticPropEvidenceConfig(**kw)


def test_static_prop_cli_preserves_bounded_motion_scope(monkeypatch):
    import json
    from pathlib import Path
    path = Path('scripts/demos/cra_live_object_isaac_demo.py').resolve()
    command = json.loads(Path('configs/restaurant_static_prop_bin.json').read_text())['commands']['static']
    command += ['--output','outputs/unused-static-cli']
    monkeypatch.setattr('sys.argv',command)
    code = compile(path.read_text().split('args.output.mkdir',1)[0],str(path),'exec')
    namespace = {'__file__':str(path)}
    exec(code,namespace)
    assert namespace['args'].static_prop_evidence
    command.remove('--image-object-approach')
    with pytest.raises(SystemExit):
        exec(code,{'__file__':str(path)})
