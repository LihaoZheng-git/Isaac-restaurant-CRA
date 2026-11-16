import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot.scene.restaurant_straight_walk import audit_straight_walk, STRAIGHT_START, STRAIGHT_GOAL
from robot.scene.restaurant_human_motion import human_navmesh_extent
from test_static_prop_evidence import prepared
from test_image_object_runtime import observed


def test_current_person_excludes_retained_static_policy_without_erasing_memory():
    runtime, original=prepared()
    runtime.service_static_props=True
    runtime.segmenter.image_proposals=lambda *a: ()
    from robot.perception.sam_instance_segmentation import SamInstanceResult
    attempts=[]
    encodes=[]
    runtime.segmenter.set_image=lambda rgb: encodes.append(rgb)
    def propose(*a,**kw):
        attempts.append(kw['track_id'])
        return SamInstanceResult(None,('controlled unaccepted generic mask',))
    runtime.segmenter.propose=propose
    source=observed(4,1.3)
    runtime.visual_person_frame=lambda p: SimpleNamespace(source_frame_id=p.camera_frame_id,
        source_timestamp_seconds=p.camera_timestamp_seconds,
        detections=[SimpleNamespace(semantic_class='person',confidence=.8,mask=np.ones((100,100),bool))])
    result=runtime(source)
    assert not any(o.diagnostics.get('static_prop_policy') for o in result.objects)
    assert original.attributes.track_id in runtime.image_object_state.static_properties.records
    assert result.field.all_obstacle_point_count==len(source.obstacle_xyz)
    assert attempts  # mixed mode must keep the original generic visual path
    assert len(encodes)==1  # same-frame encoding reused by image-object proposals
    runtime.visual_person_frame=lambda p: SimpleNamespace(source_frame_id=p.camera_frame_id,
        source_timestamp_seconds=p.camera_timestamp_seconds,detections=[])
    result=runtime(observed(5,1.4))
    assert any(o.diagnostics.get('static_prop_policy') for o in result.objects)


def test_one_way_actor_audit_rejects_return_and_side_excursion():
    points=np.linspace(STRAIGHT_START,STRAIGHT_GOAL,331)
    rows=[dict(time=i/10,actor_xyz=p.tolist(),phase='move' if i<330 else 'away_endpoint',
        straight_service_walk=True,actor_layout_clearance_m=.25,collider_alignment_error_m=0.)
        for i,p in enumerate(points)]
    assert audit_straight_walk(rows)['passed']
    rows[100]['actor_xyz'][0]+=.4
    assert not audit_straight_walk(rows)['checks']['straight_aisle']
    rows[100]['actor_xyz'][0]-=.4
    rows[100]['actor_xyz'][1]+=1.
    assert not audit_straight_walk(rows)['checks']['forward_only_actor']
    assert human_navmesh_extent(parallel_oncoming=True,straight_service_walk=True)[1][1]==25.
    with pytest.raises(ValueError):
        human_navmesh_extent(straight_service_walk=True)


def test_collinear_actor_guides_never_reverse_or_loop():
    from robot.scene.restaurant_straight_walk import StraightServiceWalk, STRAIGHT_GUIDES
    walk=StraightServiceWalk()
    for i,goal in enumerate(STRAIGHT_GUIDES):
        assert walk.goal==goal and goal[0]==STRAIGHT_START[0]
        assert goal[1]<(STRAIGHT_START if i==0 else STRAIGHT_GUIDES[i-1])[1]
        assert not walk.advance((goal[0],goal[1]+.5,0))
        assert walk.advance(goal)==(i<len(STRAIGHT_GUIDES)-1)
    assert walk.final and walk.goal==STRAIGHT_GOAL


@pytest.mark.parametrize('case',['long-parallel','long-parallel-near'])
def test_v3_cli_preserves_mixed_scene_constraints(monkeypatch,case):
    path=Path('scripts/demos/cra_live_object_isaac_demo.py').resolve()
    cmd=json.loads(Path('configs/restaurant_service_demo_v3.json').read_text())['commands'][case]
    cmd+=['--output','outputs/unused-v3-cli']
    monkeypatch.setattr('sys.argv',cmd)
    code=compile(path.read_text().split('args.output.mkdir',1)[0],str(path),'exec')
    namespace={'__file__':str(path)}
    exec(code,namespace)
    assert namespace['args'].service_static_props
    cmd.remove('--static-prop-evidence')
    with pytest.raises(SystemExit):
        exec(code,{'__file__':str(path)})
