"""Actor-only one-way service walk; never a robot planning input."""
import numpy as np

STRAIGHT_START=(29.8,23.5,0.)
STRAIGHT_GOAL=(29.8,7.,0.)
STRAIGHT_TRIGGER_X=26.


def audit_straight_walk(records):
    moving=[r for r in records if r['phase']=='move']
    points=np.array([r['actor_xyz'] for r in moving])
    times=np.array([r['time'] for r in moving])
    enough=len(moving)>=60 and np.all(np.diff(times)>0)
    checks=dict(explicit_one_way=bool(records) and all(r.get('straight_service_walk') is True for r in records),
        native_one_way_completed=bool(records) and records[-1]['phase']=='away_endpoint',
        straight_aisle=bool(enough and np.max(np.abs(points[:,0]-STRAIGHT_START[0]))<=.25),
        forward_only_actor=bool(enough and np.max(np.diff(points[:,1]))<=.03),
        complete_aisle_travel=bool(enough and points[0,1]-points[-1,1]>=16.),
        independent_geometry=bool(records) and all(r['actor_layout_clearance_m']>.02 and r['collider_alignment_error_m']<=.03 for r in records),
        grounded=bool(enough and np.max(np.abs(points[:,2]))<=.05),
        measured_walking=bool(enough and np.mean(np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1)/np.diff(times)>=.2)>=.8))
    return dict(passed=all(checks.values()),checks=checks,
        scope='One straight native actor walk, no turn or loop; truth is independent evaluation only')
