"""Actor-only one-way service walk; never a robot planning input."""
import numpy as np

STRAIGHT_START=(29.8,23.5,0.)
STRAIGHT_GOAL=(29.8,7.,0.)
STRAIGHT_TRIGGER_X=26.
STRAIGHT_GUIDES=tuple((29.8,y,0.) for y in (21.5,19.5,17.5,15.5,13.5,11.5,9.5,7.))


class StraightServiceWalk:
    """Collinear actor goals constrain accumulated native gait heading drift."""
    def __init__(self):
        self.index=0

    @property
    def goal(self):
        return STRAIGHT_GUIDES[self.index]

    @property
    def final(self):
        return self.index==len(STRAIGHT_GUIDES)-1

    def advance(self,xyz):
        if not self.final and np.linalg.norm(np.asarray(xyz)[:2]-self.goal[:2])<=.20:
            self.index+=1
            return True
        return False


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
