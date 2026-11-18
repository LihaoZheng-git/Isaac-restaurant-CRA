"""Actor-only one-way service walk; never a robot planning input."""
import math

import numpy as np

STRAIGHT_START=(29.8,23.5,0.)
STRAIGHT_GOAL=(29.8,7.,0.)
STRAIGHT_TRIGGER_X=26.
STRAIGHT_GUIDES=tuple((29.8,y,0.) for y in (21.5,19.5,17.5,15.5,13.5,11.5,9.5,7.))
LEFT_OUTBOUND_ACTOR_START = (3.15, 14.5, 0.)
LEFT_OUTBOUND_NEAR_ACTOR_START = (3.15, 11.5, 0.)
LEFT_OUTBOUND_ACTOR_GOAL = (3.15, 9., 0.)
LEFT_OUTBOUND_NEAR_ROBOT_START = (2.35, 8., .07382)
LEFT_OUTBOUND_NEAR_ROBOT_YAW = math.pi / 2
LEFT_OUTBOUND_TRIGGER_Y = 8.2
LEFT_OUTBOUND_GUIDES = ((3.15, 12.5, 0.), (3.15, 10.5, 0.), LEFT_OUTBOUND_ACTOR_GOAL)
LEFT_OUTBOUND_LOUNGE_ACTOR_GOAL = (3.15, 8., 0.)
LEFT_OUTBOUND_LOUNGE_TRIGGER_Y = 9.15
LEFT_OUTBOUND_WEST_DINER_START = (0.95, 14.5, 0.)
LEFT_OUTBOUND_WEST_DINER_NEAR_START = (0.95, 11.5, 0.)
LEFT_OUTBOUND_WEST_DINER_GOAL = (0.95, 8., 0.)
LEFT_OUTBOUND_WEST_DINER_GUIDES = (
    (0.95, 12.5, 0.), (0.95, 10.5, 0.), LEFT_OUTBOUND_WEST_DINER_GOAL,
)


def left_outbound_actor_start(*, near_view=False, west_diner=False):
    if west_diner:
        return LEFT_OUTBOUND_WEST_DINER_NEAR_START if near_view else LEFT_OUTBOUND_WEST_DINER_START
    return LEFT_OUTBOUND_NEAR_ACTOR_START if near_view else LEFT_OUTBOUND_ACTOR_START


class StraightServiceWalk:
    """Collinear actor goals constrain accumulated native gait heading drift."""
    def __init__(self, *, left_outbound_service=False, left_outbound_lounge_walk=False,
                 left_outbound_west_diner=False):
        if left_outbound_lounge_walk and not left_outbound_service:
            raise ValueError('the lounge continuation requires the left outbound actor')
        if left_outbound_west_diner and not left_outbound_lounge_walk:
            raise ValueError('the west diner requires the left outbound lounge continuation')
        self.index=0
        self.left_outbound_service = left_outbound_service
        self.left_outbound_lounge_walk = left_outbound_lounge_walk
        self.left_outbound_west_diner = left_outbound_west_diner
        self.guides = LEFT_OUTBOUND_GUIDES if left_outbound_service else STRAIGHT_GUIDES
        self.trigger_y = LEFT_OUTBOUND_TRIGGER_Y
        if left_outbound_lounge_walk:
            self.guides = (*LEFT_OUTBOUND_GUIDES[:-1], LEFT_OUTBOUND_LOUNGE_ACTOR_GOAL)
            self.trigger_y = LEFT_OUTBOUND_LOUNGE_TRIGGER_Y
        if left_outbound_west_diner:
            self.guides = LEFT_OUTBOUND_WEST_DINER_GUIDES

    @property
    def goal(self):
        return self.guides[self.index]

    @property
    def final(self):
        return self.index==len(self.guides)-1

    def ready_to_walk(self, robot_xy):
        if self.left_outbound_service:
            return robot_xy[1] >= self.trigger_y
        return robot_xy[0] >= STRAIGHT_TRIGGER_X

    def advance(self,xyz):
        if not self.final and np.linalg.norm(np.asarray(xyz)[:2]-self.goal[:2])<=.20:
            self.index+=1
            return True
        return False


def audit_straight_walk(records, *, left_outbound_service=False, left_outbound_lounge_walk=False,
                        left_outbound_west_diner=False):
    walk = StraightServiceWalk(left_outbound_service=left_outbound_service,
                               left_outbound_lounge_walk=left_outbound_lounge_walk,
                               left_outbound_west_diner=left_outbound_west_diner)
    start = (left_outbound_actor_start(west_diner=left_outbound_west_diner)
             if left_outbound_service else STRAIGHT_START)
    goal = walk.guides[-1]
    moving=[r for r in records if r['phase']=='move']
    points=np.array([r['actor_xyz'] for r in moving])
    times=np.array([r['time'] for r in moving])
    enough=len(moving)>=60 and np.all(np.diff(times)>0)
    lounge_release = not left_outbound_lounge_walk or bool(
        moving and walk.ready_to_walk(moving[0]['robot_xy'])
    )
    lounge_endpoint = not left_outbound_lounge_walk or bool(
        records and abs(records[-1]['actor_xyz'][1] - goal[1]) <= .3
    )
    checks=dict(explicit_one_way=bool(records) and all(r.get('straight_service_walk') is True
                    and r.get('left_outbound_service', False) is left_outbound_service
                    and r.get('left_outbound_lounge_walk', False) is left_outbound_lounge_walk
                    and r.get('left_outbound_west_diner', False) is left_outbound_west_diner
                    for r in records) and lounge_release,
        native_one_way_completed=bool(records) and records[-1]['phase']=='away_endpoint' and lounge_endpoint,
        straight_aisle=bool(enough and np.max(np.abs(points[:,0]-start[0]))<=.25),
        forward_only_actor=bool(enough and np.max(np.diff(points[:,1]))<=.03),
        complete_aisle_travel=bool(enough and points[0,1]-points[-1,1]>=start[1]-goal[1]-.5),
        independent_geometry=bool(records) and all(r['actor_layout_clearance_m']>.02 and r['collider_alignment_error_m']<=.03 for r in records),
        grounded=bool(enough and np.max(np.abs(points[:,2]))<=.05),
        measured_walking=bool(enough and np.mean(np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1)/np.diff(times)>=.2)>=.8))
    return dict(passed=all(checks.values()),checks=checks,
        scope='One straight native actor walk, no turn or loop; truth is independent evaluation only')
