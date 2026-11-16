"""V2 actor-only service circuit. No route or actor state enters robot control."""

import math
import numpy as np

BACKGROUND_ROUTE = ((29.8, 12.8, 0.), (28.5, 12.8, 0.),
                    (28.5, 11.2, 0.), (29.8, 11.2, 0.))
BACKGROUND_ENCOUNTER_TRIGGER_Y = 7.0
CONTINUATION_ROUTE = ((27.8, 7., 0.), (27.8, 11.2, 0.),
                      (28.5, 12.8, 0.), (29.8, 12.8, 0.))
# Separate demo authoring: keep the first encounter, then leave its corridor.
# This route is never supplied to perception, nominal control, MPC or CBF.
UPPER_DEPARTURE_ROUTE = ((27.8, 7., 0.), (27.8, 11.2, 0.), (27.8, 14.4, 0.),
                         (29.7, 15.1, 0.), (29.7, 23.5, 0.), (29.7, 18.5, 0.))


class BackgroundServiceWalk:
    """Walk continuously before/after the existing bounded oncoming segment."""

    def __init__(self, *, upper_departure=False):
        if not isinstance(upper_departure, bool):
            raise ValueError('upper departure requires explicit boolean opt-in')
        self.upper_departure = upper_departure
        self.phase = "background_walk"
        self.index = 0
        self.encounter_requested = False
        self.events = []

    @property
    def goal(self):
        if self.phase == "approach_parallel":
            return (29.8, 10.5, 0.)
        if self.phase == "move":
            return (29.8, 7., 0.)
        route = self.continuation_route if self.phase == "continue_walk" else BACKGROUND_ROUTE
        return route[self.index]

    @property
    def continuation_route(self):
        return UPPER_DEPARTURE_ROUTE if self.upper_departure else CONTINUATION_ROUTE

    def advance(self, xyz, now, *, encounter_due=False):
        xyz = np.asarray(xyz, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all() or not math.isfinite(now):
            raise ValueError("finite actual actor position and source time required")
        self.encounter_requested |= bool(encounter_due)
        if np.linalg.norm(xyz[:2]-self.goal[:2]) > .20:
            return False
        old_goal, old_phase = self.goal, self.phase
        if self.phase == "background_walk":
            if self.index == len(BACKGROUND_ROUTE)-1 and self.encounter_requested:
                self.phase = "approach_parallel"
            else:
                self.index = (self.index+1) % len(BACKGROUND_ROUTE)
        elif self.phase == "approach_parallel":
            self.phase = "move"
        elif self.phase == "move":
            self.phase, self.index = "continue_walk", 0
        else:
            self.index += 1
            if self.index == len(self.continuation_route):
                self.index = 4 if self.upper_departure else 1
        self.events.append(dict(time=float(now), previous_goal=old_goal,
                               actual_xyz=xyz.tolist(), previous_phase=old_phase,
                               next_phase=self.phase, next_goal=self.goal))
        return True


def audit_background_walk(records, events, *, minimum_background_seconds=20., upper_departure=False):
    """Independent native progress/whole-route geometry gate, no sensor claim."""
    times = np.array([r['time'] for r in records], float)
    xyz = np.array([r['actor_xyz'] for r in records], float)
    phases = np.array([r['phase'] for r in records])
    if len(times) < 30 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        return dict(passed=False, checks=dict(ordered_actual_samples=False))
    speed = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)/np.diff(times)
    background = phases == 'background_walk'
    moving = np.isin(phases[1:], ['background_walk', 'approach_parallel', 'move', 'continue_walk'])
    checks = dict(
        background_walk_exercised=background.sum() > 10 and np.ptp(times[background]) >= minimum_background_seconds,
        background_loop_completed=any(e['previous_phase'] == 'background_walk' and
            np.allclose(e['previous_goal'], BACKGROUND_ROUTE[-1]) for e in events),
        oncoming_segment_completed=any(e['previous_phase'] == 'move' and e['next_phase'] == 'continue_walk' for e in events),
        continuation_exercised=int(np.count_nonzero(phases == 'continue_walk')) >= 60,
        no_native_idle_after_start=not any(r['phase'] == 'idle_before' for r in records[np.flatnonzero(background)[0]:]) if background.any() else False,
        measured_walking=bool(moving.any()) and float(np.mean(speed[moving] >= .2)) >= .8,
        independent_geometry=all(r['actor_layout_clearance_m'] > .02 and r['collider_alignment_error_m'] <= .03 for r in records),
        grounded=bool(np.isfinite(xyz).all() and np.max(np.abs(xyz[:, 2])) <= .05),
        bounded_region=bool(((xyz[:, 0] >= 27.3) & (xyz[:, 0] <= 30.3) &
                             (xyz[:, 1] >= 6.5) & (xyz[:, 1] <= (24.1 if upper_departure else 13.4))).all()))
    if upper_departure:
        checks.update(
            explicit_upper_departure=all(r.get('background_upper_departure') is True for r in records),
            upper_aisle_reached=any(e['previous_phase'] == 'continue_walk' and
                np.allclose(e['previous_goal'], UPPER_DEPARTURE_ROUTE[3]) for e in events))
    return dict(passed=all(checks.values()), checks={k: bool(v) for k, v in checks.items()},
                phase_counts={str(k): int(np.count_nonzero(phases == k)) for k in set(phases)},
                minimum_layout_clearance_m=min(r['actor_layout_clearance_m'] for r in records),
                scope='native actor-only route validation; authoring never supplies robot perception/control')
