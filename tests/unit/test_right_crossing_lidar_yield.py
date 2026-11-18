"""A current crossing requests a yield until its LiDAR returns clear the lane."""

from types import SimpleNamespace

import numpy as np
import pytest

from robot.evaluation.restaurant_right_route import RIGHT_GUIDES
from robot.evaluation.right_crossing_lidar_yield import RightCrossingLiDARYield


@pytest.mark.parametrize(('segment', 'speed'), [(0, 1.5), (3, 0.6)], ids=['boy', 'trolley'])
def test_yield_releases_even_when_the_departing_track_is_still_visible(segment, speed):
    start, end = np.array(RIGHT_GUIDES[segment:segment + 2])
    forward = (end - start) / np.linalg.norm(end - start)
    across = np.array([-forward[1], forward[0]])
    controller = RightCrossingLiDARYield('delivery')
    offsets = np.linspace(-0.03, 0.03, 24)[:, None] * forward
    requests = []

    # Send repeated, time-ordered surface observations through the actual delivery selector.
    for tick, time in enumerate(np.arange(0.0, 1.6, 0.1)):
        center = start + forward + across * (-0.45 + speed * time)
        track = SimpleNamespace(track_id=7, matched_previous_track=True, observation_count=tick + 3,
                                centroid_world_xy=center, velocity_world_xy=across * speed)
        tracking = SimpleNamespace(timestamp_seconds=time, tracked_obstacles=[track],
                                   point_track_ids=np.full(len(offsets), 7))
        result = controller.update(tracking, center + offsets, robot_xy=start)
        requests.append(result['yield_track_ids'])

    assert requests[:2] == [(), ()]
    assert requests[2] == (7,)
    assert requests[-1] == ()
