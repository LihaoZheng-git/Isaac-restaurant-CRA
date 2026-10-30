"""Overview timestamps preserve sampled simulation timing, without control hooks."""

from types import SimpleNamespace
import math

import numpy as np

from robot.scene.service_demo_video import ServiceDemoVideo


def test_video_uses_native_camera_tuple_and_retains_pause_samples():
    samples = []
    camera = SimpleNamespace(get_data=lambda name: (
        SimpleNamespace(numpy=lambda: np.zeros((600, 800, 4), dtype=np.uint8)), None))
    recorder = ServiceDemoVideo.__new__(ServiceDemoVideo)
    recorder.camera = camera
    recorder.writer = SimpleNamespace(write=samples.append)
    recorder.last, recorder.rows = -math.inf, []
    for now in (1., 1.01, 1.1, 1.2):
        recorder.record(now, (0., 0.), (4., 5.))
    assert len(samples) == 3
    assert [r['time'] for r in recorder.rows] == [1., 1.1, 1.2]
    assert all(r['robot_command'] == [0., 0.] for r in recorder.rows)
    assert all(image.shape == (600, 800, 3) for image in samples)


def test_follow_camera_stays_behind_robot_and_wraps_yaw_smoothly():
    from robot.scene.presentation_follow_camera import follow_camera_pose
    old = (12., 4., math.pi-.01)
    smooth, eye, target = follow_camera_pose(old, (12., 4., -math.pi+.01), .1)
    assert abs(smooth[2]-old[2]) < .02
    assert eye[0] > 12 and target[0] < 12
    assert eye[2] == 2.5 and target[2] == .55
    stopped, _, _ = follow_camera_pose(smooth, smooth, .1)
    np.testing.assert_allclose(stopped, smooth)
