from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from robot.perception.human_pose import HumanPose, HumanPoseFrame, YoloHumanPose


def test_pose_adapter_requires_an_existing_audited_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        YoloHumanPose(tmp_path / "absent.pt")
    checkpoint = tmp_path / "other.pt"
    checkpoint.write_bytes(b"not the official checkpoint")
    with pytest.raises(ValueError, match="audited"):
        YoloHumanPose(checkpoint)


def test_pose_adapter_preserves_source_epoch_and_rgb_conversion(tmp_path, monkeypatch):
    import robot.perception.human_pose as module

    checkpoint = tmp_path / "controlled.pt"
    checkpoint.write_bytes(b"controlled backend test only")
    monkeypatch.setattr(
        module, "POSE_SHA256", sha256(checkpoint.read_bytes()).hexdigest()
    )
    calls = []
    keypoints = np.zeros((1, 17, 3))
    keypoints[0, 9] = [11, 12, 0.8]

    def predict(**kwargs):
        calls.append(kwargs)
        return [
            SimpleNamespace(
                boxes=SimpleNamespace(
                    xyxy=np.array([[1, 2, 20, 30]]), conf=np.array([0.9])
                ),
                keypoints=SimpleNamespace(data=keypoints),
            )
        ]

    adapter = YoloHumanPose(checkpoint, backend=SimpleNamespace(predict=predict))
    rgb = np.zeros((40, 50, 3), np.uint8)
    rgb[0, 0] = [1, 2, 3]
    frame = adapter.detect(rgb, source_frame_id=7, source_timestamp_seconds=1.2)
    assert frame.source_frame_id == 7 and frame.source_timestamp_seconds == 1.2
    assert frame.image_shape == (40, 50)
    assert frame.people[0].keypoints[9, 2] == 0.8
    np.testing.assert_array_equal(calls[0]["source"][0, 0], [3, 2, 1])
    assert calls[0]["save"] is False
    assert not frame.people[0].keypoints.flags.writeable


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_frame_id", -1),
        ("source_frame_id", True),
        ("source_timestamp_seconds", np.nan),
        ("image_shape", (0, 40)),
    ],
)
def test_pose_frame_rejects_invalid_epochs_and_shape(field, value):
    fields = dict(
        source_frame_id=1, source_timestamp_seconds=1.0, image_shape=(40, 50), people=()
    )
    fields[field] = value
    with pytest.raises(ValueError, match="source epoch"):
        HumanPoseFrame(**fields)


def test_pose_rejects_wrong_keypoint_contract_and_nonfinite_geometry():
    with pytest.raises(ValueError, match="17 COCO"):
        HumanPose((0, 0, 10, 20), 0.8, np.zeros((16, 3)))
    with pytest.raises(ValueError, match="finite box"):
        HumanPose((0, 0, np.inf, 20), 0.8, np.zeros((17, 3)))
