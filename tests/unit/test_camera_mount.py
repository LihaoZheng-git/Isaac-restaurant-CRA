import numpy as np
import pytest

from robot.scene.camera_mount import pitched_camera_transform


def test_camera_mount_rotates_local_optical_axis_without_moving_origin():
    # USD camera axes in robot coordinates: right=-Y, up=Z, forward=X.
    original = np.array(
        [[0, 0, -1, 0.3], [-1, 0, 0, 0.1], [0, 1, 0, 0.5], [0, 0, 0, 1]]
    )
    result = pitched_camera_transform(original, 15)
    np.testing.assert_array_equal(result[:3, 3], original[:3, 3])
    np.testing.assert_allclose(
        -result[:3, 2], [np.cos(np.pi / 12), 0, np.sin(np.pi / 12)]
    )
    np.testing.assert_allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-15)
    np.testing.assert_array_equal(pitched_camera_transform(original, 0), original)


@pytest.mark.parametrize("pitch", [-1, 26, float("nan"), float("inf")])
def test_camera_mount_rejects_unbounded_pitch(pitch):
    with pytest.raises(ValueError):
        pitched_camera_transform(np.eye(4), pitch)


def test_session_mount_raise_preserves_lens_source_and_rereads_actual_pose():
    from pxr import Gf, Usd, UsdGeom
    from robot.scene.camera_mount import apply_experimental_camera_pitch

    stage = Usd.Stage.CreateInMemory()
    base = UsdGeom.Xform.Define(stage, "/Base")
    base.AddTranslateOp().Set(Gf.Vec3d(3, 4, 0.1))
    base.AddRotateZOp().Set(37.0)
    camera = UsdGeom.Camera.Define(stage, "/Base/Camera")
    local = np.array([[0, 0, -1, 0.3], [-1, 0, 0, 0.1], [0, 1, 0, 0.2], [0, 0, 0, 1]])
    camera.MakeMatrixXform().Set(Gf.Matrix4d(local.T.tolist()))
    camera.CreateFocalLengthAttr(18.0)
    source_before = stage.GetRootLayer().ExportToString()
    audit = apply_experimental_camera_pitch(
        stage, "/Base/Camera", 15, "/Base", raise_metres=0.45
    )
    assert audit["before_base_origin"] == pytest.approx([0.3, 0.1, 0.2])
    assert audit["after_base_origin"] == pytest.approx([0.3, 0.1, 0.65])
    assert audit["after_base_forward"][2] == pytest.approx(np.sin(np.pi / 12))
    assert camera.GetFocalLengthAttr().Get() == 18.0
    assert stage.GetRootLayer().ExportToString() == source_before
    # Re-read USD pose rather than merely trusting the intended matrix.
    observed_world = camera.ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    ).ExtractTranslation()
    assert observed_world[2] == pytest.approx(0.75)
