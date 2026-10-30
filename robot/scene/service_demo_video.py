"""Read-only native overview recording for the explicit restaurant demo scene."""

import json
import math

import cv2
import numpy as np
from pxr import Gf, Usd, UsdGeom


class ServiceDemoVideo:
    """Rendering only. Frames/epochs never feed perception or control."""

    def __init__(self, stage, output):
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

        self.output, self.rows, self.last = output, [], -math.inf
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            path = "/World/CRAServiceDemoOverview"
            camera = RtxCamera.create(path, tick_rate=10)
            prim = UsdGeom.Camera.Get(stage, path)
            prim.MakeMatrixXform().Set(Gf.Matrix4d().SetLookAt(
                Gf.Vec3d(25.9, 6.7, 3.2), Gf.Vec3d(29., 8., .6),
                Gf.Vec3d(0, 0, 1)).GetInverse())
            prim.CreateFocalLengthAttr(18)
            prim.CreateHorizontalApertureAttr(36)
            prim.CreateVerticalApertureAttr(27)
        self.camera = CameraSensor(camera, resolution=(600, 800), annotators=["rgb"])
        self.writer = cv2.VideoWriter(str(output/"service-navigation.mp4"),
                                      cv2.VideoWriter_fourcc(*"mp4v"), 10, (800, 600))
        if not self.writer.isOpened():
            raise RuntimeError("service overview recording failed to open")

    def record(self, now, command, world_xy):
        if now-self.last < .1-1e-6:
            return
        rgb, _ = self.camera.get_data("rgb")
        rgb = np.array([]) if rgb is None else rgb.numpy()
        if rgb.shape[:2] != (600, 800):
            return
        self.writer.write(np.ascontiguousarray(rgb[:, :, :3][:, :, ::-1]))
        self.rows.append(dict(time=float(now), robot_command=list(command),
                              robot_world_xy=list(world_xy)))
        self.last = now

    def close(self):
        if self.writer is None:
            return
        self.writer.release()
        self.camera.detach_annotators("rgb")
        self.writer = None
        (self.output/"service-video-epochs.json").write_text(json.dumps(dict(
            fps=10, frames=self.rows,
            scope="native overview frame epochs; read-only presentation, not control evidence"),
            indent=2)+"\n", encoding="utf-8")
