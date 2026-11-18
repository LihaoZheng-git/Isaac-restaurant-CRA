"""Read-only native overview recording for the explicit restaurant demo scene."""

import json
import math

import cv2
import numpy as np
from pxr import Gf, Usd, UsdGeom


class ServiceDemoVideo:
    """Rendering only. Frames/epochs never feed perception or control."""

    camera_path = "/World/CRAServiceDemoOverview"
    resolution = (600, 800)

    def __init__(self, stage, output, *, follow_pose=None, risk_camera=False, risk_palette="legacy",
                 resolution=(600, 800)):
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

        self.output, self.rows, self.last = output, [], -math.inf
        self.resolution = resolution
        from robot.evaluation.risk_palette import palette_description
        self.risk_palette = palette_description(risk_palette)["name"]
        self.stage, self.follow_pose, self.pose_time = stage, follow_pose, None
        self.camera_pose_record = None
        self.risk_frame, self.risk_rows, self.risk_writer = None, [], None
        self.risk_rejections = []
        self.phase = None
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            camera = RtxCamera.create(self.camera_path, tick_rate=10)
            prim = UsdGeom.Camera.Get(stage, self.camera_path)
            self.transform = prim.MakeMatrixXform()
            self.transform.Set(Gf.Matrix4d().SetLookAt(
                Gf.Vec3d(25.9, 6.7, 3.2), Gf.Vec3d(29., 8., .6),
                Gf.Vec3d(0, 0, 1)).GetInverse())
            prim.CreateFocalLengthAttr(18)
            prim.CreateHorizontalApertureAttr(36)
            prim.CreateVerticalApertureAttr(27)
        self.camera = CameraSensor(camera, resolution=resolution, annotators=["rgb"])
        self.writer = cv2.VideoWriter(str(output/"service-navigation.mp4"),
                                      cv2.VideoWriter_fourcc(*"mp4v"), 10, (800, 600))
        if not self.writer.isOpened():
            raise RuntimeError("service overview recording failed to open")
        if risk_camera:
            self.risk_writer = cv2.VideoWriter(str(output/"service-risk-camera.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), 10, (1440, 600))
            if not self.risk_writer.isOpened():
                self.writer.release()
                self.camera.detach_annotators("rgb")
                raise RuntimeError("service risk-camera recording failed to open")
            (output/"risk-camera-sources").mkdir(exist_ok=True)
        if follow_pose is not None:
            self.update_pose(0., follow_pose)

    def update_pose(self, now, pose):
        """Update before rendering, from actual pose only; preserve controller pauses."""
        if self.follow_pose is None:
            return
        from robot.scene.presentation_follow_camera import follow_camera_pose
        self.follow_pose, eye, target = follow_camera_pose(
            self.follow_pose, pose, 0. if self.pose_time is None else now-self.pose_time)
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            self.transform.Set(Gf.Matrix4d().SetLookAt(
                Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())
        self.pose_time = now
        self.camera_pose_record = dict(source_time=float(now), eye=list(eye), target=list(target))

    def begin_phase(self, phase):
        """Do not carry an admitted field across a fresh controller phase."""
        self.phase = phase
        self.risk_frame = None

    def record(self, now, command, world_xy, *, world_yaw=None):
        if now-self.last < .1-1e-6:
            return
        rgb, _ = self.camera.get_data("rgb")
        rgb = np.array([]) if rgb is None else rgb.numpy()
        if rgb.shape[:2] != self.resolution:
            return
        if self.resolution != (600, 800):
            rgb = cv2.resize(rgb, (800, 600), interpolation=cv2.INTER_LINEAR)
        self.writer.write(np.ascontiguousarray(rgb[:, :, :3][:, :, ::-1]))
        if getattr(self, "risk_writer", None) is not None:
            from robot.evaluation.risk_camera_overlay import render_risk_panel
            panel, metadata = render_risk_panel(self.risk_frame, now,
                palette=getattr(self, "risk_palette", "legacy"))
            combined = np.concatenate((rgb[:, :, :3], panel), axis=1)
            self.risk_writer.write(np.ascontiguousarray(combined[:, :, ::-1]))
            self.risk_rows.append(dict(time=float(now), **metadata))
        row = dict(time=float(now), robot_command=list(command), robot_world_xy=list(world_xy),
                   presentation_camera=getattr(self, "camera_pose_record", None))
        if world_yaw is not None:
            row.update(robot_world_yaw=float(world_yaw), phase=self.phase)
        self.rows.append(row)
        self.last = now

    def accept_risk_assessment(self, result):
        """Only call after field admission; read its immutable source RGB/masks."""
        if getattr(self, "risk_writer", None) is None:
            return
        frame = result.risk_camera_frame
        self.risk_frame = None
        if frame is None:
            return
        if (frame.sensor_frame_id != result.field.sensor_frame_id or
                frame.sensor_timestamp_seconds != result.field.sensor_timestamp_seconds):
            self.risk_rejections.append(dict(reason="admitted_field_source_mismatch",
                sensor_frame_id=frame.sensor_frame_id))
            return
        self.risk_frame = frame
        if frame.masks:
            from robot.evaluation.risk_camera_overlay import render_risk_panel
            stem = self.output/"risk-camera-sources"/str(frame.camera_frame_id)
            np.savez_compressed(str(stem)+".npz", rgb=frame.rgb,
                masks=np.stack([m.mask for m in frame.masks]))
            stem.with_suffix(".json").write_text(json.dumps(frame.diagnostic(), indent=2)+"\n",
                                                encoding="utf-8")
            panel, _ = render_risk_panel(frame, frame.camera_timestamp_seconds,
                palette=getattr(self, "risk_palette", "legacy"))
            cv2.imwrite(str(self.output/"risk-camera-latest.png"), panel[:, :, ::-1])

    def close(self):
        if self.writer is None:
            return
        self.writer.release()
        if getattr(self, "risk_writer", None) is not None:
            self.risk_writer.release()
            self.risk_writer = None
            (self.output/"risk-camera-epochs.json").write_text(json.dumps(dict(
                fps=10, frames=self.risk_rows,
                rejected_assessments=self.risk_rejections,
                scope="latest assessed source RGB; fixed CRA phi scale; held frames show source age; no control input"),
                indent=2)+"\n", encoding="utf-8")
        self.camera.detach_annotators("rgb")
        self.writer = None
        (self.output/"service-video-epochs.json").write_text(json.dumps(dict(
            fps=10, frames=self.rows,
            rendered_height_width=self.resolution,
            scope="native overview frame epochs; read-only presentation, not control evidence"),
            indent=2)+"\n", encoding="utf-8")
