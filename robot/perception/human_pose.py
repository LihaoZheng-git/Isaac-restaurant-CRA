"""Audited RGB COCO wrist cues; no simulator joint/hand state is accepted."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time

import numpy as np

from robot.perception.yolo_detector import rgb_to_bgr_uint8

POSE_SHA256 = "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0"
POSE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"


@dataclass(frozen=True)
class HumanPose:
    box_xyxy: tuple[float, float, float, float]
    confidence: float
    keypoints: np.ndarray

    def __post_init__(self):
        box = np.asarray(self.box_xyxy, dtype=float)
        points = np.array(self.keypoints, dtype=float, copy=True)
        if (box.shape != (4,) or not np.isfinite(box).all()
            or np.any(box[2:] <= box[:2]) or not 0 <= self.confidence <= 1
            or points.shape != (17,3) or not np.isfinite(points).all()
            or np.any((points[:,2] < 0) | (points[:,2] > 1))):
            raise ValueError("pose must contain a finite box and 17 COCO keypoints/confidences")
        points.setflags(write=False)
        object.__setattr__(self,"keypoints",points)


@dataclass(frozen=True)
class HumanPoseFrame:
    source_frame_id: int
    source_timestamp_seconds: float
    image_shape: tuple[int, int]
    people: tuple[HumanPose, ...]
    model_name: str = "yolo11n-pose.pt"
    inference_wall_seconds: float = 0.

    def __post_init__(self):
        if (isinstance(self.source_frame_id,bool) or not isinstance(self.source_frame_id,int)
            or self.source_frame_id < 0 or not math.isfinite(self.source_timestamp_seconds)
            or self.source_timestamp_seconds < 0 or len(self.image_shape)!=2
            or any(not isinstance(v,int) or v<=0 for v in self.image_shape)
            or not all(isinstance(p,HumanPose) for p in self.people)):
            raise ValueError("pose frame requires a valid source epoch and image shape")


class YoloHumanPose:
    """Use an existing compatible Ultralytics install, never auto-download."""

    def __init__(self, checkpoint, *, device="0", backend=None):
        self.path, self.device = Path(checkpoint), device
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != POSE_SHA256:
            raise ValueError("pose checkpoint differs from audited official release")
        self.backend = backend

    def initialize(self):
        if self.backend is None:
            from ultralytics import YOLO
            self.backend = YOLO(str(self.path))
            if self.backend.task != "pose":
                raise ValueError("audited pose model did not load as pose")

    def detect(self, rgb, *, source_frame_id, source_timestamp_seconds):
        if self.backend is None:
            raise RuntimeError("initialize pose detector first")
        start = time.perf_counter()
        result, = self.backend.predict(source=rgb_to_bgr_uint8(rgb),conf=.5,
            imgsz=640,device=self.device,verbose=False,save=False)
        def numeric(value):
            return value.cpu().numpy() if hasattr(value,"cpu") else np.asarray(value)
        people = ()
        if result.keypoints is not None:
            people = tuple(HumanPose(tuple(box),float(conf),points)
                for box,conf,points in zip(numeric(result.boxes.xyxy),numeric(result.boxes.conf),numeric(result.keypoints.data)))
        return HumanPoseFrame(source_frame_id,source_timestamp_seconds,
            tuple(rgb.shape[:2]),people,inference_wall_seconds=time.perf_counter()-start)


def match_person_pose(person, frame, *, minimum_iou=.5):
    """Reject competing poses instead of selecting a nearby person's hands."""
    a = np.array([person.bounding_box.x_min,person.bounding_box.y_min,
                  person.bounding_box.x_max,person.bounding_box.y_max])
    candidates=[]
    for pose in frame.people:
        b=np.asarray(pose.box_xyxy)
        intersection=np.prod(np.maximum(0.,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
        iou=float(intersection/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection))
        if iou>=.2 and pose.confidence>=.5:
            candidates.append((iou,pose))
    return candidates[0][1] if len(candidates)==1 and candidates[0][0]>=minimum_iou else None
