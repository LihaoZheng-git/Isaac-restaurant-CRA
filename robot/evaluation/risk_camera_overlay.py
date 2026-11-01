"""Source-aligned CRA masks for presentation only; no control inputs or inference."""

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class RiskCameraMask:
    track_id: int
    phi: float
    applied_phi: float
    factor_modes: tuple
    mask: np.ndarray

    @property
    def basis(self):
        modes = [mode for _, mode in self.factor_modes]
        if len(modes) == 4 and all(mode in {
            "configured_prior", "configured_human_body_contact_prior"
        } for mode in modes):
            return "factor priors"
        if len(modes) == 4 and all(mode == "evidence_supported" for mode in modes):
            return "factor evidence*"
        return "evidence + priors*"

    def diagnostic(self):
        return dict(track_id=self.track_id, phi=self.phi,
                    applied_phi=self.applied_phi, basis=self.basis,
                    factor_modes=dict(self.factor_modes), pixels=int(self.mask.sum()))


@dataclass(frozen=True)
class RiskCameraFrame:
    sensor_frame_id: int
    sensor_timestamp_seconds: float
    camera_frame_id: int
    camera_timestamp_seconds: float
    rgb: np.ndarray
    masks: tuple
    rejected: tuple

    def diagnostic(self):
        return dict(sensor_frame_id=self.sensor_frame_id,
                    sensor_timestamp_seconds=self.sensor_timestamp_seconds,
                    camera_frame_id=self.camera_frame_id,
                    camera_timestamp_seconds=self.camera_timestamp_seconds,
                    masks=[m.diagnostic() for m in self.masks],
                    rejected=list(self.rejected))


def capture_risk_camera_frame(packet, objects, assessments, policies):
    """Copy only current accepted masks, paired to their original RGB and policy.

    Rejected presentation records never change perception acceptance or control.
    Policy-only, retained bounds, raw person cues and unseen tracks cannot create
    a colored mask. The individual CRA score and applied/group score stay separate.
    """
    masks, rejected = [], []
    if len(objects) != len(assessments):
        rejected.append(dict(reason="object_assessment_count_mismatch"))
        objects, assessments = (), ()
    for obj, raw in zip(objects, assessments):
        detail = obj.diagnostics
        if not (detail.get("accepted_instance") and raw.get("accepted")):
            continue
        track_id = obj.attributes.track_id
        frame = raw["frame"]
        reason = None
        if (detail.get("track_id") != track_id
                or raw["attributes"].track_id != track_id):
            reason = "object_identity_mismatch"
        elif (frame.source_frame_id != packet.camera_frame_id or
              frame.source_timestamp_seconds != packet.camera_timestamp_seconds):
            reason = "camera_source_mismatch"
        elif (len(frame.detections) != 1 or
              frame.detections[0].mask.shape != packet.rgb.shape[:2]):
            reason = "mask_shape_or_count_mismatch"
        elif (np.asarray(raw["selected"]).shape != (len(packet.obstacle_xyz),) or
              np.count_nonzero(raw["selected"]) < 4 or
              detail.get("selected_return_count") != np.count_nonzero(raw["selected"])):
            reason = "missing_verified_lidar_support"
        phi = float(obj.policy.risk)
        applied = float(policies[track_id].risk) if track_id in policies else math.nan
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (phi, applied)):
            reason = "invalid_phi"
        if reason is not None:
            rejected.append(dict(track_id=int(track_id), reason=reason))
            continue
        mask = np.array(frame.detections[0].mask, dtype=bool, copy=True)
        if not np.any(mask):
            rejected.append(dict(track_id=int(track_id), reason="empty_mask"))
            continue
        mask.setflags(write=False)
        modes = tuple(sorted((key, value["mode"])
                             for key, value in detail["factor_scores"].items()))
        masks.append(RiskCameraMask(int(track_id), phi, applied, modes, mask))
    rgb = np.array(packet.rgb, dtype=np.uint8, copy=True)
    rgb.setflags(write=False)
    return RiskCameraFrame(int(packet.sensor_frame_id), float(packet.sensor_timestamp_seconds),
                           int(packet.camera_frame_id), float(packet.camera_timestamp_seconds),
                           rgb, tuple(masks), tuple(rejected))


def risk_color_rgb(phi):
    """Fixed piecewise-linear 0..1 scale: green, yellow, orange, red."""
    if not math.isfinite(phi) or not 0 <= phi <= 1:
        raise ValueError("phi must be finite and within [0, 1]")
    stops = np.array([0., .5, .75, 1.])
    colors = np.array([[35, 195, 85], [245, 220, 45], [245, 135, 35], [225, 45, 45]])
    return tuple(int(round(np.interp(phi, stops, colors[:, i]))) for i in range(3))


def overlay_rgb(frame):
    """Maximum CRA score wins in overlapping accepted masks, independent of order."""
    image = frame.rgb.copy()
    risk = np.full(image.shape[:2], -1., dtype=float)
    for item in frame.masks:
        risk[item.mask] = np.maximum(risk[item.mask], item.phi)
    for phi in np.unique(risk[risk >= 0]):
        selected = risk == phi
        image[selected] = np.rint(.62 * image[selected] +
                                  .38 * np.array(risk_color_rgb(float(phi)))).astype(np.uint8)
    for item in sorted(frame.masks, key=lambda m: (m.phi, m.track_id)):
        contours, _ = cv2.findContours(item.mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        color = risk_color_rgb(item.phi)
        cv2.drawContours(image, contours, -1, color, 1)
        yy, xx = np.nonzero(item.mask)
        label = f"ID {item.track_id}  phi {item.phi:.2f}"
        x, y = min(int(xx.min()), max(0, image.shape[1]-180)), max(15, int(yy.min())-4)
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 3)
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
    return image


def render_risk_panel(frame, now):
    """640x600 RGB panel. Hold assessed pixels, never paste masks on a new view."""
    canvas = np.full((600, 640, 3), (22, 27, 34), dtype=np.uint8)

    def line(text, y, color=(225, 231, 238), scale=.45):
        cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1,
                    cv2.LINE_AA)

    line("ROBOT CAMERA | VERIFIED IMAGE + LiDAR", 25, scale=.58)
    metadata = dict(status="awaiting_assessment", source_age_seconds=None, masks=[])
    if frame is None:
        line("Waiting for the first source-aligned assessment", 240)
    elif not math.isfinite(now) or now < frame.camera_timestamp_seconds:
        line("Assessment withheld: camera source is in the future", 240)
        metadata["status"] = "invalid_presentation_time"
    else:
        age = now-frame.camera_timestamp_seconds
        metadata = dict(status="latest_assessed_frame", source_age_seconds=age,
                        **frame.diagnostic())
        line(f"Assessed frame {frame.camera_frame_id} | source {frame.camera_timestamp_seconds:.2f}s | age {age:.2f}s", 47)
        picture = overlay_rgb(frame)
        height, width = picture.shape[:2]
        scale = min(640/width, 400/height)
        size = (max(1, round(width*scale)), max(1, round(height*scale)))
        picture = cv2.resize(picture, size, interpolation=cv2.INTER_AREA)
        x, y = (640-size[0])//2, 58+(400-size[1])//2
        canvas[y:y+size[1], x:x+size[0]] = picture
        if frame.masks:
            for index, item in enumerate(frame.masks[:2]):
                text = f"ID {item.track_id}: phi {item.phi:.3f} | {item.basis}"
                if not math.isclose(item.phi, item.applied_phi, abs_tol=1e-9):
                    text += f" | field {item.applied_phi:.3f}"
                line(text, 478+18*index, risk_color_rgb(item.phi), .42)
            if len(frame.masks) > 2:
                line(f"+ {len(frame.masks)-2} masks; full provenance in frame ledger", 512, scale=.4)
        else:
            line("No accepted image-LiDAR masks in this assessment", 481)
        if frame.rejected:
            line(f"{len(frame.rejected)} presentation record(s) withheld; see ledger", 512, scale=.4)
    for x in range(24, 616):
        canvas[531:543, x] = risk_color_rgb((x-24)/591)
    line("0  Lower assessed risk", 561, scale=.43)
    cv2.putText(canvas, "Higher assessed risk  1", (438, 561), cv2.FONT_HERSHEY_SIMPLEX,
                .43, (225, 231, 238), 1, cv2.LINE_AA)
    line("* May include retained/inferred evidence. Green is not a safety guarantee.", 580, scale=.38)
    line("Latest assessed RGB held between updates; presentation only.", 595, scale=.38)
    return canvas, metadata
