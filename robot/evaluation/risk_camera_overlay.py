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
    retained_lidar_points: bool = False

    @property
    def basis(self):
        if self.retained_lidar_points:
            return 'retained evidence / LiDAR points'
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


def capture_risk_camera_frame(packet, objects, assessments, policies, *, current_ownership_only=False):
    """Copy only current accepted masks, paired to their original RGB and policy.

    Rejected presentation records never change perception acceptance or control.
    Policy-only, retained bounds, raw person cues and unseen tracks cannot create
    a colored mask. The individual CRA score and applied/group score stay separate.
    The optional calibrated presentation also withholds accepted views that own
    no current policy returns; it never borrows another owner's mask or score.
    """
    if not isinstance(current_ownership_only, bool):
        raise TypeError("current_ownership_only must be bool")
    masks, rejected = [], []
    if len(objects) != len(assessments):
        rejected.append(dict(reason="object_assessment_count_mismatch"))
        objects, assessments = (), ()
    for obj, raw in zip(objects, assessments):
        detail = obj.diagnostics
        mask_candidate = (detail.get("accepted_instance") and raw.get("accepted") or
                          detail.get("static_prop_policy") and raw.get("static_prop_policy"))
        if current_ownership_only and mask_candidate and detail.get("policy_return_count", 0) <= 0:
            rejected.append(dict(track_id=obj.attributes.track_id,
                reason="no_current_policy_return_ownership", policy_return_count=detail.get("policy_return_count", 0)))
            continue
        if detail.get('static_prop_policy') and raw.get('static_prop_policy'):
            # Explicit point overlay, not a fabricated current image mask.
            pixels = np.asarray(raw.get('static_prop_display_pixels', []))
            selected = np.asarray(raw['selected'])
            valid = (raw['attributes'].track_id == obj.attributes.track_id
                     and raw['frame'].source_frame_id == packet.camera_frame_id
                     and raw['frame'].source_timestamp_seconds == packet.camera_timestamp_seconds
                     and selected.shape == (len(packet.obstacle_xyz),)
                     and selected.sum() == detail.get('selected_return_count')
                     and selected.sum() >= 8 and pixels.ndim == 2 and pixels.shape[1] == 2)
            if not valid:
                rejected.append(dict(track_id=obj.attributes.track_id, reason='invalid_retained_point_overlay'))
                continue
            mask = np.zeros(packet.rgb.shape[:2], np.uint8)
            for x,y in pixels:
                if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                    cv2.circle(mask, (int(x),int(y)), 1, 1, -1)
            if mask.any():
                mask = mask.astype(bool)
                mask.setflags(write=False)
                modes = tuple(sorted((k,v['mode']) for k,v in detail['factor_scores'].items()))
                masks.append(RiskCameraMask(obj.attributes.track_id, float(obj.policy.risk),
                    float(policies[obj.attributes.track_id].risk), modes, mask, True))
            continue
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
    from robot.evaluation.risk_palette import risk_color_rgb as fixed_color
    return fixed_color(phi)


def overlay_rgb(frame, *, palette="legacy"):
    """Maximum CRA score wins in overlapping accepted masks, independent of order."""
    image = frame.rgb.copy()
    risk = np.full(image.shape[:2], -1., dtype=float)
    for item in frame.masks:
        risk[item.mask] = np.maximum(risk[item.mask], item.phi)
    for phi in np.unique(risk[risk >= 0]):
        selected = risk == phi
        image[selected] = np.rint(.62 * image[selected] +
                                  .38 * np.array(_color(float(phi), palette))).astype(np.uint8)
    for item in sorted(frame.masks, key=lambda m: (m.phi, m.track_id)):
        contours, _ = cv2.findContours(item.mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        color = _color(item.phi, palette)
        cv2.drawContours(image, contours, -1, color, 1)
        yy, xx = np.nonzero(item.mask)
        label = f"ID {item.track_id}  phi {item.phi:.2f}"
        x, y = min(int(xx.min()), max(0, image.shape[1]-180)), max(15, int(yy.min())-4)
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 3)
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
    return image


def _color(phi, palette):
    from robot.evaluation.risk_palette import risk_color_rgb as fixed_color
    return fixed_color(phi, palette=palette)


def camera_metadata_note(mask_count, rejected_count):
    """One compact footer; extra masks and withheld records never overlap."""
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 0
           for n in (mask_count, rejected_count)):
        raise ValueError("camera metadata counts must be nonnegative integers")
    notes = []
    if mask_count > 2:
        notes.append(f"+ {mask_count-2} masks")
    if rejected_count:
        notes.append(f"{rejected_count} records withheld")
    return "; ".join([*notes, "see ledger"]) if notes else None


def repair_camera_metadata_notes(panel, epoch):
    """Fix only the baked metadata strip, using the exact native epoch counts.

    The original camera-image region, masks, score labels, color bar and timing
    text remain byte-identical. This does not redraw or estimate any object.
    """
    panel = np.asarray(panel)
    if panel.dtype != np.uint8 or panel.shape != (600, 640, 3):
        raise ValueError("native camera panel must be 640x600 RGB uint8")
    if epoch["status"] not in {"awaiting_assessment", "latest_assessed_frame"}:
        raise ValueError("unsupported native camera epoch for metadata repair")
    count, rejected = len(epoch["masks"]), len(epoch.get("rejected", []))
    note = camera_metadata_note(count, rejected)
    result = panel.copy()
    strip = np.full((23, 640, 3), (22, 27, 34), np.uint8)
    if note:
        cv2.putText(strip, note, (12, 10), cv2.FONT_HERSHEY_SIMPLEX, .4,
                    (225, 231, 238), 1, cv2.LINE_AA)
    result[502:525] = strip
    return result, dict(basis="native_epoch_counts_metadata_layout_only",
        changed_row_range=[502,525], mask_count=count, rejected_count=rejected, note=note,
        sensor_frame_id=epoch.get("sensor_frame_id"), camera_frame_id=epoch.get("camera_frame_id"))


def render_risk_panel(frame, now, *, palette="legacy"):
    """640x600 RGB panel. Hold assessed pixels, never paste masks on a new view."""
    canvas = np.full((600, 640, 3), (22, 27, 34), dtype=np.uint8)

    def line(text, y, color=(225, 231, 238), scale=.45):
        cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1,
                    cv2.LINE_AA)

    retained_points = frame is not None and any(m.retained_lidar_points for m in frame.masks)
    line("ROBOT CAMERA | RETAINED EVIDENCE + LiDAR" if retained_points
         else "ROBOT CAMERA | VERIFIED IMAGE + LiDAR", 25, scale=.51 if retained_points else .58)
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
        picture = overlay_rgb(frame, palette=palette)
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
                line(text, 478+18*index, _color(item.phi, palette), .42)
            if len(frame.masks) > 2:
                line(f"+ {len(frame.masks)-2} masks; full provenance in frame ledger", 512, scale=.4)
        else:
            line("No accepted image-LiDAR masks in this assessment", 481)
        if frame.rejected:
            line(f"{len(frame.rejected)} presentation record(s) withheld; see ledger", 512, scale=.4)
    for x in range(24, 616):
        canvas[531:543, x] = _color((x-24)/591, palette)
    line("0  Lower assessed risk", 561, scale=.43)
    cv2.putText(canvas, "Higher assessed risk  1", (438, 561), cv2.FONT_HERSHEY_SIMPLEX,
                .43, (225, 231, 238), 1, cv2.LINE_AA)
    line("* May include retained/inferred evidence. Green is not a safety guarantee.", 580, scale=.38)
    line("Latest assessed RGB held between updates; presentation only.", 595, scale=.38)
    return canvas, metadata
