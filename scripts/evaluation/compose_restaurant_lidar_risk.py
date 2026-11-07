"""Compose unchanged native camera frames with source-aligned applied LiDAR risk."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robot.evaluation.lidar_risk_map import render_lidar_risk_map
from robot.evaluation.risk_camera_overlay import repair_camera_metadata_notes
from robot.evaluation.risk_palette import palette_description
from scripts.evaluation.compose_restaurant_v3_layout import (
    contact_mission_layout, digest, read, verify_archive,
)


def validate_frame_index(manifest, epochs, risk_epochs=None):
    """No retiming, dropped frames or future field commits in the visual join."""
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported risk manifest schema")
    frames = manifest["frames"]
    if len(frames) != len(epochs):
        raise ValueError("risk map/source frame count differs")
    if risk_epochs is not None and len(risk_epochs) != len(epochs):
        raise ValueError("camera risk epoch count differs")
    previous_commit = -np.inf
    for index, (frame, epoch) in enumerate(zip(frames, epochs)):
        if (frame["source_frame"] != index or not np.isfinite((frame["time"], epoch["time"])).all()
                or abs(frame["time"] - epoch["time"]) > 1e-8):
            raise ValueError("risk map/source frame timing differs")
        if risk_epochs is not None:
            camera = risk_epochs[index]
            if abs(camera["time"] - epoch["time"]) > 1e-8:
                raise ValueError("camera risk/source timing differs")
            if camera["status"] == "latest_assessed_frame":
                if (frame["geometry_file"] is None or int(frame["field_id"]) != camera["sensor_frame_id"]
                        or abs(frame["source_time"] - camera["sensor_timestamp_seconds"]) > 1e-8):
                    raise ValueError("risk field differs from recorded current camera/field receipt")
            elif camera["status"] != "awaiting_assessment" or frame["geometry_file"] is not None:
                raise ValueError("unknown epoch state or fabricated committed field")
        if frame["geometry_file"] is not None:
            source, commit, now = (frame[k] for k in ("source_time", "commit_time", "time"))
            if not np.isfinite((source, commit, now)).all() or not source <= commit + 1e-8 <= now + 2e-8:
                raise ValueError("future or invalid risk field")
            if commit < previous_commit - 1e-8:
                raise ValueError("risk field selection moves backwards")
            previous_commit = commit
    return frames


def geometry_path(folder, name):
    path = (folder / name).resolve()
    if not path.is_relative_to(folder.resolve()) or path.suffix != ".npz":
        raise ValueError("geometry file must be an NPZ inside the export folder")
    return path


def load_verified_geometry(folder, name, expected_sha256):
    path = geometry_path(folder, name)
    actual = digest(path)
    if actual != expected_sha256:
        raise ValueError(f"export geometry hash mismatch: {name}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return arrays, path, actual


def presentation_options(*, replay=False, palette="legacy", title=None, video_name=None,
                         repair_camera_notes=False):
    """Presentation-only controls, with original legacy defaults unchanged."""
    palette_description(palette)
    if replay:
        if repair_camera_notes:
            raise ValueError("baked native camera-note repair is separate from score replay")
        palette = "restaurant_v1"
    title = title if title is not None else (
        "V3 | restaurant risk calibration replay" if replay else "Restaurant service | v4")
    if not isinstance(title, str) or not title.strip() or "\n" in title:
        raise ValueError("presentation title must be one nonempty line")
    filename = video_name if video_name is not None else (
        "restaurant-risk-calibration-replay.mp4" if replay else "restaurant-service-v4-lidar-risk.mp4")
    if (not isinstance(filename, str) or Path(filename).name != filename
            or "/" in filename or "\\" in filename or Path(filename).suffix != ".mp4"):
        raise ValueError("video name must be an MP4 basename inside the new output")
    if not isinstance(repair_camera_notes, bool):
        raise TypeError("repair_camera_notes must be bool")
    return dict(palette=palette, title=title, video_name=filename,
                repair_camera_notes=repair_camera_notes)


def main(source, archive, risk_manifest, output, preview_only=False, preserve=(), calibration_replay=None,
         *, palette="legacy", title=None, video_name=None, repair_camera_notes=False):
    if output.exists():
        raise ValueError("use a fresh presentation folder; existing demos are preserved")
    protected = {folder.resolve(): verify_archive(folder) for folder in (archive, *preserve)}
    manifest = read(risk_manifest)
    options = presentation_options(replay=calibration_replay is not None, palette=palette,
        title=title, video_name=video_name, repair_camera_notes=repair_camera_notes)
    replay = None
    if calibration_replay is not None:
        from robot.evaluation.calibration_replay import CalibrationReplay, candidate_geometry
        from robot.evaluation.risk_camera_overlay import render_risk_panel
        replay = CalibrationReplay(calibration_replay, source)
        if Path(replay.manifest["original_manifest"]).resolve() != risk_manifest.resolve():
            raise ValueError("candidate scoring is bound to a different geometry export")
    if Path(manifest["source_directory"]).resolve() != source.resolve():
        raise ValueError("risk export belongs to a different native mission")
    if manifest.get("source_hashes_verified_after_export") is not True:
        raise ValueError("completed source-preserving risk export required")
    if not replay:
        command = read(source / "invocation.json")["command"]
        native_palette = ("restaurant_v1" if "--restaurant-calibration" in command
            and command[command.index("--restaurant-calibration")+1] != "off" else "legacy")
        if options["palette"] != native_palette:
            raise ValueError("LiDAR palette must match the recorded native camera palette")
    epochs = read(source / "service-video-epochs.json")["frames"]
    risk_epochs = read(source / "risk-camera-epochs.json")["frames"]
    if [r["time"] for r in epochs] != [r["time"] for r in risk_epochs]:
        raise ValueError("native camera source epochs differ")
    fields = validate_frame_index(manifest, epochs, risk_epochs)
    layout, _, status = contact_mission_layout(source, epochs)
    first, last = layout["first_source_frame"], layout["last_source_frame"]
    credits_layout = read(archive / "restaurant-service-v3.json")
    route = np.asarray(manifest["nominal_route_world"], dtype=float)
    footprint = manifest.get("robot_footprint_local_xy")
    inputs = [risk_manifest, source / "service-risk-camera.mp4", source / "invocation.json",
              source / "service-video-epochs.json", source / "risk-camera-epochs.json",
              archive / "restaurant-service-v3.mp4", archive / "restaurant-service-v3.json",
              archive / "ASSET_CREDITS.txt"]
    inputs.extend(source / f"{name}.json" for name in ("stationary-fields", "report", "execution",
                  "static-prop-audit", "contact-coverage", "camera-presentation-audit"))
    hashes = {str(path.resolve()): digest(path) for path in inputs}
    cap = cv2.VideoCapture(str(source / "service-risk-camera.mp4"))
    if (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(epochs)
            or cap.get(cv2.CAP_PROP_FPS) != 10
            or (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (1440, 600)):
        raise ValueError("native video shape/epochs differ")
    previews = {first, 1030, 1270, last}
    cover = read(source / "contact-coverage.json")["summary"]
    for item in cover.values():
        ids = set(item["fresh_ids"] + item["retained_ids"])
        pixels, index = max((sum(m["pixels"] for m in risk_epochs[i]["masks"] if m["track_id"] in ids), i)
                            for i in range(first, last + 1))
        if pixels:
            previews.add(index)
    # If present, illustrate retained properties without any camera overlay.
    # Select from actual records; never fabricate such an assessment for a card.
    field_details = {row["field_id"]: row for row in manifest["fields"]}
    candidates = [i for i in range(first, last + 1)
                  if not risk_epochs[i]["masks"] and fields[i]["geometry_file"] is not None
                  and field_details[fields[i]["field_id"]]["retained_evidence_count"] > 0]
    sensor_only_retained_preview = None
    checked = set()
    for index in sorted(candidates, key=lambda i: field_details[fields[i]["field_id"]]["retained_evidence_count"], reverse=True):
        name = fields[index]["geometry_file"]
        if name in checked:
            continue
        checked.add(name)
        data, path, verified = load_verified_geometry(risk_manifest.parent, name, manifest["geometry_sha256"][name])
        hashes[str(path)] = verified
        if np.any((data["point_basis"] == 2) & data["point_in_field"]):
            previews.add(index)
            sensor_only_retained_preview = index
            break
    output.mkdir(parents=True)
    fonts = {size: ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", size) for size in (18, 20, 22, 30)}
    video_path = output / options["video_name"]
    writer = None if preview_only else cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1920, 1080))
    if writer is not None and not writer.isOpened():
        raise RuntimeError("cannot open presentation output")
    ledger, travelled = [], []
    last_file, geometry, replay_row, camera_frame = None, None, None, None
    native_count = credit_count = 0
    try:
        for index, epoch in enumerate(epochs):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"missing original frame {index}")
            if index < first:
                continue
            travelled.append(epoch["robot_world_xy"])
            if preview_only and index not in previews:
                continue
            field = fields[index]
            if replay and camera_frame is not None and field["geometry_file"] == last_file:
                for key in ("sensor_frame_id", "sensor_timestamp_seconds", "camera_frame_id", "camera_timestamp_seconds"):
                    if getattr(camera_frame, key) != risk_epochs[index][key]:
                        raise ValueError("held camera epoch changed without a new field")
            name = field["geometry_file"]
            if name is None:
                geometry, last_file = None, None
                replay_row, camera_frame = None, None
            elif name != last_file:
                geometry, path, verified = load_verified_geometry(
                    risk_manifest.parent, name, manifest["geometry_sha256"][name])
                if not {"point_in_field", "field_bounds_world"} <= geometry.keys():
                    raise ValueError("native presentation requires original active-field membership and boundary")
                hashes[str(path)] = verified
                last_file = name
                if replay:
                    replay_row = replay.field(field)
                    geometry = candidate_geometry(geometry, replay_row)
                    camera_frame = replay.camera(risk_epochs[index], replay_row)
            sensor_map, map_audit = render_lidar_risk_map(
                geometry, field, route, travelled, robot_footprint_local_xy=footprint,
                **(dict(palette=options["palette"], score_title="CANDIDATE phi | REPLAY", replay_scores=True)
                   if replay else dict(palette=options["palette"])))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            canvas = Image.new("RGB", (1920, 1080), "#101b26")
            draw = ImageDraw.Draw(canvas)
            elapsed = max(0.0, epoch["time"] - layout["navigation_start_simulation_time"])
            draw.text((24, 14), options["title"], font=fonts[30], fill="white")
            draw.text((1016, 20), f"Simulation time {elapsed:5.1f} s  |  1x playback", font=fonts[22], fill="#cbd9e3")
            camera_panel = rgb[:, 800:1440]
            if replay:
                camera_panel, _ = render_risk_panel(camera_frame, epoch["time"], palette="restaurant_v1")
            note_audit = None
            if options["repair_camera_notes"]:
                camera_panel, note_audit = repair_camera_metadata_notes(camera_panel, risk_epochs[index])
            canvas.paste(Image.fromarray(camera_panel).resize((960, 900), Image.Resampling.LANCZOS), (24, 84))
            draw.text((1016, 51), "LIDAR RISK | candidate current scores" if replay else "LIDAR OBSTACLE RISK | applied phi", font=fonts[20], fill="#cbd9e3")
            canvas.paste(sensor_map, (1016, 80))
            draw.text((1096, 522), "FOLLOW CAMERA | presentation view", font=fonts[20], fill="#cbd9e3")
            canvas.paste(Image.fromarray(rgb[:, :800]).resize((672, 504), Image.Resampling.LANCZOS), (1096, 552))
            draw.text((24, 1004), "SCORE REPLAY ONLY | Original V3 trajectory and prior geometry; no new navigation validation." if replay else status, font=fonts[20], fill="#e5c697")
            draw.text((24, 1038), "Fixed color scale. Human protection prior 0.80; thermal/potential priors retained. Lambda unchanged." if replay else "Same native run and all pauses. Thermal/potential priors remain explicit.", font=fonts[18], fill="#aebfce")
            result = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
            if index in previews:
                if not cv2.imwrite(str(output / f"preview-{index:04d}.png"), result):
                    raise RuntimeError("could not write preview")
            ledger.append(dict(output_frame=None if preview_only else native_count, original_source_frame=index,
                               source_time=epoch["time"], field_id=field["field_id"],
                               field_source_time=field.get("source_time"), field_commit_time=field.get("commit_time"),
                               geometry_file=name, robot_world_xy_yaw=field["robot_world_xy_yaw"], map=map_audit,
                               **({"camera_note_repair":note_audit} if note_audit is not None else {})))
            if writer is not None:
                writer.write(result)
                native_count += 1
        if writer is not None:
            credits = cv2.VideoCapture(str(archive / "restaurant-service-v3.mp4"))
            credits.set(cv2.CAP_PROP_POS_FRAMES, credits_layout["native_navigation_frames"])
            try:
                for _ in range(credits_layout["credit_frames"]):
                    ok, frame = credits.read()
                    if not ok:
                        raise RuntimeError("original attribution card missing")
                    canvas = np.full((1080, 1920, 3), (38, 27, 16), np.uint8)
                    canvas[81:999] = cv2.resize(frame, (1920, 918), interpolation=cv2.INTER_CUBIC)
                    writer.write(canvas)
                    credit_count += 1
            finally:
                credits.release()
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    decoded = 0
    if not preview_only:
        check = cv2.VideoCapture(str(video_path))
        try:
            while True:
                ok, frame = check.read()
                if not ok:
                    break
                if frame.shape != (1080, 1920, 3):
                    raise ValueError("wrong rendered frame dimensions")
                decoded += 1
        finally:
            check.release()
        if decoded != native_count + credit_count or native_count != layout["native_navigation_frames"]:
            raise ValueError("output frame count or duration differs")
    if any(digest(Path(path)) != value for path, value in hashes.items()):
        raise ValueError("presentation inputs changed")
    if replay:
        replay.verify()
        hashes.update(replay.hashes)
    if any(verify_archive(path) != count for path, count in protected.items()):
        raise ValueError("a protected demo archive changed")
    summary = dict(presentation_only=True, preview_only=preview_only, resolution=[1920, 1080], fps=10,
                   presentation_options=options, palette=palette_description(options["palette"]),
                   native_navigation_frames=native_count, credit_frames=credit_count,
                   fully_decoded_frames=decoded, removed_navigation_frames=0,
                   first_source_frame=first, last_source_frame=last, input_sha256=hashes,
                   mission_source=str(source.resolve()), risk_export=str(risk_manifest.resolve()),
                   sensor_only_retained_preview_source_frame=sensor_only_retained_preview,
                   protected_archives={str(k): v for k, v in protected.items()},
                   layout="left original perception; upper-right local sensor risk; lower-right native follow",
                   scope="Same native recording and controller values. Map colors use observation/geometry applied phi, not numerical h. Original camera uses individual phi. Latest scan age is explicit. Prior/retained geometry is hollow. No new inference, control, physics or acceptance.")
    if replay:
        summary.update(calibration_replay=str(calibration_replay.resolve()),
            palette=palette_description("restaurant_v1"), new_closed_loop_validation=False,
            calibration_config=replay.manifest["config"],
            layout="left original pixels/masks with candidate individual scores; right candidate current-point scores + original prior geometry/follow camera",
            scope="Candidate scoring on archived source-bound evidence. Original V3 path, timing, ownership and prior geometry are preserved; ordinary-object policy and external risk floors are unchanged. Not a counterfactual controller rollout or a new navigation pass.")
    (output / "presentation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "frame-mapping.json").write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(archive / "ASSET_CREDITS.txt", output / "ASSET_CREDITS.txt")
    (output / "README.txt").write_text(
        f"{options['title']} - LiDAR risk presentation\n\n"
        f"Open {options['video_name']}. Same native run, every navigation/stopping frame and original attribution card.\n"
        "Solid points are latest committed source returns; hollow points are prior/retained geometry. Colors use controller-applied phi, not numerical h.\n"
        "Grey points are outside the active controller field; its boundary is outlined. The fixed view clips distant observations.\n"
        "Retained CRA means current returns reuse retained properties; it is distinct from old occupancy. No current RGB mask is required.\n"
        "The bin/crate retained records reuse verified mass/contact with the current motion policy, including velocity priors when measurement is unavailable.\n"
        "The original camera shows individual CRA phi, which can differ from the applied/group policy in the map.\n"
        "The map uses recorded simulator localization and fixed north; no authored obstacle or human ground truth is added.\n"
        "This presentation does not improve perception or repair unavailable object association. All original validation limitations remain.\n"
        + status + "\n", encoding="utf-8")
    if replay:
        (output / "README.txt").write_text(
            "Restaurant CRA calibration - archived V3 score replay\n\n"
            "Open restaurant-risk-calibration-replay.mp4. Every original navigation/stopping frame and credit card is retained.\n"
            "New candidate scores were computed from archived evidence, not used to drive this trajectory. Fresh closed-loop validation remains required.\n"
            "Current returns use candidate applied scores; original external floors and hollow prior-geometry scores remain. Camera masks show candidate individual scores.\n"
            "Fixed restaurant_v1 palette; no per-frame or per-object color normalization. Human protection floor0.80 is a prior, not sharpness or measured injury probability.\n"
            "Potential/thermal priors, unavailable-speed priors, original ownership, evidence clocks and all failed views remain explicit.\n"
            "Original V3 and V4 archives are unchanged. This replay does not reconstruct the counterfactual memory/geometry lifecycle.\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(dict(files={str(path.relative_to(output)):
        dict(bytes=path.stat().st_size, sha256=digest(path)) for path in sorted(output.rglob("*")) if path.is_file()}), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "input_sha256"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "archive", "risk-manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--preserve", type=Path, action="append", default=[])
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--calibration-replay", type=Path)
    parser.add_argument("--palette", choices=("legacy", "restaurant_v1"), default="legacy")
    parser.add_argument("--title")
    parser.add_argument("--video-name")
    parser.add_argument("--repair-camera-notes", action="store_true")
    args = parser.parse_args()
    main(args.source, args.archive, args.risk_manifest, args.output, args.preview_only, args.preserve,
         args.calibration_replay, palette=args.palette, title=args.title, video_name=args.video_name,
         repair_camera_notes=args.repair_camera_notes)
