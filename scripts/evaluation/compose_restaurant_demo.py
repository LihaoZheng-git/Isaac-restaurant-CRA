"""Compose explicitly separated native trials; never hide interaction pauses."""

import argparse
import hashlib
import json
from pathlib import Path
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "C:/Windows/Fonts/segoeui.ttf"
BOLD = "C:/Windows/Fonts/segoeuib.ttf"
W, H, FPS = 1280, 720, 10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(draw, xy, value, size=22, color="white", bold=False):
    draw.text(
        xy, value, font=ImageFont.truetype(BOLD if bold else FONT, size), fill=color
    )


def panel(title, lines, picture=None):
    canvas = Image.new("RGB", (W, H), "#111c29")
    if picture is not None:
        image = (
            Image.fromarray(picture)
            if isinstance(picture, np.ndarray)
            else Image.open(picture).convert("RGB")
        )
        image.thumbnail((930, 600))
        canvas.paste(
            image, (15 + (930 - image.width) // 2, 85 + (600 - image.height) // 2)
        )
    draw = ImageDraw.Draw(canvas)
    text(draw, (24, 18), title, 29, bold=True)
    text(
        draw,
        (24, 58),
        "CRA + Safe-SAGE  |  Isaac Sim  |  First service-demo progress",
        17,
        "#a9c1d4",
    )
    y = 100
    for line in lines:
        for part in textwrap.wrap(line, 28):
            text(draw, (970, y), part, 19)
            y += 26
        y += 16
    return canvas


def human_fields(folder):
    rows = []
    with (folder / "live-fields.jsonl").open() as source:
        for line in source:
            r = json.loads(line)
            bodies = r["diagnostics"].get("human_body_envelope_priors", [])
            if len(bodies) != 1:
                continue
            body = bodies[0]
            obj = next(
                (o for o in r["objects"] if o["track_id"] == body["track_id"]), None
            )
            rows.append(
                dict(
                    time=r["commit_simulation_time"],
                    source=r["sensor_timestamp"],
                    phi=body["applied_cra_phi"],
                    track_phi=obj["cra_control"]["cra"]["phi"] if obj else None,
                    factors=obj["diagnostics"]["factor_scores"] if obj else None,
                )
            )
    return rows


def static_fields(folder):
    rows = []
    with (folder / "live-fields.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            objects = row.get("objects", [])
            values = [o["cra_control"]["cra"]["phi"] for o in objects]
            prior_count = sum(
                any(
                    f["mode"] == "configured_prior"
                    for f in o["diagnostics"]["factor_scores"].values()
                )
                for o in objects
            )
            rows.append(
                dict(
                    time=row["commit_simulation_time"],
                    count=len(objects),
                    minimum=min(values) if values else None,
                    maximum=max(values) if values else None,
                    prior_count=prior_count,
                )
            )
    return rows


def native_frames(chapter):
    folder = Path(chapter["source"])
    report = json.loads((folder / "report.json").read_text())
    execution = json.loads((folder / "execution.json").read_text())
    if not (
        report["passed"]
        and report["reason"] == "goal_reached"
        and execution["passed"]
        and execution["source_unchanged"]
    ):
        raise ValueError(
            "cannot present a failed navigation trial as a successful chapter"
        )
    human = chapter.get("actor", "human") == "human"
    truth = (
        json.loads((folder / "human-truth-evaluation-only.json").read_text())
        if human
        else json.loads((folder / "service-video-epochs.json").read_text())["frames"]
    )
    epochs = []
    last = -np.inf
    for r in truth:
        if r["time"] - last >= 0.1 - 1e-6:
            epochs.append(r["time"])
            last = r["time"]
    first = next(r["time"] for r in truth if np.linalg.norm(r["robot_command"]) > 1e-8)
    start = max(0, int(np.searchsorted(epochs, first)) - 1)
    cap = cv2.VideoCapture(
        str(folder / ("human-navigation.mp4" if human else "service-navigation.mp4"))
    )
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == FPS
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(epochs)
    fields = human_fields(folder) if human else static_fields(folder)
    index = -1
    try:
        for i, t in enumerate(epochs):
            ok, frame = cap.read()
            assert ok
            if i < start:
                continue
            while index + 1 < len(fields) and fields[index + 1]["time"] <= t:
                index += 1
            lines = ["Separate live trial", f"Simulation t = {t:.1f} s"]
            if index >= 0 and t - fields[index]["time"] <= 0.35 + 1e-6:
                field = fields[index]
                if not human:
                    lines += [
                        f"LiDAR tracks: {field['count']}",
                        f"Policy phi range: {field['minimum']:.3f} - {field['maximum']:.3f}",
                        f"Tracks using missing-factor priors: {field['prior_count']}",
                    ]
                else:
                    if field["track_phi"] is not None:
                        lines.append(f"Assessed track phi: {field['track_phi']:.3f}")
                    lines.append(f"Applied envelope phi: {field['phi']:.3f}")
                factors = field.get("factors")
                if factors:
                    for name in ["potential", "kinetic", "contact", "thermal"]:
                        item = factors[name]
                        mode = (
                            "prior"
                            if item["mode"] == "configured_prior"
                            else "cue / prior-assisted"
                        )
                        lines.append(f"{name.title()}: {item['score']:.3f} ({mode})")
            else:
                lines.append("No fresh displayed policy")
            lines += (
                ["Envelope retains its prior-risk floor."]
                if human
                else ["Measured cues + labeled priors"]
            )
            lines += ["Native simulation timing; paced computation."]
            yield panel(chapter["title"], lines, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--playlist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve existing videos")
    manifest_path = args.output.with_name(args.output.stem + "-manifest.json")
    if manifest_path.exists() or manifest_path.resolve() == args.playlist.resolve():
        raise ValueError("manifest must be new and distinct from the playlist")
    playlist = json.loads(args.playlist.read_text())
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)
    )
    assert writer.isOpened()
    count = 0
    records = []
    try:
        for chapter in playlist["chapters"]:
            start = count
            if chapter["kind"] == "native":
                frames = native_frames(chapter)
                names = [
                    "report.json",
                    "execution.json",
                    "invocation.json",
                    "live-fields.jsonl",
                ] + (
                    ["human-navigation.mp4", "human-truth-evaluation-only.json"]
                    if chapter.get("actor", "human") == "human"
                    else ["service-navigation.mp4", "service-video-epochs.json"]
                )
                files = [Path(chapter["source"]) / n for n in names]
            else:
                card = panel(chapter["title"], chapter["lines"], chapter.get("image"))
                frames = (card for _ in range(round(chapter["seconds"] * FPS)))
                files = [Path(chapter["image"])] if chapter.get("image") else []
            for frame in frames:
                writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
                count += 1
            records.append(
                dict(
                    title=chapter["title"],
                    kind=chapter["kind"],
                    start_seconds=start / FPS,
                    end_seconds=count / FPS,
                    source_files={str(p): sha(p) for p in files},
                )
            )
    finally:
        writer.release()
    check = cv2.VideoCapture(str(args.output))
    assert int(check.get(cv2.CAP_PROP_FRAME_COUNT)) == count
    for number in [0, count // 3, count // 2, count - 1]:
        check.set(cv2.CAP_PROP_POS_FRAMES, number)
        ok, frame = check.read()
        assert ok and frame.std() > 10
        cv2.imwrite(
            str(args.output.with_name(args.output.stem + f"-frame-{number}.png")), frame
        )
    check.release()
    manifest = dict(
        fps=FPS,
        frames=count,
        seconds=count / FPS,
        chapters=records,
        scope="Separate trials with explicit chapter cuts. Native 10fps interaction pauses retained. No continuous mission, real-time or sensor-only four-factor claim.",
        video_sha256=sha(args.output),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(video=str(args.output), seconds=count / FPS)))


if __name__ == "__main__":
    main()
