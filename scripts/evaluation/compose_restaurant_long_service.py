"""Present one continuous accepted native route; preserve every navigation pause."""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    p = args.source
    report = json.loads((p / "report.json").read_text())
    execution = json.loads((p / "execution.json").read_text())
    if report.get("native_human_mode") is not None:
        raise ValueError("this presentation recipe is route-only")
    if not (report["passed"] and execution["passed"] and execution["source_unchanged"]):
        raise ValueError("accepted unchanged-source run required")
    epochs = json.loads((p / "service-video-epochs.json").read_text())["frames"]
    first = next(
        i for i, r in enumerate(epochs) if np.linalg.norm(r["robot_command"]) > 1e-8
    )
    first = max(0, first - 1)
    route = np.array(report["nominal_route_policy"]["configured_anchors_local_xy"]) + [
        7.8,
        4.05,
    ]
    route = np.vstack(([7.8, 4.05], route))
    # Authored floor map is presentation/evaluation information only.
    mask = np.asarray(
        Image.open(
            "outputs/restaurant_service/layout-monitor-v1/authored-height-column-mask.png"
        ).convert("L")
    )
    image = np.full((*mask.shape, 3), (238, 242, 245), np.uint8)
    image[mask > 0] = (153, 164, 174)
    # Display only the service region; raster row zero is maximum world y.
    crop = Image.fromarray(
        image[
            round((32 - 19) / 0.025) : round((32 - 2) / 0.025),
            round((5 + 2) / 0.025) : round((34 + 2) / 0.025),
        ]
    )
    crop = crop.resize((390, 229))

    def pixel(q):
        return (860 + (q[0] - 5) / 29 * 390, 86 + (19 - q[1]) / 17 * 229)

    cap = cv2.VideoCapture(str(p / "service-navigation.mp4"))
    if (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(epochs)
        or cap.get(cv2.CAP_PROP_FPS) != 10
    ):
        raise ValueError("video and source epochs differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720)
    )
    if not writer.isOpened():
        raise RuntimeError("cannot open output video")

    def font(size):
        return ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", size)

    sources = []
    for line in (p / "live-fields.jsonl").open():
        r = json.loads(line)
        values = [o["cra_control"]["safe_sage"]["risk"] for o in r["objects"]]
        sources.append(
            (r["commit_simulation_time"], len(values), min(values), max(values))
        )
    index = -1
    path = []
    for i, r in enumerate(epochs):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("missing native frame")
        if i < first:
            continue
        while index + 1 < len(sources) and sources[index + 1][0] <= r["time"]:
            index += 1
        canvas = Image.new("RGB", (1280, 720), "#101d2b")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (24, 18),
            "Restaurant service | continuous native route",
            font=font(29),
            fill="white",
        )
        canvas.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (24, 85))
        canvas.paste(crop, (860, 86))
        draw = ImageDraw.Draw(canvas)
        draw.line([pixel(q) for q in route], fill="#ffffff", width=3)
        path.append(pixel(r["robot_world_xy"]))
        if len(path) > 1:
            draw.line(path, fill="#047fbe", width=4)
        x, y = path[-1]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#ffc54b")
        lines = [
            f"Simulation time: {r['time'] - epochs[first]['time']:.1f} s",
            "Native motion at 1x simulation time",
            "Waypoint nominal + online MPC/CBF",
            "Camera: presentation only",
        ]
        if index >= 0:
            _, n, low, high = sources[index]
            lines += [
                f"Latest committed LiDAR tracks: {n}",
                f"Computed track phi: {low:.3f} - {high:.3f}",
            ]
        lines += [
            "Missing factors use labeled priors",
            "No live temperature claim",
            "Route-only validation; no human here",
        ]
        for j, line in enumerate(lines):
            draw.text((860, 345 + 32 * j), line, font=font(18), fill="#d6e5ee")
        draw.text(
            (24, 690),
            "Paced simulation. No pauses removed. Blue: actual route; white: nominal.",
            font=font(17),
            fill="#adbfcd",
        )
        writer.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
    writer.release()
    cap.release()
    print(
        json.dumps(
            dict(
                frames=len(epochs) - first,
                fps=10,
                trimmed_initial_stationary_frames=first,
                removed_navigation_frames=0,
                output=str(args.output),
            )
        )
    )


if __name__ == "__main__":
    main()
