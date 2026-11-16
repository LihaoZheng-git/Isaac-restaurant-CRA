"""Replay archived pre-encounter LiDAR sources through the furniture prior.

This diagnoses geometry retention only, not navigation or human recognition.
It excludes archived observed-person classes but does not reproduce raw person
masks or validated body-envelope exclusions. Keep archival source gaps explicit;
this module replay cannot establish runtime human exclusion or wheel acceptance.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robot.evaluation.restaurant_baselines import fixed_class_comparison_policy
from robot.perception.stationary_furniture_memory import StationaryFurnitureMemory
from robot.safety.obstacle_tracking import TrackedObstacle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve the previous audit")
    sources = {}
    for path in args.archive.glob("track-loss-*.json"):
        row = json.loads(path.read_text())
        for label, prefix in (("previous", "original"), ("current", "current")):
            time = row[prefix+"_source_time"]
            sources[time] = (path.with_name(path.stem+"-"+label+".npz"),
                             row[prefix+"_tracks"], row[prefix+"_field_policies"])
    memory = StationaryFurnitureMemory()
    base = fixed_class_comparison_policy().parameters_for("unknown")
    rows = []
    for now, (path, tracks, policies) in sorted(sources.items()):
        with np.load(path, allow_pickle=False) as archive:
            points, ids = archive["current_points_world_xy"], archive["current_point_track_ids"]
            robot = archive["packet_robot_position_world_xy"]
            frame = int(archive["packet_sensor_frame_id"])
        tracks = tuple(TrackedObstacle(**t) for t in tracks)
        values = {p["track_or_group_id"]: p for p in policies}
        guidance = {t.track_id: replace(base, safety_parameters=replace(base.safety_parameters,
            risk=values[t.track_id]["applied_field_risk"],
            boundary_flux_magnitude=-values[t.track_id]["boundary_flux"])) for t in tracks}
        excluded = [k for k, p in values.items() if p.get("observed_class") == "person"]
        snapshot, audit = memory.update(points, ids, tracks, guidance,
            source_time=now, source_frame=frame, robot_xy=robot, excluded_track_ids=excluded)
        rows.append(dict(time=now, frame=frame, enrolled=audit["enrolled_track_ids"],
                         retained=audit["retained"], observed_coverage=audit["observed_coverage"],
                         excluded=audit["excluded_track_ids"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(scope=__doc__, archived_sources=len(rows),
                  maximum_gap=max(np.diff(sorted(sources))), rows=rows,
                  final_retained=[r.track_id for r in snapshot.surfaces],
                  final_observed_coverage=[r.original.track_id for r in snapshot.observed_surfaces])
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "scope"}}))


if __name__ == "__main__":
    main()
