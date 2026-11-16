"""Reconstruct archived native fields to compare qualified-object boundary gains.

This is a bounded field/zero-temporal analytical diagnostic, not a trajectory
replay or navigation acceptance. It never supplies a command to the simulator.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robot.evaluation.boundary_response import (
    cold_mpc_probe, pipeline_difference, reconstruct_pipeline, static_analytical_probe,
)
from robot.evaluation.crate_progress_diagnosis import load


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True,
                        help="trusted locally produced planner metadata, never external pickle")
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--unit-flux", nargs="+", type=float, default=[1.7, 2.3, 2.9, 4.1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cold-mpc-frame", type=int)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.source.resolve()):
        raise ValueError("use a new diagnostic directory outside the native archive")
    args.output.mkdir(parents=True)
    manifest_path = args.export / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if Path(manifest["source_directory"]).resolve() != args.source.resolve():
        raise ValueError("export source mismatch")
    frames = set(args.frames)
    if len(frames) > 12 or not frames or len(args.unit_flux) > 6 or args.unit_flux[0] != 1.7:
        raise ValueError("bounded diagnostic requires baseline1.7 first, <=12fields and <=6variants")
    blob, metadata = load(args.snapshot)
    saved = pickle.loads(blob)  # Trusted repository-native, hash-verified checkpoint only.
    original_field = saved["request"][2]
    footprint = original_field.pipeline.configuration_space.footprint
    source_path = args.source / "live-fields.jsonl"
    source_hash = digest(source_path)
    if source_hash != manifest["source_hashes"][str(source_path.resolve())]:
        raise ValueError("native fields no longer match audited export")
    records = {}
    with source_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["sensor_frame_id"] in frames:
                records[row["sensor_frame_id"]] = row
    if records.keys() != frames:
        raise ValueError("requested exact field IDs missing")
    anchor = np.asarray(json.loads((args.source / "fixture-monitor.json").read_text())["anchor_world_xy_yaw"])
    commands = json.loads((args.source / "commands.json").read_text())
    print("Loaded exact source fields; rebuilding bounded variants", flush=True)
    output = dict(scope="archived placement/policy field sensitivity; zero-temporal analytical probes only; no wheel motion or full MPC replay",
        snapshot_baseline_parity=None, source_sha256=source_hash,
        source_directory=str(args.source.resolve()), records=[],
        source_hashes={str(manifest_path.resolve()): digest(manifest_path),
                       str(args.snapshot.resolve()): digest(args.snapshot)})
    # The native report stores the simulation epoch of navigation start.
    report = json.loads((args.source / "report.json").read_text())
    start_time = report.get("navigation_start_simulation_time")
    if start_time is None:
        # Captures are the authoritative chronological record; avoid inventing
        # a live-command alignment if this report schema lacks the start epoch.
        start_time = None
    for frame in sorted(frames):
        record = records[frame]
        geometry_name = f"fields/{frame:06d}.npz"
        path = args.export / geometry_name
        expected = manifest["geometry_sha256"][geometry_name]
        if digest(path) != expected:
            raise ValueError("selected exported geometry changed")
        output["source_hashes"][str(path.resolve())] = expected
        with np.load(path, allow_pickle=False) as data:
            geometry = {k: data[k] for k in data.files}
        state = record["state_at_commit"]
        if frame == original_field.sensor_frame_id:
            state = np.asarray(saved["request"][0]).tolist()
            native_nominal = np.asarray(saved["request"][1])
            nominal = (native_nominal[0] if native_nominal.ndim == 2 else native_nominal).tolist()
            command_scope = "exact saved planner request pose/nominal; temporal derivative set zero"
        elif start_time is not None:
            command = min(commands, key=lambda x: abs(x["time"] + start_time - record["commit_simulation_time"]))
            nominal = [command["nominal_v"], command["nominal_omega"]]
            command_scope = "nearest native nominal to recorded commit pose; temporal derivative zero"
        else:
            nominal = [.2, 0.]
            command_scope = "declared straight .2m/s,0rad/s probe at recorded commit pose; not replayed native nominal"
        row = dict(frame=frame, source_time=record["sensor_timestamp"], state=state,
                   nominal=nominal, command_scope=command_scope, variants=[])
        baseline = None
        for endpoint in args.unit_flux:
            pipeline, policies = reconstruct_pipeline(record, geometry, anchor, footprint, endpoint)
            if baseline is None:
                baseline = pipeline
                if frame == original_field.sensor_frame_id:
                    output["snapshot_baseline_parity"] = pipeline_difference(original_field.pipeline, baseline)
            entry = dict(unit_flux=endpoint, policies=policies,
                **pipeline_difference(baseline, pipeline),
                analytical=static_analytical_probe(saved["control"], original_field, pipeline, state, nominal),
                forcing_floor_fraction=float(np.mean(np.abs(pipeline.h_full.forcing_fields) <= 1.00001e-9)),
                maximum_forcing_magnitude=float(np.max(np.abs(pipeline.h_full.forcing_fields))))
            entry["synthetic_lateral_queries"] = [dict(offset_grid_y=dy,
                **static_analytical_probe(saved["control"], original_field, pipeline,
                    np.asarray(state) + [0., dy, 0.], nominal)) for dy in [-.1, -.2, -.3, -.4]]
            if frame == args.cold_mpc_frame and endpoint in (1.7, 2.9):
                entry["cold_mpc"] = [dict(offset_grid_y=dy,
                    **cold_mpc_probe(saved["control"], pipeline,
                        np.asarray(state) + [0., dy, 0.], nominal)) for dy in [0., -.3]]
            row["variants"].append(entry)
            np.savez_compressed(args.output / f"field-{frame}-{endpoint:.1f}.npz",
                h=pipeline.h_full.values, guidance=np.stack([r.vectors_robot_xy for r in pipeline.guidance_fields]),
                forcing=pipeline.h_full.forcing_fields)
            print(f"frame{frame} endpoint{endpoint}: delta_h={entry['h_max_abs_difference']:.6g}; command={entry['analytical']['command']}", flush=True)
        output["records"].append(row)
        (args.output / "results.json").write_text(json.dumps(output, indent=2, allow_nan=False))
    if digest(source_path) != source_hash:
        raise ValueError("native fields changed during diagnostic")
    output["source_unchanged"] = True
    (args.output / "results.json").write_text(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
