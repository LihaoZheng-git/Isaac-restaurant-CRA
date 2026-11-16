"""Read original LiDAR ownership and applied policies for a presentation export.

This module never estimates physical factors, recomputes CRA, or controls a robot.
It supports the archived V3 no-bounds-retention recipe and fails on unsupported
ownership. Current samples and prior/remembered occupied geometry stay separate.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from robot.perception.camera_lidar_fusion import (
    FThetaCameraCalibration, depth_visible_projection,
    project_fisheye_polynomial_points,
)
from robot.perception.instance_lidar_geometry import masked_track_return_selection
from robot.perception.human_body_envelope import envelope_grid_points
from robot.perception.stationary_furniture_memory import raster_surface
from robot.safety.local_occupancy_grid import LocalOccupancyGridConfig


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def applied_policy_table(record):
    """The audit follows all baseline, human, fragment and retained-risk floors."""
    result = {}
    for row in record["diagnostics"]["applied_field_policies"]:
        key, risk = int(row["track_or_group_id"]), float(row["applied_field_risk"])
        if key in result or not math.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError("duplicate/invalid final applied policy")
        if not row["field_used_for_control"]:
            raise ValueError("archive does not use the Safe-SAGE field for control")
        result[key] = row
    return result


def static_selection(world, raw_ids, diagnostic):
    """Exact accepted static-location predicate; no re-estimation or new match."""
    proof = diagnostic["static_prop_association"]
    if proof["rejections"]:
        raise ValueError("rejected static evidence cannot color current returns")
    allowance = proof["position_allowance_m"]
    selected = np.all((world >= np.asarray(proof["observed_bounds_minimum"]) - allowance)
                      & (world <= np.asarray(proof["observed_bounds_maximum"]) + allowance), axis=1)
    selected &= np.isin(raw_ids, [f["raw_track_id"] for f in proof["fragments"]])
    if int(selected.sum()) != diagnostic["selected_return_count"]:
        raise ValueError("retained current-point count differs from original proof")
    for fragment in proof["fragments"]:
        own = raw_ids == fragment["raw_track_id"]
        if int((selected & own).sum()) != fragment["hits"]:
            raise ValueError("retained fragment membership differs from original proof")
    return selected


def assign_current_points(record, world, raw_ids, selection):
    """Replay only saved ownership, then look up unchanged final policy numbers.

    ``selection(obj)`` supplies the verified original mask-selection array.
    In the supported no-bounds recipe an unaccepted raw track keeps all its
    remaining points or none. Final recorded counts disambiguate those two
    cases after exact image/static overrides. This avoids using final human
    mass-dependent factors, which were updated AFTER ownership was resolved.
    """
    ids = np.array(raw_ids, dtype=np.int64, copy=True)
    basis = np.zeros(len(ids), dtype=np.uint8)
    policies = applied_policy_table(record)
    objects = record["objects"]
    zero_owner_generic = []
    for obj in objects:
        key, d = obj["track_id"], obj["diagnostics"]
        if d.get("image_object_association") or d.get("static_prop_policy"):
            continue
        own = raw_ids == key
        if d["accepted_instance"]:
            if d["policy_return_count"] == 0:
                # A later image/static override consumed every surviving
                # generic-mask return. Its hidden selection cannot affect the
                # final IDs: overridden points get their exact saved owner,
                # and every remainder is the existing auxiliary obstacle.
                # The calibrated presentation deliberately omits this mask.
                ids[own], basis[own] = key + 1_000_000_000, 0
                zero_owner_generic.append(obj)
                continue
            selected = selection(obj)
            if selected.shape != own.shape or np.any(selected & ~own):
                raise ValueError("generic mask ownership expanded beyond original raw track")
            ids[own & ~selected] = key + 1_000_000_000
            basis[selected] = 1
        elif any(d.get("property_evidence", {}).get(name, {}).get("mode") == "persistence_prior"
                 for name in ("mass", "contact")):
            basis[own] = 2
    for obj in objects:
        d, key = obj["diagnostics"], obj["track_id"]
        if d.get("image_object_association") and d["accepted_instance"]:
            selected = selection(obj)
            ids[selected], basis[selected] = key, 1
    for obj in objects:
        d, key = obj["diagnostics"], obj["track_id"]
        if d.get("static_prop_policy"):
            selected = static_selection(world, raw_ids, d)
            ids[selected], basis[selected] = key, 2
    ids[ids < 0] = 2_000_000_000
    for obj in zero_owner_generic:
        key, d = obj["track_id"], obj["diagnostics"]
        selected_count = d["selected_return_count"]
        covered = np.count_nonzero((raw_ids == key) & (ids != key + 1_000_000_000))
        if (not isinstance(selected_count, int) or isinstance(selected_count, bool)
                or selected_count < 0 or covered < selected_count):
            raise ValueError("zero-owner generic selection lacks recorded overriding coverage")
    for obj in objects:
        key, d = obj["track_id"], obj["diagnostics"]
        if d["accepted_instance"] or d.get("static_prop_policy") or d.get("image_object_association"):
            continue
        own = ids == key
        expected = d["policy_return_count"]
        if expected == 0 and own.any():
            ids[own], basis[own] = key + 1_000_000_000, 0
        elif expected != int(own.sum()):
            raise ValueError("unsupported partial ownership on unaccepted no-bounds track")
    for obj in objects:
        if np.count_nonzero(ids == obj["track_id"]) != obj["diagnostics"]["policy_return_count"]:
            raise ValueError(f"final ownership count mismatch for {obj['track_id']}")
    phi = np.empty(len(ids), dtype=np.float64)
    for key in np.unique(ids):
        if int(key) not in policies:
            raise ValueError(f"no original applied policy for current point ID {key}")
        phi[ids == key] = policies[int(key)]["applied_field_risk"]
    return ids, phi, basis


def admitted_video_fields(fields, epochs, risk_epochs):
    """Use the actual admitted camera field, including same-tick record/commit order."""
    if len(epochs) != len(risk_epochs):
        raise ValueError("native camera and video frame counts differ")
    by_id = {f["field_id"]: f for f in fields}
    if len(by_id) != len(fields):
        raise ValueError("duplicate committed field IDs")
    result = []
    for row, risk in zip(epochs, risk_epochs):
        if risk["time"] != row["time"]:
            raise ValueError("camera and overview video epochs differ")
        field = by_id.get(risk.get("sensor_frame_id"))
        if field is None and risk["status"] != "awaiting_assessment":
            raise ValueError("displayed assessment has no committed field")
        if field and (field["commit_time"] > row["time"]
                      or field["source_time"] != risk["sensor_timestamp_seconds"]):
            raise ValueError("displayed field source is mismatched or in the future")
        result.append(field)
    return result


def video_robot_poses(reader, epochs, stationary):
    """Use recorded ego poses only; independent obstacle truth is not read here."""
    records = reader.load_json("independent-clearance-observation.json")["records"]
    start = stationary[-1]["commit_simulation_time"]
    times = np.array([r["time_seconds"] + start for r in records])
    states = np.array([r["local_state"] for r in records])
    if np.any(np.diff(times) < 0):
        raise ValueError("recorded ego pose times must be ordered")
    captures = reader.load_json("captures.json")
    before = [r for r in captures if r["camera_time"] < times[0]]
    if before:
        times = np.r_[[r["camera_time"] for r in before], times]
        states = np.vstack(([r["state"] for r in before], states))
    yaw = np.unwrap(states[:, 2] + reader.anchor[2])
    poses = []
    for epoch in epochs:
        time = epoch["time"]
        position = np.array([np.interp(time, times, states[:, i]) for i in range(2)])
        world = position @ reader.rotation.T + reader.anchor[:2]
        if time >= start and np.linalg.norm(world - epoch["robot_world_xy"]) > 1e-5:
            raise ValueError("recorded ego poses do not align to video epochs")
        poses.append([*epoch["robot_world_xy"], float(np.interp(time, times, yaw))])
    return poses


class ArchiveReader:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.hashes = {}
        self.camera = FThetaCameraCalibration(**self.load_json("camera-calibration.json"))
        self.anchor = np.asarray(self.load_json("fixture-monitor.json")["anchor_world_xy_yaw"])
        c, s = math.cos(self.anchor[2]), math.sin(self.anchor[2])
        self.rotation = np.array([[c, -s], [s, c]])
        self.recent = {}
        self.geometry_cache = {}
        self.source_paths = {}
        invocation = self.load_json("invocation.json")
        repository = Path(__file__).resolve().parents[2]
        for name in ("robot/perception/camera_lidar_fusion.py",
                     "robot/perception/instance_lidar_geometry.py",
                     "robot/perception/stationary_furniture_memory.py",
                     "robot/perception/human_body_envelope.py"):
            if sha256(repository / name) != invocation["source_sha256"][name]:
                raise ValueError(f"replayed geometry implementation differs from recorded source: {name}")

    def remember(self, path):
        path = Path(path).resolve()
        path.relative_to(self.root)
        if path not in self.hashes:
            self.hashes[path] = sha256(path)
        return path

    def load_json(self, name):
        return read_json(self.remember(self.root / name))

    def load_npz(self, path, keys=None):
        with np.load(self.remember(path), allow_pickle=False) as archive:
            return {k: archive[k] for k in (archive.files if keys is None else keys)}

    def source(self, frame):
        if frame not in self.source_paths:
            raise ValueError("retained source has no original admitted archive reference")
        return self.load_npz(self.source_paths[frame], ("world_xyz", "raw_track_ids"))

    def current(self, record):
        d = record["diagnostics"]
        config = d["image_object_groups"]["static_prop_evidence"]["property_configuration"]
        if config["allow_instance_bounds_prior"] or d["occupancy_retention_scope"]:
            raise ValueError("this archive export does not support bounds/static occupancy retention")
        if d.get("carried_object_groups"):
            raise ValueError("carried-object ownership needs a separate exact archival adapter")
        candidate_path = Path(d["image_object_groups"]["candidate_archive"])
        source = self.load_npz(candidate_path, ("source_frame_id", "sensor_timestamp",
            "world_xyz", "robot_xyz", "raw_track_ids", "candidates_json"))
        if (int(source["source_frame_id"]) != record["sensor_frame_id"]
                or float(source["sensor_timestamp"]) != record["sensor_timestamp"]):
            raise ValueError("candidate archive source mismatch")
        self.source_paths[record["sensor_frame_id"]] = candidate_path
        world, raw_ids = source["world_xyz"], source["raw_track_ids"]
        sensor = self.load_npz(self.root / "sensor-snapshots" / f"{record['sensor_frame_id']:06d}.npz",
                               ("obstacle_xyz", "camera_from_scan_robot"))
        if not np.array_equal(sensor["obstacle_xyz"], source["robot_xyz"]):
            raise ValueError("sensor and image-group original points disagree")
        projection = None
        masks = None
        candidates = json.loads(str(source["candidates_json"]))

        def selection(obj):
            nonlocal projection, masks
            if projection is None:
                projection = depth_visible_projection(project_fisheye_polynomial_points(
                    source["robot_xyz"], camera_from_source=sensor["camera_from_scan_robot"],
                    calibration=self.camera), radius_pixels=4)
            key, detail = obj["track_id"], obj["diagnostics"]
            ids = raw_ids
            group = detail.get("image_object_association")
            if group:
                indices = [i for i, c in enumerate(candidates)
                           if c["proposal_index"] == group["proposal_index"]]
                if len(indices) != 1:
                    raise ValueError("missing/ambiguous original accepted group selection")
                index = indices[0]
                if "selected" not in source:
                    source.update(self.load_npz(candidate_path, ("selected", "masks")))
                ids = raw_ids.copy()
                ids[source["selected"][index]] = key
                mask = source["masks"][index]
            else:
                if masks is None:
                    timestamp = obj["attributes"]["timestamp_seconds"]
                    # Camera source IDs are original evidence clocks, not sensor IDs.
                    source_frame = next((m for m in self.camera_sources.values()
                                         if m["sensor_frame_id"] == record["sensor_frame_id"]), None)
                    if source_frame is None or source_frame["camera_timestamp_seconds"] != timestamp:
                        raise ValueError("original accepted generic camera mask is unavailable")
                    data = self.load_npz(self.root / "risk-camera-sources" / f"{source_frame['camera_frame_id']}.npz")
                    masks = {m["track_id"]: data["masks"][i] for i, m in enumerate(source_frame["masks"])
                             if m["basis"] != "retained evidence / LiDAR points"}
                if key not in masks:
                    raise ValueError(f"no original accepted generic mask for {key}")
                mask = masks[key]
            selected = masked_track_return_selection(ids, projection, mask, track_id=key)
            if int(selected.sum()) != detail["selected_return_count"]:
                raise ValueError(f"original mask/point selection count mismatch for {key}")
            return selected

        ids, phi, basis = assign_current_points(record, world, raw_ids, selection)
        return world[:, :2], ids, phi, basis

    def retained(self, record, policies, cfg, cells_world):
        d = record["diagnostics"]
        points, ids, risks = [], [], []

        def append(cloud, key):
            if key not in policies:
                raise ValueError(f"retained geometry has no applied policy {key}")
            cloud = np.asarray(cloud).reshape(-1, 2)
            points.append(cloud)
            ids.append(np.full(len(cloud), key, dtype=np.int64))
            risks.append(np.full(len(cloud), policies[key]["applied_field_risk"]))

        for epoch in d["retained_occupancy_epochs"]:
            if epoch["sensor_frame_id"] not in self.recent:
                raise ValueError("missing exact recent scan geometry/ownership")
            cloud, previous_ids = self.recent[epoch["sensor_frame_id"]]
            for key in np.unique(previous_ids):
                append(cloud[previous_ids == key], int(key))
        refs = {r["track_id"]: r for r in d["unknown_obstacle_geometry_archive"]["regions"]}
        for region in (d.get("unknown_obstacle_memory") or {}).get("regions", []):
            ref = refs[region["track_id"]]
            path = self.root / ref["path"]
            if path not in self.geometry_cache:
                archive = self.load_npz(path)
                cloud = archive["original_points_world_xy"]
                digest = hashlib.sha256(str(archive["metadata_json"]).encode()
                                        + np.ascontiguousarray(cloud, dtype='<f8').tobytes()).hexdigest()
                if digest != ref["geometry_sha256"] or self.hashes[path.resolve()] != ref["file_sha256"]:
                    raise ValueError("retained original geometry digest mismatch")
                self.geometry_cache[path] = cloud
            cloud = self.geometry_cache[path]
            if (len(cloud) != region["original_point_count"]
                    or ref["original_source_time"] != region["original_source_time"]):
                raise ValueError("retained geometry source mismatch")
            selected = cKDTree(cloud).query(cells_world)[0] <= region["radius_m"] + cfg.resolution / math.sqrt(2)
            append(cells_world[selected], 7_000_000_000 + region["track_id"])
        for region in (d.get("stationary_furniture_memory") or {}).get("retained", []):
            cache_key = (region["original_source_frame"], region["track_id"])
            if cache_key not in self.geometry_cache:
                original = self.source(region["original_source_frame"])
                self.geometry_cache[cache_key] = original["world_xyz"][original["raw_track_ids"] == region["track_id"], :2]
            cloud = self.geometry_cache[cache_key]
            if len(cloud) != region["original_point_count"]:
                raise ValueError("original furniture surface point count mismatch")
            cells, _ = raster_surface(cloud, region["uncertainty_radius_m"], cfg,
                                      self.anchor[:2], self.anchor[2])
            local = np.array([cfg.minimum_x, cfg.minimum_y]) + (cells + .5) * cfg.resolution
            append(local @ self.rotation.T + self.anchor[:2], 5_000_000_000 + region["track_id"])
        for region in d.get("human_body_envelope_priors", []):
            if region.get("applied_to_occupancy"):
                local, _ = envelope_grid_points([region], grid_config=cfg,
                    anchor_xy=self.anchor[:2], anchor_yaw=self.anchor[2])
                append(local @ self.rotation.T + self.anchor[:2], region["envelope_track_id"])
        for region in d.get("tracking_uncertainty_regions", []):
            selected = np.linalg.norm(cells_world - region["center_world_xy"], axis=1) <= region["radius_m"] + cfg.resolution / math.sqrt(2)
            if int(selected.sum()) != region["field_occupied_cells"]:
                raise ValueError("tracking uncertainty raster differs from original field")
            append(cells_world[selected], region["geometry_track_id"])
        if not points:
            return np.empty((0, 2)), np.empty(0), np.empty(0, np.int64)
        return np.concatenate(points), np.concatenate(risks), np.concatenate(ids)

    def export_field(self, record, output):
        world, ids, phi, basis = self.current(record)
        b = record["diagnostics"]["rolling_local_field"]["current_bounds"]
        cfg = LocalOccupancyGridConfig(**{k: v for k, v in b.items() if not k.startswith('_')})
        cols, rows = np.meshgrid(np.arange(cfg.column_count), np.arange(cfg.row_count))
        local_cells = np.column_stack((cols.ravel(), rows.ravel())) * cfg.resolution
        local_cells += np.array([cfg.minimum_x, cfg.minimum_y]) + .5 * cfg.resolution
        cells_world = local_cells @ self.rotation.T + self.anchor[:2]
        held, held_phi, held_ids = self.retained(record, applied_policy_table(record), cfg, cells_world)
        if len(world) + len(held) != record["diagnostics"]["field_raw_placement_count"]:
            raise ValueError("original field placement count differs; unsupported or missing geometry")
        local = (world - self.anchor[:2]) @ self.rotation
        inside = ((local[:, 0] >= cfg.minimum_x) & (local[:, 0] < cfg.maximum_x)
                  & (local[:, 1] >= cfg.minimum_y) & (local[:, 1] < cfg.maximum_y))
        boundary = np.array([[cfg.minimum_x, cfg.minimum_y], [cfg.maximum_x, cfg.minimum_y],
                             [cfg.maximum_x, cfg.maximum_y], [cfg.minimum_x, cfg.maximum_y]])
        name = f"fields/{record['sensor_frame_id']:06d}.npz"
        np.savez_compressed(output / name, points_world_xy=world, point_phi=phi,
            point_basis=basis, point_track_ids=ids, point_in_field=inside,
            retained_world_xy=held, retained_phi=held_phi, retained_track_ids=held_ids,
            field_bounds_world=boundary @ self.rotation.T + self.anchor[:2])
        # Recent occupancy is actual source samples, not expanded prior geometry.
        self.recent[record["sensor_frame_id"]] = (world, ids)
        keep = {e["sensor_frame_id"] for e in record["diagnostics"]["retained_occupancy_epochs"]}
        keep.add(record["sensor_frame_id"])
        self.recent = {k: v for k, v in self.recent.items() if k in keep}
        return dict(field_id=record["sensor_frame_id"], source_time=record["sensor_timestamp"],
                    commit_time=record["commit_simulation_time"], geometry_file=name,
                    current_count=len(world), retained_count=len(held), in_field_count=int(inside.sum()),
                    fresh_count=int((basis == 1).sum()), retained_evidence_count=int((basis == 2).sum()))


def export_archive(source, output):
    output = Path(output)
    if output.exists():
        raise ValueError("use a new output directory; preserve prior exports")
    reader = ArchiveReader(source)
    reader.camera_sources = {}
    for path in (reader.root / "risk-camera-sources").glob("*.json"):
        item = reader.load_json(path.relative_to(reader.root))
        reader.camera_sources[item["camera_frame_id"]] = item
    output.mkdir(parents=True)
    (output / "fields").mkdir()
    fields = []
    try:
        stationary = reader.load_json("stationary-fields.json")
        for record in stationary:
            fields.append(reader.export_field(record, output))
        live = reader.remember(reader.root / "live-fields.jsonl")
        with live.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream):
                fields.append(reader.export_field(json.loads(line), output))
                if number % 100 == 0:
                    print(f"Exported {len(fields)} committed fields", flush=True)
        if any(b["commit_time"] < a["commit_time"] for a, b in zip(fields, fields[1:])):
            raise ValueError("field commits are not ordered")
        epochs = reader.load_json("service-video-epochs.json")["frames"]
        risk_epochs = reader.load_json("risk-camera-epochs.json")["frames"]
        admitted = admitted_video_fields(fields, epochs, risk_epochs)
        poses = video_robot_poses(reader, epochs, stationary)
        frames = []
        for index, row in enumerate(epochs):
            field = admitted[index]
            frames.append(dict(source_frame=index, time=row["time"],
                **({k: field[k] for k in ("field_id", "source_time", "commit_time", "geometry_file")}
                   if field else dict(field_id=None, source_time=None, commit_time=None, geometry_file=None)),
                robot_world_xy_yaw=poses[index]))
        report = reader.load_json("report.json")
        route = np.array([[0., 0.], *report["nominal_route_local_xy"]])
        manifest = dict(schema_version=1, source_directory=str(reader.root), frames=frames,
            fields=fields, nominal_route_world=(route @ reader.rotation.T + reader.anchor[:2]).tolist(),
            point_basis={"0": "ordinary policy (may include inferred motion and priors)", "1": "fresh accepted visual assessment (may include priors)", "2": "retained mass/contact with current motion"},
            scope="Original applied per-ID controller phi on original current LiDAR returns. Prior/retained geometry separate. No new perception, scoring, physics or control.",
            robot_pose_scope="original recorded video XY; yaw interpolated from recorded ego localization (stationary capture poses before navigation), presentation only",
            source_hashes={str(k): v for k, v in reader.hashes.items()})
        for path, digest in reader.hashes.items():
            if sha256(path) != digest:
                raise ValueError(f"source changed during export: {path}")
        manifest["geometry_sha256"] = {f["geometry_file"]: sha256(output / f["geometry_file"]) for f in fields}
        manifest["source_hashes_verified_after_export"] = True
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + '\n')
        return manifest
    except Exception as error:
        (output / "FAILED.json").write_text(json.dumps(dict(error=str(error), completed_fields=len(fields)), indent=2) + '\n')
        raise
