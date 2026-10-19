"""Prespecified bounded baseline matrix and failure-preserving result extraction."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from robot.evaluation.cra_experiments import PRIMARY_BASELINES


def comparison_plan(*, phase="pilot", repetitions=None):
    if phase not in {"pilot", "study"}:
        raise ValueError("phase must be pilot or study")
    repeats = (1 if phase == "pilot" else 5) if repetitions is None else repetitions
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 5:
        raise ValueError("repetitions must be 1..5")
    cases = [("static_aisle", None), ("standing_server", None), ("crossing_8s", 8.)]
    if phase == "study":
        cases += [("crossing_7s", 7.), ("crossing_9s", 9.)]
    trials = []
    for index, (case, delay) in enumerate(cases):
        for repeat in range(repeats):
            # Rotate order before collecting outcomes; no best-run selection.
            start = (index + repeat) % len(PRIMARY_BASELINES)
            order = PRIMARY_BASELINES[start:] + PRIMARY_BASELINES[:start]
            for baseline in order:
                trials.append({"case": case, "repeat": repeat,
                               "baseline": baseline.value, "crossing_delay_seconds": delay})
    return trials


def trial_options(trial):
    options = [
        "--persistent-object-risk-map", "--vectorized-fields", "--save-sensor-snapshots",
        "--plume-view-fov", "120", "--simulation-rate", ".05",
        "--qp-iteration-limit", "1000", "--nonlinear-rescue-iterations", "20",
        "--stream-live-records", "--immediate-replan", "--planner-result-max-age", ".5",
        "--maximum-seconds", "90",
    ]
    if trial["case"] == "static_aisle":
        options += ["--restaurant-scene", "scenes/generated/restaurant_service_v2.usda",
                    "--restaurant-route", "service", "--static-occupancy-memory-prior",
                    "--sam-observed-context-box", "--stationary-frames", "60", "--visual-reassessment-interval", "5",
                    "--live-visual-tracks", "1"]
    else:
        mode = "standing" if trial["case"] == "standing_server" else "crossing"
        options += ["--restaurant-scene", "scenes/generated/restaurant_service_isaac_server_v2.usda",
                    "--restaurant-route", "human-service", "--restaurant-server",
                    "--sam-observed-context-box",
                    "--restaurant-human", mode, "--human-camera-upward-degrees", "15",
                    "--human-body-contact-prior", ".10", "--stationary-frames", "60",
                    "--visual-reassessment-interval", ".2", "--live-visual-tracks", "2"]
        if mode == "crossing":
            options += ["--human-walking-speed", ".5", "--human-crossing-delay-seconds",
                        str(trial["crossing_delay_seconds"])]
    options += ["--restaurant-layout-survey", "outputs/restaurant_service/layout-monitor-v1/geometry-survey.json"]
    return options


def contract_fingerprint(options):
    return hashlib.sha256(json.dumps(options, separators=(",", ":")).encode()).hexdigest()


def _read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def extract_trial(directory, trial):
    """Never turn missing reports, prior scores, or clearance guards into passes."""
    directory = Path(directory)
    report = _read(directory / "report.json", {})
    gate = _read(directory / "stationary-gate.json", {})
    if not gate:
        gate = _read(directory / "stationary-report.json", {})
    commands = _read(directory / "commands.json", [])
    trajectory = np.asarray(_read(directory / "trajectory.json", []), dtype=float)
    modes, observed_classes, mass_statuses = Counter(), Counter(), Counter()
    accepted = owned = total = factor_supported = 0
    ages, risks, applied, build_seconds = [], [], [], []
    source = directory / "live-fields.jsonl"
    if source.is_file():
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                ages.append(record["commit_simulation_time"] - record["sensor_timestamp"])
                build_seconds.append(record["build_wall_seconds"])
                for obj in record.get("objects", []):
                    total += 1
                    diagnostic = obj["diagnostics"]
                    mass_statuses[obj["attributes"]["mass"]["mass_kg"]["status"]] += 1
                    accepted += bool(diagnostic["accepted_instance"])
                    owned += diagnostic["policy_return_count"] > 0
                    risks.append(obj["cra_control"]["safe_sage"]["risk"])
                    for factor, detail in diagnostic["factor_scores"].items():
                        modes[f"{factor}:{detail['mode']}"] += 1
                        factor_supported += detail["mode"] == "evidence_supported"
                for item in record["diagnostics"].get("applied_field_policies", []):
                    observed_classes[item["observed_class"]] += 1
                    applied.append(item["applied_field_risk"])
    dt = np.diff([c["time"] for c in commands], append=report.get("navigation_simulation_seconds", 0.))
    dt = np.maximum(dt, 0.)
    stopped = np.array([abs(c["v"]) < .01 and abs(c["omega"]) < .02 for c in commands])
    intervention = [abs(c["v"] - c.get("nominal_v", c["v"])) > 1e-6
                    or abs(c["omega"] - c.get("nominal_omega", c["omega"])) > 1e-6 for c in commands]
    reasons = [c.get("stop_reason") for c in commands]
    clearance = report.get("minimum_clearance_m")
    return {
        **trial, "directory": str(directory), "evidence_mode": "prior",
        "report_available": bool(report), "stationary_gate_passed": gate.get("passed"),
        "passed": report.get("passed", False), "arrived": report.get("completed", False),
        "reason": report.get("reason", "stationary_gate_failed" if gate.get("passed") is False else "missing_motion_report"),
        "motion_started": bool(commands or report),
        "minimum_signed_monitor_clearance_m": clearance,
        "monitor_overlap": None if clearance is None else clearance < 0,
        "collision_scope": "inflated independent geometry monitor; physical contact is not measured",
        "stop_drift_m": report.get("stop_drift_m"),
        "minimum_executed_cbf_margin": report.get("minimum_cbf_margin"),
        "cbf_margin_scope": "distance-squared circular CBF" if trial["baseline"] == "geometric_velocity_cbf" else "Safe-SAGE h_full CBF; magnitudes not comparable to geometric baseline",
        "navigation_simulation_seconds": report.get("navigation_simulation_seconds"),
        "navigation_wall_seconds": report.get("wall_seconds"),
        "path_length_m": float(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1).sum()) if len(trajectory) > 1 else None,
        "stopped_simulation_seconds": float(dt[stopped].sum()) if len(commands) else None,
        "intervention_sample_count": int(sum(intervention)), "command_sample_count": len(commands),
        "safety_stop_episodes": sum(bool(reason) and (i == 0 or reasons[i-1] != reason) for i, reason in enumerate(reasons)),
        "stop_reason_samples": dict(Counter(r for r in reasons if r)),
        "perception": {"object_assessments": total, "accepted_instance_assessments": accepted,
            "positively_owned_assessments": owned, "evidence_supported_factor_count": factor_supported,
            "factor_modes": dict(modes), "class_assignments": dict(observed_classes),
            "mass_statuses": dict(mass_statuses),
            "source_to_commit_maximum_simulation_seconds": max(ages, default=None),
            "field_build_wall_seconds_p50_p95": np.percentile(build_seconds, [50, 95]).tolist() if build_seconds else None,
            "cra_phi_range": [min(risks), max(risks)] if risks else None,
            "applied_field_risk_range": [min(applied), max(applied)] if applied else None},
        "configuration": {key: report.get(key) for key in (
            "experiment_baseline", "fixed_lambda", "nominal_route_local_xy", "scene_source_sha256",
            "simulation_rate_limit", "qp_iteration_limit", "nonlinear_rescue_iteration_limit",
            "planner_candidate_maximum_age_seconds", "human_crossing_delay_seconds",
            "human_body_contact_prior_config", "experimental_camera_mount")},
    }


def comparison_summary(rows):
    """Incomplete cells stay visible; complete failures remain in denominators."""
    cells = {}
    for row in rows:
        key = (row["case"], row["repeat"])
        cell = cells.setdefault(key, {})
        if row["baseline"] in cell:
            raise ValueError("duplicate baseline trial")
        cell[row["baseline"]] = row
    expected = {b.value for b in PRIMARY_BASELINES}
    matches = []
    common = ("fixed_lambda", "nominal_route_local_xy", "scene_source_sha256", "simulation_rate_limit",
              "qp_iteration_limit", "nonlinear_rescue_iteration_limit", "planner_candidate_maximum_age_seconds",
              "human_crossing_delay_seconds", "human_body_contact_prior_config", "experimental_camera_mount")
    for key, cell in cells.items():
        available = expected <= cell.keys() and all(r["report_available"] for r in cell.values())
        mismatch = []
        if available:
            first = next(iter(cell.values()))["configuration"]
            mismatch = [k for k in common if any(r["configuration"][k] != first[k] for r in cell.values())]
        matches.append({"case": key[0], "repeat": key[1], "all_motion_reports": available,
                        "matched_configuration": available and not mismatch, "mismatches": mismatch})
    summaries = []
    for case in dict.fromkeys(r["case"] for r in rows):
        for baseline in PRIMARY_BASELINES:
            selected = [r for r in rows if r["case"] == case and r["baseline"] == baseline.value]
            if not selected:
                continue
            summaries.append({"case": case, "baseline": baseline.value, "attempts": len(selected),
                              "passes": sum(r["passed"] for r in selected),
                              "arrivals": sum(r["arrived"] for r in selected),
                              "missing_reports": sum(not r["report_available"] for r in selected)})
    return {"scope": "bounded prior-assisted simulation comparison; descriptive repetitions; no real-time or superiority claim",
            "matched_cells": matches, "summary": summaries, "trials": rows}
