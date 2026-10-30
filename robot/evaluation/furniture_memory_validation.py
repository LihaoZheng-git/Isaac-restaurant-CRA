"""Evaluation-only human truth audit; never consumed by memory or control."""
import numpy as np


def audit_snapshot(snapshot, truth):
    if snapshot is None:
        return None
    time = snapshot.source_time
    surfaces = [r.points for r in snapshot.surfaces] + [r.points for r in snapshot.observed_surfaces]
    if not surfaces:
        return dict(passed=True, source_time=time, retained_surfaces=0,
                    minimum_retained_surface_human_distance_m=None,
                    scope="empty retained geometry; no human overlap claim is needed")
    times = np.asarray([r["time"] for r in truth])
    i = int(np.searchsorted(times, time))
    if not len(times) or time < times[0] - 1e-8 or time > times[-1] + 1e-8:
        return {"passed": False, "reason": "missing_source_aligned_truth"}
    i = min(i, len(times)-1)
    j = max(0, i-1)
    alpha = 0. if times[i] == times[j] else (time-times[j])/(times[i]-times[j])
    center = (1-alpha)*np.array(truth[j]["actor_xyz"][:2]) + alpha*np.array(truth[i]["actor_xyz"][:2])
    distances = [float(np.linalg.norm(points-center, axis=1).min()) for points in surfaces]
    return dict(passed=bool(not distances or min(distances) > .4),
                source_time=time, retained_surfaces=len(snapshot.surfaces),
                observed_surface_proofs=len(snapshot.observed_surfaces),
                minimum_retained_surface_human_distance_m=min(distances) if distances else None,
                human_reference_radius_m=.4,
                scope="independent current native human root audit; raw retained and observed-coverage points outside .4m; not control input or a complete mesh test")


def summarize_memory(rows):
    audits = [r.get("furniture_memory_human_audit") for r in rows]
    diagnostics = [r.get("stationary_furniture_memory") for r in rows]
    checks = dict(nonempty_records=bool(rows),
                  all_source_aligned_human_audits_pass=all(a and a.get("passed") for a in audits),
                  repeated_surface_enrollment_exercised=any(d and d["enrolled_track_ids"] for d in diagnostics),
                  human_exclusion_exercised=any(d and d["exclusion_point_count"] for d in diagnostics))
    return dict(passed=all(checks.values()), checks=checks, sources=len(rows),
                retained_sources=sum(bool(d and d["retained_count"]) for d in diagnostics),
                scope="stationary robot/native walking human memory exclusion; not wheel arrival or general static classification")
