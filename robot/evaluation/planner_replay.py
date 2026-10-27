"""Opt-in snapshots of trusted, locally generated numeric planner inputs.

Pickle checkpoints are for offline diagnostics in the matching repository and
environment, never runtime evidence or a format for loading external files.
Keep the latest request in each five-second source-time bucket. Export work is
inside worker timing; it cannot make a late plan usable or renew source clocks.
"""

import hashlib
import json
import math
from pathlib import Path
import pickle

import numpy as np

from robot.safety.safe_sage_runtime import SafeSAGEControlRuntime


def save_planner_replay(directory, control, request, result):
    if not isinstance(control, SafeSAGEControlRuntime) or len(request) not in (5, 6):
        raise TypeError("planner replay requires the native runtime and five/six-part request")
    field = request[2]
    control._validate_field(field)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bucket = math.floor(field.sensor_timestamp_seconds / 5.)
    stem = f"planner-{bucket:06d}"
    payload = pickle.dumps(dict(format_version=1, control=control, request=request,
        expected_result=result), protocol=pickle.HIGHEST_PROTOCOL)
    path = directory / f"{stem}.pickle"
    temporary = directory / f"{stem}.pickle.tmp"
    temporary.write_bytes(payload)
    temporary.replace(path)
    nominal = np.asarray(request[1])
    metadata = dict(format_version=1, file=path.name,
        sha256=hashlib.sha256(payload).hexdigest(),
        sensor_frame_id=field.sensor_frame_id,
        sensor_timestamp_seconds=field.sensor_timestamp_seconds,
        prediction_state_timestamp_seconds=request[5] if len(request) == 6 else None,
        initial_chassis_state_grid=np.asarray(request[0]).tolist(),
        nominal_first_command=(nominal[0] if nominal.ndim == 2 else nominal).tolist(),
        nominal_goal_grid_xy=None if request[4] is None else np.asarray(request[4]).tolist(),
        result_success=bool(result.success), result_status=result.status,
        first_command=list(result.first_command),
        source_time_bucket_seconds=5.,
        scope="trusted local offline numeric replay only; original source epochs; no command execution or renewed evidence")
    temporary = directory / f"{stem}.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(directory / f"{stem}.json")
    # Keep the first actually requested passing-selection field too. The last
    # request in a five-second bucket may already be an unrecoverable state.
    # This diagnostic is never loaded by control and never changes source time.
    selected = getattr(field, "motion_passing_reference_track_ids", ())
    if selected and not (directory / "first-selected.pickle").exists():
        selected_metadata = dict(metadata, file="first-selected.pickle",
            snapshot_basis="first planner request with current positive passing selection",
            selected_track_ids=selected,
            selected_sides=getattr(field, "motion_passing_reference_sides", ()),
            original_selection_timestamp_seconds=getattr(field,
                "motion_passing_reference_source_timestamp_seconds", None))
        (directory / "first-selected.pickle").write_bytes(payload)
        (directory / "first-selected.json").write_text(
            json.dumps(selected_metadata, indent=2, allow_nan=False), encoding="utf-8")
    rear = getattr(result, 'rear_pass_reference', {})
    # Two bounded failure/continuation snapshots retain the exact first inputs
    # that a five-second bucket would otherwise overwrite. Export remains in
    # worker timing; these files never enter control or renew evidence clocks.
    special = []
    if rear.get('context', {}).get('mode') == 'rear' and rear.get('selected') is False:
        special.append('first-rear-unavailable')
    if rear.get('selection_tier') == 'checked_turn_first_fallback':
        special.append('first-rear-turn-fallback')
    for name in special:
        if (directory / f'{name}.pickle').exists():
            continue
        (directory / f'{name}.pickle').write_bytes(payload)
        special_metadata = dict(metadata, file=f'{name}.pickle',
            snapshot_basis=name, rear_reference_reason=rear.get('reason'),
            rear_reference_selection_tier=rear.get('selection_tier'))
        (directory / f'{name}.json').write_text(
            json.dumps(special_metadata, indent=2, allow_nan=False), encoding='utf-8')
    return metadata
