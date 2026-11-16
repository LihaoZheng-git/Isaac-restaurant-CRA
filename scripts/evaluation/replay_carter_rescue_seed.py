"""Replay a trusted local numeric request; never execute robot commands.

Pickle requires the matching repository/environment and may execute code.
Only use task-generated archives after verifying their recorded SHA-256.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trusted-local-replay', action='store_true')
    parser.add_argument('--variant', choices=('tapered_seed', 'temporal_jacobian'), default='tapered_seed')
    args = parser.parse_args()
    if not args.trusted_local_replay:
        parser.error('explicit trusted-local-replay acknowledgement required; external pickle is unsupported')
    if args.output.exists():
        parser.error('output must be fresh; original archives remain read-only')
    directory = args.replay_directory.resolve()
    meta = json.loads((directory / 'first-no-progress-reference.json').read_text())
    path = (directory / meta['file']).resolve()
    if path.parent != directory or path.suffix != '.pickle':
        parser.error('archive metadata must name one local pickle in the same directory')
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != meta['sha256']:
        parser.error('original archive SHA-256 mismatch')
    rows = []
    for enabled in (False, True):
        saved = pickle.loads(blob)  # Explicit trusted local acknowledgement above.
        state, nominal, field, warm, goal, *epoch = saved['request']
        control = saved['control']
        original_config = asdict(control.mpc.config)
        option = ('enable_tapered_rescue_seed' if args.variant == 'tapered_seed'
            else 'use_consistent_temporal_jacobian')
        if original_config[option]:
            parser.error('selected option must be disabled in the recorded baseline')
        control.mpc.config = replace(control.mpc.config, **{option: enabled})
        config = asdict(control.mpc.config)
        assert all(config[k] == v for k, v in original_config.items() if k != option)
        before = pickle.dumps(field)
        result = control.plan(state, nominal, field, initial_command_guess=warm,
            **({'nominal_goal_grid_xy': goal} if goal is not None else {}),
            **({'prediction_state_timestamp_seconds': epoch[0]} if epoch else {}))
        if not enabled:
            np.testing.assert_allclose(result.command_sequence,
                saved['expected_result'].command_sequence, rtol=0, atol=1e-10)
        revalidated = False
        analytical = None
        if result.success:
            revalidated = control.mpc.revalidate_commands(state, result.command_sequence,
                configuration_space=field.pipeline.configuration_space, h_full=field.pipeline.h_full,
                guidance_fields=field.pipeline.guidance_fields,
                temporal_safety_derivative=field.temporal_safety_derivative,
                **({'predictive_geometry': field.predictive_human_geometry,
                    'predictive_source_age_seconds': epoch[0] - field.sensor_timestamp_seconds} if epoch else {}))
            filtered = control.filter_command(result.first_command, state, field)
            analytical = dict(safe_command=[filtered.safe_linear_velocity, filtered.safe_angular_velocity],
                margin=filtered.cbf_margin, emergency_stop=filtered.emergency_stop_active,
                first_command_unchanged=bool(np.allclose(result.first_command,
                    [filtered.safe_linear_velocity, filtered.safe_angular_velocity], atol=1e-9, rtol=0)))
        assert pickle.dumps(field) == before
        rows.append(dict(enabled=enabled, success=bool(result.success), status=result.status,
            first_command=list(result.first_command), command_sequence=result.command_sequence.tolist(),
            minimum_dtcbf_margin=float(result.minimum_dtcbf_margin) if result.success else None,
            full_exact_revalidation=bool(revalidated), analytical=analytical,
            seed_diagnostic=result.tapered_rescue_seed,
            source_frame_id=field.sensor_frame_id, source_timestamp_seconds=field.sensor_timestamp_seconds,
            unchanged_solver_budgets=dict(qp=config['maximum_solver_iterations'],
                nonlinear=config['maximum_nonlinear_rescue_iterations'])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(original_snapshot_sha256=meta['sha256'], variant=args.variant, rows=rows,
        scope='Offline regression only. Default full-command parity; unchanged field bytes/epochs/solver budgets. No wheel execution, native smoothness or real-time acceptance.'), indent=2, allow_nan=False))
    assert path.read_bytes() == blob
    print(json.dumps([{k: r[k] for k in ('enabled', 'success', 'first_command', 'full_exact_revalidation')}
        for r in rows], allow_nan=False))


if __name__ == '__main__':
    main()
