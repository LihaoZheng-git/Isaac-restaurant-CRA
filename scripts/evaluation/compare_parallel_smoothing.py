"""Compare two completed native parallel runs without modifying either result."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robot.evaluation.parallel_smoothing import (  # noqa: E402
    activation_only_command_change, compare_parallel_smoothness, outward_only_command_change,
    outward_identity_command_change, summarize_parallel_smoothness, translation_only_command_change,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variation', choices=('activation', 'outward', 'outward_identity', 'translation'), default='activation')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('comparison output must be fresh; original trial files are read-only')
    metrics, invocations, hashes = [], [], {}
    for path in (args.baseline, args.candidate):
        report_path = path/'report.json'
        blob = report_path.read_bytes()
        metrics.append(summarize_parallel_smoothness(json.loads(blob),
            json.loads((path/'human-truth-evaluation-only.json').read_text())))
        invocations.append(json.loads((path/'invocation.json').read_text()))
        hashes[str(report_path.resolve())] = hashlib.sha256(blob).hexdigest()
    compare_commands = {'activation': activation_only_command_change,
        'outward': outward_only_command_change, 'outward_identity': outward_identity_command_change,
        'translation': translation_only_command_change}[args.variation]
    audit = compare_commands(*(r['command'] for r in invocations))
    changed = subprocess.check_output(['git', 'diff', '--name-only',
        invocations[0]['source_revision'], invocations[1]['source_revision']], cwd=ROOT, text=True).splitlines()
    # Activation changes no runtime source. The outward variant explicitly
    # permits its tested reference/CLI change; this is a filename scope audit,
    # not independent semantic proof of controller equivalence.
    allowed = {'AGENTS.md', 'docs/parallel_smoothing.md', 'docs/restaurant_behavior_validation.md',
        'robot/evaluation/parallel_smoothing.py', 'scripts/evaluation/compare_parallel_smoothing.py',
        'tests/unit/test_parallel_smoothing.py', 'tests/unit/test_parallel_outward_reference.py',
        'tests/unit/test_parallel_translation_reference.py'}
    reference_files = {'robot/safety/laplace_guidance_tracking_mpc.py',
        'scripts/demos/cra_live_object_isaac_demo.py'}
    identity_files = {'robot/evaluation/parallel_walking_validation.py',
        'robot/evaluation/restaurant_human_sensor.py', 'scripts/demos/restaurant_human_locomotion_isaac.py',
        'tests/unit/test_parallel_identity_smoothing.py'}
    audit['runtime_source_unchanged'] = set(changed) <= allowed
    audit['declared_source_scope'] = set(changed) <= (allowed | reference_files
        | (identity_files if args.variation in {'outward_identity', 'translation'} else set())
        if args.variation != 'activation' else allowed)
    audit['variation'] = args.variation
    audit['changed_source_files'] = changed
    result = compare_parallel_smoothness(*metrics,
        matched_configuration=audit['matched'] and audit['declared_source_scope'])
    result.update(configuration_audit=audit, original_report_sha256=hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in hashes.items())
    print(json.dumps(dict(passed=result['passed'], checks=result['checks'], output=str(args.output))))


if __name__ == '__main__':
    main()
