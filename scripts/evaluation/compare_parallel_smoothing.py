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
    activation_only_command_change, compare_parallel_smoothness, summarize_parallel_smoothness,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
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
    audit = activation_only_command_change(*(r['command'] for r in invocations))
    changed = subprocess.check_output(['git', 'diff', '--name-only',
        invocations[0]['source_revision'], invocations[1]['source_revision']], cwd=ROOT, text=True).splitlines()
    # This first track changes only an existing runtime parameter. New offline
    # evaluation, tests and documentation do not grant runtime behavior changes.
    allowed = {'AGENTS.md', 'docs/parallel_smoothing.md', 'docs/restaurant_behavior_validation.md',
        'robot/evaluation/parallel_smoothing.py', 'scripts/evaluation/compare_parallel_smoothing.py',
        'tests/unit/test_parallel_smoothing.py'}
    audit['runtime_source_unchanged'] = set(changed) <= allowed
    audit['changed_source_files'] = changed
    result = compare_parallel_smoothness(*metrics,
        matched_configuration=audit['matched'] and audit['runtime_source_unchanged'])
    result.update(configuration_audit=audit, original_report_sha256=hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in hashes.items())
    print(json.dumps(dict(passed=result['passed'], checks=result['checks'], output=str(args.output))))


if __name__ == '__main__':
    main()
