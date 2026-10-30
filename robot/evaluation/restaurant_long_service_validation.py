"""Gate the distant-start mission without claiming a distant human was observed."""

import hashlib
import json
from pathlib import Path


def verify_near_human_gates(folders, root, scene):
    """Require current runtime/scene bytes and both independent human gates."""
    root, scene = Path(root), Path(scene).resolve()
    runtime = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for directory in ('robot', 'scripts') for p in (root/directory).rglob('*.py')
               if p.relative_to(root).as_posix() != 'scripts/evaluation/run_restaurant_demo.py'}
    cases = (('long-parallel-component', 'native/validation.json'),
             ('long-parallel-near', 'stationary-gate.json'))
    if len(folders) != 2:
        raise ValueError('two current-source human gates required')
    for folder, (case, filename) in zip(folders, cases):
        folder = Path(folder)
        meta = json.loads((folder/'invocation.json').read_text())
        execution = json.loads((folder/'execution.json').read_text())
        report = json.loads((folder/filename).read_text())
        previous = {p: h for p, h in meta['source_sha256'].items()
                    if p != 'scripts/evaluation/run_restaurant_demo.py'}
        scenes = {Path(p).resolve(): h for p, h in meta['scene_sha256'].items()}
        if (meta['case'] != case or not execution.get('passed')
            or not execution.get('source_unchanged') or report.get('passed') is not True
            or previous != runtime or scene not in scenes
            or any(hashlib.sha256(p.read_bytes()).hexdigest() != h for p, h in scenes.items())):
            raise ValueError('missing, failed or stale near-human gate')
        if case.endswith('-near'):
            if not (report.get('human_component_checks') and report.get('human_body_envelope_checks')
                    and all(report['human_component_checks'].values())
                    and all(report['human_body_envelope_checks'].values())):
                raise ValueError('near-view human ownership and body envelope must pass')
        else:
            from scripts.evaluation.run_restaurant_demo import component_reports_pass
            if not component_reports_pass(folder):
                raise ValueError('all native component reports must pass')
    return True
