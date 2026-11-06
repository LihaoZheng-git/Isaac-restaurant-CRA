"""Author the original crate fixture or a separate aisle-placement layer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
dll = Path(sys.executable).parent / 'Lib/site-packages/usd_exchange.libs'
if os.name == 'nt' and dll.is_dir():
    _cookie = os.add_dll_directory(str(dll))

from pxr import Gf, Sdf, Usd, UsdGeom

from robot.scene.restaurant_demo_build import DEMO_ROOT, demo_prop_records


def build(*, setback_cm=0):
    base = ROOT / 'scenes/generated/restaurant_service_demo_v3_open_aisle_bench.usda'
    output = base.with_name('restaurant_crate_route_probe.usda')
    if setback_cm == 15:
        base = output
        output = base.with_name('restaurant_crate_route_setback15.usda')
    elif setback_cm != 0:
        raise ValueError('unsupported crate placement')
    additions = output.with_name(output.stem + '_props.usda')
    if any(p.exists() for p in (output, additions, output.with_suffix('.json'))):
        raise ValueError('new output paths required; preserve every previous scene')
    manifest = json.loads(base.with_suffix('.json').read_text())
    paths = [base, base.with_suffix('.json'), Path(manifest['additions']),
             *map(Path, manifest['source_sha256'])]
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if any(sources[p] != h for p, h in manifest['source_sha256'].items()):
        raise ValueError('source differs from its scene manifest')
    base_stage = Usd.Stage.Open(str(base))
    stage = Usd.Stage.CreateNew(str(output))
    layer = Sdf.Layer.CreateNew(str(additions))
    stage.GetRootLayer().subLayerPaths = [additions.name, base.name]
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    stage.SetStartTimeCode(base_stage.GetStartTimeCode())
    stage.SetEndTimeCode(base_stage.GetEndTimeCode())
    stage.SetTimeCodesPerSecond(base_stage.GetTimeCodesPerSecond())
    with Usd.EditContext(stage, layer):
        if not setback_cm:
            stage.GetPrimAtPath(DEMO_ROOT + '/ServiceContainer').SetActive(False)
        prim = stage.GetPrimAtPath(DEMO_ROOT + '/BrokenCrate')
        attr = prim.GetAttribute('xformOp:translate')
        position = attr.Get()
        shift_y = -setback_cm / 100 if setback_cm else .45
        attr.Set(Gf.Vec3d(position[0], position[1] + shift_y, position[2]))
        props = demo_prop_records(stage)
    crate, = (prop for prop in props if prop['name'] == 'BrokenCrate')
    center = [(lo + hi) / 2 for lo, hi in zip(crate['bounds_minimum'][:2], crate['bounds_maximum'][:2])]
    if max(abs(center[0] - 17.5), abs(center[1] - (3.45 - setback_cm / 100))) > 1e-5:
        raise ValueError('composed crate bounds differ from the intended placement')
    layer.Save()
    stage.GetRootLayer().Save()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in sources.items())
    result = dict(scene=str(output), additions=str(additions), source_sha256=sources,
                  props=props, source_files_unchanged=True,
                  crate_center_world_xy=center, robot_start_world_xy=[15.65, 4.05],
                  nominal_goal_local_xy=[4., 0.],
                  scope='Crate-only placement and straight route. Authored geometry for scene/evaluation only.')
    output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setback-cm', type=int, choices=(0, 15), default=0,
                        help='Distance away from the route, measured from the original closer-crate fixture.')
    build(setback_cm=parser.parse_args().setback_cm)
