"""Separate v3 scene assembly from original licensed prop layers."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom

from robot.scene.restaurant_container_bench import CONTAINER_PATH
from robot.scene.restaurant_demo_build import DEMO_ROOT, demo_prop_records


def build_service_v3(output, base_scene, bin_scene, crate_scene):
    output = Path(output).resolve()
    additions = output.with_name(output.stem + '_props.usda')
    if any(p.exists() for p in (output, additions, output.with_suffix('.json'))):
        raise ValueError('fresh v3 output paths required')
    sources = {}
    for path in (base_scene, bin_scene, crate_scene):
        path = Path(path).resolve()
        manifest = json.loads(path.with_suffix('.json').read_text())
        for source in (path, path.with_suffix('.json'), Path(manifest['additions']),
                       *map(Path, manifest['source_sha256'])):
            sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    base = Usd.Stage.Open(str(base_scene))
    stage = Usd.Stage.CreateNew(str(output))
    layer = Sdf.Layer.CreateNew(str(additions))
    stage.GetRootLayer().subLayerPaths = [additions.name,
        os.path.relpath(Path(base_scene).resolve(),output.parent).replace('\\','/')]
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,'Z')
    stage.SetStartTimeCode(base.GetStartTimeCode())
    stage.SetEndTimeCode(base.GetEndTimeCode())
    stage.SetTimeCodesPerSecond(base.GetTimeCodesPerSecond())
    with Usd.EditContext(stage,layer):
        stage.GetPrimAtPath(DEMO_ROOT+'/RoundedBin').SetActive(False)
        for scene, target, xy in ((bin_scene,CONTAINER_PATH,(28.15,4.9)),
                                  (crate_scene,DEMO_ROOT+'/BrokenCrate',(28.15,6.1))):
            manifest=json.loads(Path(scene).with_suffix('.json').read_text())
            donor=Sdf.Layer.FindOrOpen(manifest['additions'])
            if Path(donor.realPath).parent != output.parent:
                raise ValueError('same-directory asset references required')
            Sdf.CreatePrimInLayer(layer,Sdf.Path(target).GetParentPath())
            if not Sdf.CopySpec(donor,CONTAINER_PATH,layer,target):
                raise RuntimeError('original asset copy failed')
            prim=stage.GetPrimAtPath(target)
            bounds=UsdGeom.BBoxCache(0,['default','render']).ComputeWorldBound(prim).ComputeAlignedRange()
            center=(np.array(bounds.GetMin())+np.array(bounds.GetMax()))/2
            attr=prim.GetAttribute('xformOp:translate')
            translation=np.array(attr.Get())
            translation[:2]+=np.array(xy)-center[:2]
            attr.Set(Gf.Vec3d(*translation))
        props=demo_prop_records(stage)
    layer.Save()
    stage.GetRootLayer().Save()
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in sources.items()):
        raise RuntimeError('original scene or asset changed')
    result=dict(scene=str(output),additions=str(additions),source_sha256=sources,
        props=props,source_files_unchanged=True,
        scope='Separate v3 authored arrangement; original meshes/materials/scales. No geometry, labels or scores supplied to perception. Human one-way timing is actor-only. Native acceptance pending.')
    output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
