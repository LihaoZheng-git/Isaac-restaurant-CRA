"""Separate v3 scene assembly from original licensed prop layers."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom

from robot.scene.restaurant_container_bench import CONTAINER_PATH
from robot.scene.restaurant_demo_build import DEMO_ROOT, demo_prop_records


SEPARATED_CONTACT_CENTERS = ((22.0, 4.65), (25.7, 4.9))
STRAIGHT_CONTACT_CENTERS = ((20.3, 4.85), (24.0, 5.1))
OPEN_AISLE_CONTACT_CENTERS = ((21.2, 3.2), (17.5, 3.0))
REMOVED_CONTACT_DEMO_PATHS = (
    DEMO_ROOT+'/ClearingStation', DEMO_ROOT+'/ClosedCaddy',
    DEMO_ROOT+'/UtensilCaddy', '/World/CRAServiceDressing/SecondPot',
    '/World/CRAServiceDemoV2Steam',
)


def build_service_v3(output, base_scene, bin_scene, crate_scene, *, separated_contact=False,
                     props_only=False, straight_contact=False, open_aisle_contact=False):
    if ((straight_contact or open_aisle_contact) and not separated_contact
            or straight_contact and open_aisle_contact):
        raise ValueError('choose one separated contact placement variant')
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
        if props_only:
            stage.GetPrimAtPath('/World/CRAServiceActors/Server').SetActive(False)
        stage.GetPrimAtPath(DEMO_ROOT+'/RoundedBin').SetActive(False)
        centers = SEPARATED_CONTACT_CENTERS if separated_contact else ((28.15,4.9), (28.15,6.1))
        if straight_contact:
            centers = STRAIGHT_CONTACT_CENTERS
        if open_aisle_contact:
            centers = OPEN_AISLE_CONTACT_CENTERS
        if separated_contact:
            for path in REMOVED_CONTACT_DEMO_PATHS:
                prim = stage.GetPrimAtPath(path)
                if not prim or not prim.IsActive():
                    raise ValueError(f'expected original active dressing: {path}')
                prim.SetActive(False)
        for scene, target, xy in ((bin_scene,CONTAINER_PATH,centers[0]),
                                  (crate_scene,DEMO_ROOT+'/BrokenCrate',centers[1])):
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
        separated_contact_layout=separated_contact,
        straight_contact_layout=straight_contact,
        open_aisle_contact_layout=open_aisle_contact,
        stationary_props_only=props_only,
        removed_paths=list(REMOVED_CONTACT_DEMO_PATHS) if separated_contact else [],
        contact_prop_centers_world_xy=centers,
        scope='Separate v3 authored arrangement; original meshes/materials/scales. No geometry, labels or scores supplied to perception. Human one-way timing is actor-only. Native acceptance pending.')
    output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
