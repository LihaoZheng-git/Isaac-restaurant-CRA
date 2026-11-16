"""Rebuild the authored-geometry abort raster; no sensing, CRA or simulation.

Original USD faces are projected through a declared height column. This is
independent evaluation information, never an input to the navigation algorithm.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
_directory = Path(sys.executable).parent / "Lib/site-packages/usd_exchange.libs"
if os.name == "nt" and _directory.is_dir():
    _dll_cookie = os.add_dll_directory(str(_directory))
from pxr import Usd, UsdGeom


def clip_height(polygon, height, above):
    if not len(polygon):
        return []
    result=[]
    previous=polygon[-1]
    previous_in=previous[2]>=height if above else previous[2]<=height
    for point in polygon:
        inside=point[2]>=height if above else point[2]<=height
        if inside != previous_in:
            fraction=(height-previous[2])/(point[2]-previous[2])
            result.append(previous+fraction*(point-previous))
        if inside:
            result.append(point)
        previous,previous_in=point,inside
    return result


def export_survey(scene, robot_scene, output):
    scene,robot_scene,output=Path(scene).resolve(),Path(robot_scene).resolve(),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    sources={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (scene,robot_scene)}
    stage=Usd.Stage.Open(str(scene))
    if UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError("survey requires a metre scene")
    # These bounds contain the surveyed restaurant, including exterior walls.
    limits=(-2.,42.,-2.,32.)
    resolution=.025
    image=Image.new("L",(1760,1360),0)
    draw=ImageDraw.Draw(image)
    cache=UsdGeom.XformCache()
    faces_drawn=meshes_drawn=0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh=UsdGeom.Mesh(prim)
        if mesh.ComputeVisibility()==UsdGeom.Tokens.invisible:
            continue
        points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
        if not len(points):
            continue
        world=(np.column_stack((points,np.ones(len(points)))) @ np.array(cache.GetLocalToWorldTransform(prim)))[:,:3]
        if world[:,2].max()<.1 or world[:,2].min()>1.8:
            continue
        if world[:,0].min()<limits[0] or world[:,0].max()>limits[1] or world[:,1].min()<limits[2] or world[:,1].max()>limits[3]:
            raise ValueError("visible mesh extends outside the explicit layout domain")
        counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        holes=set(mesh.GetHoleIndicesAttr().Get() or [])
        if np.all(counts==3) and not holes:
            triangles=world[indices].reshape(-1,3,3)
            faces=triangles[(triangles[:,:,2].max(1)>=.1)&(triangles[:,:,2].min(1)<=1.8)]
        else:
            faces=[]
            offset=0
            for index,count in enumerate(counts):
                face=world[indices[offset:offset+count]]
                offset+=count
                if index not in holes and face[:,2].max()>=.1 and face[:,2].min()<=1.8:
                    faces.append(face)
        for face in faces:
            polygon=clip_height(clip_height(face,.1,True),1.8,False)
            if len(polygon)>=3:
                draw.polygon([((v[0]-limits[0])/resolution,(limits[3]-v[1])/resolution) for v in polygon],fill=255)
                faces_drawn+=1
        meshes_drawn+=1
    mask=output/"authored-height-column-mask.png"
    image.save(mask)
    assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha for path,sha in sources.items())
    report={"basis":"authored geometry / oracle abort survey; NOT RGB/LiDAR evidence or navigation acceptance",
            "source_sha256":sources,"source_files_unchanged":True,"units_m_per_unit":1.,
            "vertical_projection_slab_m":[.1,1.8],"raster_limits_xmin_xmax_ymin_ymax":limits,
            "raster_resolution_m":resolution,"mask_sha256":hashlib.sha256(mask.read_bytes()).hexdigest(),
            "projected_faces":faces_drawn,"projected_meshes":meshes_drawn}
    (output/"geometry-survey.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene",type=Path,default=ROOT/"scenes/generated/restaurant_base.usda")
    parser.add_argument("--robot-scene",type=Path,default=ROOT/"scenes/generated/restaurant_robot.usda")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(export_survey(args.scene,args.robot_scene,args.output),indent=2))


if __name__ == "__main__":
    main()
