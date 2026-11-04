"""Build the separate restaurant v3 scene from existing licensed layers."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
dll=Path(sys.executable).parent/'Lib/site-packages/usd_exchange.libs'
if os.name=='nt' and dll.is_dir():
    _cookie=os.add_dll_directory(str(dll))
from robot.scene.restaurant_service_v3 import build_service_v3

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=build_service_v3(args.output,ROOT/'scenes/generated/restaurant_service_demo_v2d.usda',
        ROOT/'scenes/generated/restaurant_rounded_trash_simple_trash_can_v1.usda',
        ROOT/'scenes/generated/restaurant_rounded_trash_broken_crate_v1.usda')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))
