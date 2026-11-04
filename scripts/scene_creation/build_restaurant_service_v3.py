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
    parser.add_argument('--separated-contact',action='store_true',
        help='separate bin/crate along the earlier aisle; remove unused pot/plume/clearing station')
    parser.add_argument('--props-only',action='store_true',
        help='separate stationary prop diagnostic layer, without the native server')
    parser.add_argument('--straight-contact',action='store_true',
        help='separate contact variant earlier on the straight aisle, with more route clearance')
    parser.add_argument('--open-aisle-contact',action='store_true',
        help='separate props in surveyed furniture gaps on the opposite side of the early aisle')
    args=parser.parse_args()
    result=build_service_v3(args.output,ROOT/'scenes/generated/restaurant_service_demo_v2d.usda',
        ROOT/'scenes/generated/restaurant_rounded_trash_simple_trash_can_v1.usda',
        ROOT/'scenes/generated/restaurant_rounded_trash_broken_crate_v1.usda',
        separated_contact=args.separated_contact or args.straight_contact or args.open_aisle_contact,
        props_only=args.props_only,straight_contact=args.straight_contact,
        open_aisle_contact=args.open_aisle_contact)
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))
