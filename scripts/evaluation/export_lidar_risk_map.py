"""Export original V3 LiDAR policy colors without rerunning CRA or simulation."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from robot.evaluation.lidar_risk_archive import export_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export_archive(args.source, args.output)
    print(f"Exported {len(result['fields'])} original fields / {len(result['frames'])} video frames")


if __name__ == '__main__':
    main()
