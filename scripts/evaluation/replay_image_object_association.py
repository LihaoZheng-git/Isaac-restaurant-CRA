"""Replay original RGB/scan sources through stationary group association only.

No fixture truth, scores, retained masks or synthetic returns enter association.
This does not run the final CRA/field commit and is not native acceptance.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robot.perception.camera_lidar_fusion import FThetaCameraCalibration, depth_visible_projection, project_fisheye_polynomial_points
from robot.perception.image_object_association import ImageObjectConfig, StationaryImageObjects, associate_image_objects
from robot.perception.sam_instance_segmentation import SamInstanceConfig, SamInstanceSegmenter
from robot.perception.sensor_epochs import gravity_aligned_robot_transform
from robot.safety.obstacle_tracking import ObstacleTracker, ObstacleTrackerConfig
from robot.perception.moving_view_association import ImageObjectView, MovingViewAssociationConfig


def replay(folder, model, *, moving_view=False, include_live=False):
    hashes = {}
    def checked(path):
        hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    records = json.loads(checked(folder/'stationary-fields.json').read_text())
    if include_live:
        records += [json.loads(line) for line in checked(folder/'live-fields.jsonl').read_text().splitlines() if line.strip()]
    calibration = FThetaCameraCalibration(**json.loads(checked(folder/'camera-calibration.json').read_text()))
    tracker = ObstacleTracker(ObstacleTrackerConfig(cluster_radius=.12, minimum_cluster_points=4,
        maximum_association_distance=.5, clustering_backend='kdtree'))
    states = {name: StationaryImageObjects(ImageObjectConfig(hold_overlapping_hypotheses=enabled))
              for name, enabled in (('legacy', False), ('overlap_hold', True))}
    if moving_view:
        states['moving_view'] = StationaryImageObjects(ImageObjectConfig(hold_overlapping_hypotheses=True))
    rows = []
    for record in records:
        frame = record['sensor_frame_id']
        source = folder/'sensor-snapshots'/f'{frame:06d}'
        scan = np.load(checked(source.with_suffix('.npz')))
        rgb = np.array(Image.open(checked(source.with_suffix('.png'))).convert('RGB'))
        transform = scan['world_from_robot'] @ np.linalg.inv(gravity_aligned_robot_transform(scan['world_from_robot']))
        yaw = float(np.arctan2(transform[1, 0], transform[0, 0]))
        xyz = scan['obstacle_xyz']
        world = xyz.astype(float).copy()
        c, s = np.cos(yaw), np.sin(yaw)
        world[:, :2] = world[:, :2] @ np.array([[c, s], [-s, c]]) + transform[:2, 3]
        world[:, 2] += transform[2, 3]
        camera_from_world = scan['camera_from_scan_robot'] @ np.linalg.inv(transform)
        view = ImageObjectView(camera_from_world, calibration, MovingViewAssociationConfig())
        tracking = tracker.update(xyz[:, :2], robot_position_world_xy=transform[:2, 3],
            robot_yaw=yaw, timestamp_seconds=record['sensor_timestamp'], clustering_points_robot_xyz=xyz)
        projection = depth_visible_projection(project_fisheye_polynomial_points(xyz,
            camera_from_source=scan['camera_from_scan_robot'], calibration=calibration), radius_pixels=4)
        model.set_image(rgb)
        proposals = model.image_proposals(tracking.point_track_ids, projection)
        candidates, audit = associate_image_objects(proposals, projection, tracking.point_track_ids, xyz, ImageObjectConfig())
        row = {'frame': frame, 'source_time': record['sensor_timestamp'], 'candidate_count': len(candidates),
               'original_accepted_ids': record['diagnostics']['image_object_groups']['accepted_object_ids'],
               'association_audit': audit}
        for name, state in states.items():
            staged = copy.deepcopy(candidates)
            accepted = state.update(staged, world, record['sensor_timestamp'], view=view if name=='moving_view' else None)
            row[name] = {'hypothesis_count': len(state.records),
                'candidate_audit': [a['diagnostic'] for a in staged],
                'accepted': [{'id': a['object_id'], 'points': int(a['selected'].sum()),
                    'centroid_world_xyz': world[a['selected']].mean(axis=0).tolist(),
                    'point_indices_sha256': hashlib.sha256(np.flatnonzero(a['selected']).tobytes()).hexdigest(),
                    'mask_sha256': hashlib.sha256(a['proposal'].detection.mask.tobytes()).hexdigest()}
                    for a in accepted]}
        rows.append(row)
        print(json.dumps({'source': str(folder), 'frame': frame,
            'accepted': {n: len(row[n]['accepted']) for n in states}}), flush=True)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
        raise RuntimeError('archived source changed')
    return {'scope': __doc__, 'source_hashes': hashes, 'rows': rows,
        'summary': {n: {'accepted_sources': sum(bool(r[n]['accepted']) for r in rows),
            'accepted_assessments': sum(len(r[n]['accepted']) for r in rows),
            'final_hypotheses': rows[-1][n]['hypothesis_count']} for n in states},
        'legacy_accepted_ids_match_original': all([a['id'] for a in r['legacy']['accepted']] == r['original_accepted_ids'] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--moving-view', action='store_true', help='also audit bounded reprojected partial-surface correspondence')
    parser.add_argument('--include-live', action='store_true', help='requires full source RGB/scan archives for every live field; never synthesize missing sources')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('fresh output required; preserve old replays')
    args.output.mkdir(parents=True)
    model = SamInstanceSegmenter(SamInstanceConfig(ROOT/'outputs/sam2_integration/sam2.1_t.pt',
        observed_context_box_retry=True, foreground_depth_gap_m=.5))
    reports = {}
    for index, folder in enumerate(args.source):
        result = replay(folder, model, moving_view=args.moving_view, include_live=args.include_live)
        name = f'{index}-{folder.name}'
        (args.output/f'{name}.json').write_text(json.dumps(result, indent=2)+'\n')
        reports[str(folder)] = {k: result[k] for k in ('summary', 'legacy_accepted_ids_match_original')}
    (args.output/'summary.json').write_text(json.dumps(reports, indent=2)+'\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
