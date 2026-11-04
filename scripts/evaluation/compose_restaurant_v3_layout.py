"""Reformat the archived v3 recording; map data is presentation-only.

Large perception panel left; authored route map and native follow camera right.
No simulation, perception, control, score, source clock or acceptance changes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from robot.evaluation.parallel_walking_validation import anchored_nominal_route_world


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_archive(folder):
    manifest = read(folder/'manifest.json')
    files = manifest['files']
    items = files.items() if isinstance(files, dict) else ((r['file'], r) for r in files)
    for name, item in items:
        expected = item['sha256'] if isinstance(item, dict) else item
        if digest(folder/name) != expected:
            raise ValueError(f'original archive changed: {folder/name}')
    return len(manifest['files'])


def star(draw, center, radius, fill):
    points = []
    for i in range(10):
        angle = -math.pi/2+i*math.pi/5
        r = radius if i % 2 == 0 else radius*.45
        points.append((center[0]+r*math.cos(angle), center[1]+r*math.sin(angle)))
    draw.polygon(points, fill=fill)


def dashed(draw, points, fill, width=2):
    for start, end in zip(points, points[1:]):
        start, end = np.asarray(start), np.asarray(end)
        length = np.linalg.norm(end-start)
        if length == 0:
            continue
        for offset in np.arange(0., length, 10.):
            a = start+(end-start)*offset/length
            b = start+(end-start)*min(offset+6., length)/length
            draw.line([tuple(a), tuple(b)], fill=fill, width=width)


def main(source, archive, output, map_folder, preview_only):
    if output.exists():
        raise ValueError('use a fresh presentation output; preserve existing videos')
    protected = verify_archive(archive)
    protected_v1 = verify_archive(ROOT/'outputs/final demo/v1')
    old_layout = read(archive/'restaurant-service-v3.json')
    if old_layout['removed_navigation_frames'] != 0:
        raise ValueError('untrimmed navigation source required')
    report = read(archive/'validation/evaluation-correction/report.json')
    correction = read(archive/'validation/evaluation-correction/correction.json')
    if not report['passed'] or not correction['passed']:
        raise ValueError('documented v3 post-run evaluation required')
    epochs = read(source/'service-video-epochs.json')['frames']
    risk_epochs = read(source/'risk-camera-epochs.json')['frames']
    if [r['time'] for r in epochs] != [r['time'] for r in risk_epochs]:
        raise ValueError('camera source epochs differ')
    first, last = old_layout['first_source_frame'], old_layout['last_source_frame']
    if last != len(epochs)-1 or last-first+1 != old_layout['native_navigation_frames']:
        raise ValueError('original navigation frame mapping differs')
    anchor = read(source/'fixture-monitor.json')['anchor_world_xy_yaw']
    route = anchored_nominal_route_world(anchor, report['nominal_route_local_xy'])
    truth = read(source/'human-truth-evaluation-only.json')
    times = np.array([r['time'] for r in truth])
    human = np.array([r['actor_xyz'][:2] for r in truth])
    if not np.all(np.diff(times) > 0) or not times[0] <= epochs[first]['time'] <= epochs[last]['time'] <= times[-1]:
        raise ValueError('ordered matching actor presentation trace required')
    survey = read(map_folder/'geometry-survey.json')
    mask_path = map_folder/'authored-height-column-mask.png'
    if digest(mask_path) != survey['mask_sha256']:
        raise ValueError('v1 authored floor raster changed')
    sources = [source/'service-risk-camera.mp4', source/'service-video-epochs.json',
        source/'risk-camera-epochs.json', source/'human-truth-evaluation-only.json',
        source/'fixture-monitor.json', archive/'restaurant-service-v3.mp4',
        archive/'restaurant-service-v3.json', archive/'manifest.json', mask_path,
        map_folder/'geometry-survey.json']
    hashes = {str(p.resolve()): digest(p) for p in sources}
    # The same authored raster as v1, expanded north to include the one-way start.
    bounds = (5., 34.5, 2., 25.)
    xmin, _, _, ymax = survey['raster_limits_xmin_xmax_ymin_ymax']
    resolution = survey['raster_resolution_m']
    mask = np.asarray(Image.open(mask_path).convert('L'))
    x0, x1, y0, y1 = bounds
    cut = mask[round((ymax-y1)/resolution):round((ymax-y0)/resolution),
               round((x0-xmin)/resolution):round((x1-xmin)/resolution)]
    background = np.full((*cut.shape, 3), (238, 242, 245), np.uint8)
    background[cut > 0] = (153, 164, 174)
    map_width, map_height = 420, 328
    map_base = Image.fromarray(background).resize((map_width, map_height), Image.Resampling.LANCZOS)

    def pixel(q):
        return ((q[0]-x0)/(x1-x0)*map_width, (y1-q[1])/(y1-y0)*map_height)

    for q in np.vstack((route, human)):
        if not x0 <= q[0] <= x1 or not y0 <= q[1] <= y1:
            raise ValueError('route/actor outside the displayed map')
    map_draw = ImageDraw.Draw(map_base)
    dashed(map_draw, [pixel(q) for q in route], '#637485')
    start = pixel(route[0])
    map_draw.ellipse((start[0]-4, start[1]-4, start[0]+4, start[1]+4), outline='#146b45', width=2)
    star(map_draw, pixel(route[-1]), 7, '#168259')
    fonts = {size: ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', size) for size in (18, 20, 22, 30)}
    cap = cv2.VideoCapture(str(source/'service-risk-camera.mp4'))
    if (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(epochs)
            or cap.get(cv2.CAP_PROP_FPS) != 10
            or (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (1440, 600)):
        raise ValueError('native video shape/epochs differ')
    output.mkdir(parents=True)
    path = []
    human_path = []

    def render(index, frame):
        row = epochs[index]
        elapsed = max(0., row['time']-old_layout['navigation_start_simulation_time'])
        actor = [np.interp(row['time'], times, human[:, j]) for j in range(2)]
        canvas = Image.new('RGB', (1920, 1080), '#101b26')
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 14), 'Restaurant service | v3', font=fonts[30], fill='white')
        draw.text((1016, 20), f'Simulation time {elapsed:5.1f} s  |  1x playback', font=fonts[22], fill='#cbd9e3')
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Original annotated perception panel, including source age and risk legend.
        canvas.paste(Image.fromarray(rgb[:, 800:1440]).resize((960, 900), Image.Resampling.LANCZOS), (24, 84))
        draw.text((1016, 51), 'ROUTE MAP | recorded poses', font=fonts[20], fill='#cbd9e3')
        current_map = map_base.copy()
        md = ImageDraw.Draw(current_map)
        if len(path) > 1:
            md.line(path, fill='#0088c2', width=3)
            md.line(human_path, fill='#de8b24', width=2)
        x, y = pixel(row['robot_world_xy'])
        md.ellipse((x-5, y-5, x+5, y+5), fill='#00a2db', outline='white', width=2)
        x, y = pixel(actor)
        md.ellipse((x-4, y-4, x+4, y+4), fill='#e98c23', outline='white', width=1)
        canvas.paste(current_map, (1016, 80))
        legend_x = 1460
        for j, (color, label) in enumerate((('#0088c2', 'Robot / travelled path'),
                                          ('#b8c4cf', 'Nominal waypoint route'),
                                          ('#e98c23', 'Human / recorded path'),
                                          ('#30b880', 'Start / delivery goal'))):
            y = 91+j*36
            draw.line((legend_x, y+12, legend_x+26, y+12), fill=color, width=3)
            draw.text((legend_x+38, y), label, font=fonts[20], fill='#d4dee6')
        remaining = np.linalg.norm(np.array(row['robot_world_xy'])-route[-1])
        draw.text((legend_x, 253), f'Distance to goal: {remaining:.2f} m', font=fonts[22], fill='white')
        for j, label in enumerate(('Authored map + recorded poses.', 'Presentation only; not sensor input.',
                                   'CRA + priors; online MPC / CBF.', 'All navigation pauses retained.')):
            draw.text((legend_x, 295+j*26), label, font=fonts[18], fill='#aebfce')
        draw.text((1040, 412), 'FOLLOW CAMERA | presentation view', font=fonts[20], fill='#cbd9e3')
        canvas.paste(Image.fromarray(rgb[:, :800]).resize((832, 624), Image.Resampling.LANCZOS), (1040, 440))
        draw.text((24, 1004), 'Bin: retained CRA evidence  |  Crate: ordinary obstacle prior in this mission', font=fonts[20], fill='#e5c697')
        draw.text((24, 1038), 'Thermal response unvalidated. Paced simulation; original recording and scores retained.', font=fonts[18], fill='#aebfce')
        return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)

    previews = {first, 1030, 1270, last}
    writer = None if preview_only else cv2.VideoWriter(str(output/'restaurant-service-v3-map.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'), 10, (1920, 1080))
    if writer is not None and not writer.isOpened():
        raise RuntimeError('cannot open presentation output')
    native_count = 0
    ledger = []
    for i, row in enumerate(epochs):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'missing original frame {i}')
        if i < first:
            continue
        actor = [np.interp(row['time'], times, human[:, j]) for j in range(2)]
        path.append(pixel(row['robot_world_xy']))
        human_path.append(pixel(actor))
        if preview_only and i not in previews:
            continue
        result = render(i, frame)
        if i in previews:
            cv2.imwrite(str(output/f'preview-{i:04d}.png'), result)
        if writer is not None:
            writer.write(result)
            ledger.append(dict(output_frame=native_count, original_source_frame=i,
                               source_time=row['time'], robot_world_xy=row['robot_world_xy'],
                               human_presentation_world_xy=actor))
            native_count += 1
    cap.release()
    credit_count = 0
    if writer is not None:
        # Preserve the original six-second attribution card in the new layout.
        credits = cv2.VideoCapture(str(archive/'restaurant-service-v3.mp4'))
        credits.set(cv2.CAP_PROP_POS_FRAMES, old_layout['native_navigation_frames'])
        for _ in range(old_layout['credit_frames']):
            ok, frame = credits.read()
            if not ok:
                raise RuntimeError('original credit card missing')
            canvas = np.full((1080, 1920, 3), (38, 27, 16), np.uint8)
            scaled = cv2.resize(frame, (1920, 918), interpolation=cv2.INTER_CUBIC)
            canvas[81:999] = scaled
            writer.write(canvas)
            credit_count += 1
        credits.release()
        writer.release()
        check = cv2.VideoCapture(str(output/'restaurant-service-v3-map.mp4'))
        decoded = 0
        while True:
            ok, frame = check.read()
            if not ok:
                break
            if frame.shape != (1080, 1920, 3):
                raise ValueError('wrong rendered dimensions')
            decoded += 1
        check.release()
        if decoded != native_count+credit_count or native_count != old_layout['native_navigation_frames']:
            raise ValueError('output duration or source-frame loss')
    if any(digest(Path(p)) != value for p, value in hashes.items()):
        raise ValueError('presentation source changed')
    if verify_archive(archive) != protected or verify_archive(ROOT/'outputs/final demo/v1') != protected_v1:
        raise ValueError('protected archive changed')
    summary = dict(presentation_only=True, preview_only=preview_only, resolution=[1920, 1080], fps=10,
        native_navigation_frames=native_count, credit_frames=credit_count, removed_navigation_frames=0,
        first_source_frame=first, last_source_frame=last, original_v3_files_verified=protected,
        original_v1_files_verified=protected_v1, input_sha256=hashes,
        layout='left: original annotated perception panel; upper right: authored map with actual recorded paths; lower right: native follow camera',
        map_world_bounds_xmin_xmax_ymin_ymax=bounds,
        scope='Same v3 source recording, scores, held-frame ages and pauses. Authored map and interpolated human root are presentation only, never CRA/control input. Original native FAIL and separate corrected post-run audit remain unchanged.')
    (output/'presentation.json').write_text(json.dumps(summary, indent=2)+'\n')
    (output/'frame-mapping.json').write_text(json.dumps(ledger, indent=2)+'\n')
    shutil.copy2(archive/'ASSET_CREDITS.txt', output/'ASSET_CREDITS.txt')
    (output/'README.txt').write_text('V3 presentation with route map\n\nOpen restaurant-service-v3-map.mp4.\n'
        'Left: perception camera with original CRA overlays. Right top: route map; right bottom: follow camera.\n'
        'Same original navigation/stopping frames at 1x simulation playback, with original asset credits.\n'
        'This is a presentation-only re-layout. The original v3 archive and all validation limitations remain unchanged.\n'
        'Bin evidence is retained; crate uses ordinary obstacle policy in the full mission. Thermal response remains unvalidated.\n'
        'Original native report failed a legacy actor-region check; separate exact-data post-run correction passed.\n'
        'Original archive: '+str(archive.resolve())+'\n', encoding='utf-8')
    (output/'manifest.json').write_text(json.dumps(dict(files={str(p.relative_to(output)):
        dict(bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(output.rglob('*')) if p.is_file()}), indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'input_sha256'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--map-folder', type=Path, default=ROOT/'outputs/restaurant_service/layout-monitor-v1')
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    main(args.source, args.archive, args.output, args.map_folder, args.preview_only)
