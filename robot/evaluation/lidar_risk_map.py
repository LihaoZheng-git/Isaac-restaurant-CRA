"""Render archived controller-applied point risk; never infer or change a score."""

import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robot.evaluation.risk_camera_overlay import risk_color_rgb


@lru_cache(maxsize=16)
def _font(size):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", size)
    except OSError:
        return ImageFont.load_default()


def map_pixels(points, center, rectangle, half_span_m=4.0):
    """North-up equal-scale coordinates; half_span controls the vertical extent."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    center = np.asarray(center, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(points).all():
        raise ValueError("finite world coordinates required")
    if not math.isfinite(half_span_m) or half_span_m <= 0:
        raise ValueError("positive finite map extent required")
    x, y, width, height = rectangle
    if width <= 0 or height <= 0:
        raise ValueError("positive viewport dimensions required")
    scale = height / (2 * half_span_m)
    pixels = (points - center) * (scale, -scale) + (x + width / 2, y + height / 2)
    visible = ((pixels[:, 0] >= x) & (pixels[:, 0] < x + width)
               & (pixels[:, 1] >= y) & (pixels[:, 1] < y + height))
    return pixels, visible


def validate_geometry(geometry):
    """Reject invalid presentation inputs instead of substituting a risk value."""
    arrays = {}
    for prefix, key in (("point", "points_world_xy"), ("retained", "retained_world_xy")):
        points = np.asarray(geometry[key], dtype=float)
        phi = np.asarray(geometry[f"{prefix}_phi"], dtype=float)
        ids = np.asarray(geometry[f"{prefix}_track_ids"])
        if points.ndim != 2 or points.shape[1] != 2 or phi.shape != (len(points),) or ids.shape != phi.shape:
            raise ValueError(f"inconsistent {prefix} geometry arrays")
        if not np.isfinite(points).all() or not np.isfinite(phi).all() or ((phi < 0) | (phi > 1)).any():
            raise ValueError(f"invalid {prefix} coordinates or applied phi")
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("integer track identifiers required")
        arrays[key], arrays[f"{prefix}_phi"], arrays[f"{prefix}_track_ids"] = points, phi, ids
    basis = np.asarray(geometry["point_basis"])
    if basis.shape != arrays["point_phi"].shape or not np.isin(basis, (0, 1, 2)).all():
        raise ValueError("point basis must be 0 ordinary policy, 1 fresh visual CRA, or 2 retained properties")
    arrays["point_basis"] = basis
    if "field_bounds_world" in geometry:
        boundary = np.asarray(geometry["field_bounds_world"], dtype=float)
        if boundary.shape != (4, 2) or not np.isfinite(boundary).all():
            raise ValueError("four finite active-field corners required")
        in_field = np.asarray(geometry["point_in_field"])
        if in_field.shape != basis.shape or in_field.dtype != np.dtype(bool):
            raise ValueError("original current-point field membership required")
        arrays["field_bounds_world"], arrays["point_in_field"] = boundary, in_field
    return arrays


def points_in_field(points, boundary):
    """Test retained samples against the archived rectangular field footprint."""
    origin = boundary[0]
    edges = np.column_stack((boundary[1] - origin, boundary[3] - origin))
    if abs(np.linalg.det(edges)) < 1e-12:
        raise ValueError("degenerate field boundary")
    if not np.allclose(boundary[2], boundary[1] + boundary[3] - origin, atol=1e-8, rtol=0):
        raise ValueError("field boundary must be a parallelogram")
    uv = (points - origin) @ np.linalg.inv(edges).T
    return np.all((uv >= 0) & (uv < 1), axis=1)


def render_lidar_risk_map(geometry, frame, nominal_route_world, travelled_world,
                          size=(856, 420), half_span_m=4.0, robot_footprint_local_xy=None):
    """Return image and viewport audit from an already source-aligned field.

    Solid points are the field's latest source returns, not a fictitious scan at
    video time. Hollow points are separately archived retained occupancy. Risk
    controls color only; provenance never changes the color. At overlapping
    solid display pixels, the highest applied point risk is drawn last.
    """
    if size != (856, 420):
        raise ValueError("this presentation layout requires 856 x 420 pixels")
    pose = np.asarray(frame["robot_world_xy_yaw"], dtype=float)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError("finite recorded presentation pose required")
    image = Image.new("RGB", size, "#182735")
    draw = ImageDraw.Draw(image)
    viewport = (10, 10, 524, 380)
    left, top, width, height = viewport
    scale = height / (2 * half_span_m)
    draw.rectangle((left, top, left + width, top + height), fill="#eef2f5", outline="#7b8b98")
    center = pose[:2]
    for axis in range(2):
        extent = (width if axis == 0 else height) / (2 * scale)
        for value in np.arange(math.ceil(center[axis] - extent), center[axis] + extent, 1.0):
            q = center.copy()
            q[axis] = value
            p = map_pixels([q], center, viewport, half_span_m)[0][0]
            line = (p[0], top, p[0], top + height) if axis == 0 else (left, p[1], left + width, p[1])
            draw.line(line, fill="#d9e0e5", width=1)

    # Clip path lines to the viewport by drawing them on a temporary image.
    path_image = Image.new("RGBA", (width, height))
    pd = ImageDraw.Draw(path_image)
    for route, color, line_width in ((nominal_route_world, "#929ba5", 2),
                                      (travelled_world, "#0088c2", 3)):
        pixels, _ = map_pixels(route, center, (0, 0, width, height), half_span_m)
        if len(pixels) > 1:
            pd.line([tuple(p) for p in pixels], fill=color, width=line_width)
    image.paste(path_image, (left, top), path_image)
    diagnostic = dict(viewport_world_center=center.tolist(), viewport_half_height_m=half_span_m,
                      viewport_half_width_m=width / (2 * scale), north_up=True,
                      map_score="controller_applied_phi", current_points=0, retained_points=0)
    info_x = 552
    if geometry is None:
        draw.text((info_x, 13), "No committed field", font=_font(21), fill="#f0cb88")
        draw.text((info_x, 45), "No risk colors fabricated", font=_font(17), fill="#c7d6e0")
        diagnostic["status"] = "no_committed_field"
    else:
        current_time, source_time, commit_time = (float(frame[k]) for k in ("time", "source_time", "commit_time"))
        if (not all(math.isfinite(t) for t in (current_time, source_time, commit_time))
                or source_time > commit_time + 1e-8 or commit_time > current_time + 1e-8):
            raise ValueError("future or invalid field clock")
        arrays = validate_geometry(geometry)
        active = arrays.get("point_in_field", np.ones(len(arrays["point_phi"]), dtype=bool))
        retained_active = (points_in_field(arrays["retained_world_xy"], arrays["field_bounds_world"])
                           if "field_bounds_world" in arrays else np.ones(len(arrays["retained_phi"]), dtype=bool))
        diagnostic["status"] = "committed_field"
        diagnostic["source_age_seconds"] = current_time - source_time
        for prefix, key, radius, hollow in (("retained", "retained_world_xy", 2, True),
                                            ("point", "points_world_xy", 2, False)):
            pixels, visible = map_pixels(arrays[key], center, viewport, half_span_m)
            phi = arrays[f"{prefix}_phi"]
            in_field = retained_active if hollow else active
            colors = {float(value): risk_color_rgb(float(value)) for value in np.unique(phi)}
            # Display-only max per pixel avoids drawing thousands of duplicate
            # vertical returns. The full point array/export is unchanged.
            order = np.flatnonzero(visible)
            order = order[np.lexsort((phi[order], in_field[order]))]
            _, unique = np.unique(np.rint(pixels[order[::-1]]).astype(np.int64), axis=0, return_index=True)
            rendered = order[::-1][unique]
            rendered = rendered[np.lexsort((phi[rendered], in_field[rendered]))]
            for j in rendered:
                x, y = pixels[j]
                bounds = (x - radius, y - radius, x + radius, y + radius)
                color = colors[float(phi[j])] if in_field[j] else "#a2aeb8"
                if hollow:
                    draw.ellipse(bounds, outline=color, width=1)
                elif arrays["point_basis"][j] == 0:
                    draw.rectangle((x - 1, y - 1, x + 1, y + 1), fill=color)
                elif arrays["point_basis"][j] == 2:
                    draw.polygon(((x, y - 3), (x + 3, y), (x, y + 3), (x - 3, y)), fill=color)
                else:
                    draw.ellipse(bounds, fill=color)
            diagnostic[f"{prefix}_visible"] = int(visible.sum())
            diagnostic[f"{prefix}_clipped"] = int((~visible).sum())
            diagnostic[f"{prefix}_display_pixels"] = int(len(rendered))
            diagnostic[f"{prefix}_in_field"] = int(in_field.sum())
            diagnostic[f"{prefix}_outside_field"] = int((~in_field).sum())
        if "field_bounds_world" in arrays:
            boundary = map_pixels(arrays["field_bounds_world"], center, (0, 0, width, height), half_span_m)[0]
            boundary_image = Image.new("RGBA", (width, height))
            ImageDraw.Draw(boundary_image).line([tuple(p) for p in boundary] + [tuple(boundary[0])],
                                               fill="#657785", width=1)
            image.paste(boundary_image, (left, top), boundary_image)
        basis = arrays["point_basis"]
        counts = [int(((basis == k) & active).sum()) for k in range(3)]
        diagnostic.update(current_points=len(basis), retained_points=len(arrays["retained_phi"]),
                          point_basis_counts=dict(ordinary_policy=counts[0], fresh=counts[1], retained=counts[2]))
        draw.text((info_x, 10), "CONTROLLER-APPLIED phi", font=_font(20), fill="white")
        draw.text((info_x, 39), f"Latest scan age: {current_time-source_time:.2f} s", font=_font(18), fill="#d1dce4")
        draw.text((info_x, 67), f"{int(active.sum()):,} field / {len(basis):,} returns", font=_font(18), fill="#d1dce4")
        for row, text in enumerate((f"Fresh visual CRA*: {counts[1]:,}",
                                    f"Retained properties*: {counts[2]:,}",
                                    f"Ordinary policy*: {counts[0]:,}")):
            y = 96 + 25 * row
            if row == 0:
                draw.ellipse((info_x, y + 7, info_x + 7, y + 14), fill="#d1dce4")
            elif row == 1:
                draw.polygon(((info_x + 4, y + 6), (info_x + 8, y + 10),
                              (info_x + 4, y + 14), (info_x, y + 10)), fill="#d1dce4")
            else:
                draw.rectangle((info_x + 1, y + 7, info_x + 7, y + 13), fill="#d1dce4")
            draw.text((info_x + 16, y), text, font=_font(17), fill="#d1dce4")
        draw.text((info_x, 174), "*Includes configured factor priors", font=_font(16), fill="#b7c5d0")
    for x in range(272):
        draw.line((info_x + x, 208, info_x + x, 224), fill=risk_color_rgb(x / 271))
    draw.text((info_x, 229), "0  lower risk", font=_font(16), fill="#d1dce4")
    draw.text((info_x + 175, 229), "higher risk  1", font=_font(16), fill="#d1dce4")
    for row, text in enumerate(("Solid: latest source returns", "Hollow: prior / retained geometry",
                                "Blue: travelled; grey: nominal", "Camera: original individual phi",
                                "Grey: outside active field")):
        draw.text((info_x, 263 + 26 * row), text, font=_font(16), fill="#b7c5d0")
    if robot_footprint_local_xy is not None:
        footprint = np.asarray(robot_footprint_local_xy, dtype=float)
        if footprint.ndim != 2 or footprint.shape[1] != 2 or len(footprint) < 3 or not np.isfinite(footprint).all():
            raise ValueError("finite recorded chassis polygon required")
        c, s = math.cos(pose[2]), math.sin(pose[2])
        world = footprint @ np.array([[c, s], [-s, c]]) + center
        outline = map_pixels(world, center, viewport, half_span_m)[0]
        draw.line([tuple(p) for p in outline] + [tuple(outline[0])], fill="#006c9e", width=2)
        diagnostic["robot_footprint"] = "recorded_chassis_polygon"
    # The central glyph alone is a localization marker, never an occupancy circle.
    p = map_pixels([center], center, viewport, half_span_m)[0][0]
    heading = np.array([math.cos(pose[2]), -math.sin(pose[2])])
    side = np.array([-heading[1], heading[0]])
    draw.polygon([tuple(p + 10 * heading), tuple(p - 6 * heading + 6 * side),
                  tuple(p - 6 * heading - 6 * side)], fill="#006c9e", outline="white")
    draw.text((left + 9, top + 7), "N / +Y", font=_font(15), fill="#445662")
    draw.text((left + width - 58, top + height - 27), "+X >", font=_font(15), fill="#445662")
    draw.line((left + 12, top + height - 18, left + 12 + scale, top + height - 18), fill="#445662", width=2)
    draw.text((left + 12, top + height - 41), "1 m", font=_font(14), fill="#445662")
    draw.text((12, 395), "Robot-centred view | recorded simulator localization | original scores", font=_font(15), fill="#b7c5d0")
    return image, diagnostic
