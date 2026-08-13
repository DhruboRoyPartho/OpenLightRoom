"""Geometric mask primitives: Radial, Ellipse, Linear Gradient, Rectangle,
Polygon, and Brush - the shape-based mask types in the Masking panel.

Every function here takes only (height, width) plus normalized ([0, 1])
geometry - never raw pixel coordinates - so a mask defined while editing at
one preview resolution renders identically (just re-rasterized) at any
other resolution, exactly like the app's existing normalized Crop rect.
Every function is fully vectorized numpy/OpenCV; the only Python-level
loops are over a bounded number of brush dabs (tens to a couple hundred
per stroke), never over image pixels.
"""

import math
import numpy as np
import cv2

from core.processing.color_space import smoothstep

DEFAULT_BRUSH_RADIUS = 0.04     # fraction of the image diagonal
DEFAULT_BRUSH_HARDNESS = 80.0   # 0..100
DEFAULT_BRUSH_FLOW = 100.0      # 0..100


def _distance_falloff(distance: np.ndarray, feather_pct: float) -> np.ndarray:
    """distance: 0 at the shape's center, 1 at its boundary (any
    monotonic per-shape distance metric - Euclidean for ellipses,
    Chebyshev for rectangles). feather_pct: 0..100, how much of the
    radius the soft transition occupies - 0 is a hard edge exactly at
    distance=1, 100 softens all the way in to the center. Shared by
    radial_mask/ellipse_mask/rectangle_mask so they read identically.
    """
    feather_frac = np.clip(feather_pct, 0.0, 100.0) / 100.0
    inner = 1.0 - feather_frac
    t = np.clip((1.0 - distance) / max(1.0 - inner, 1e-6), 0.0, 1.0)
    return smoothstep(t)


def _rotated_local_coords(h: int, w: int, center_x: float, center_y: float, angle_deg: float):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = center_x * w, center_y * h
    dx, dy = xx - cx, yy - cy
    theta = math.radians(angle_deg)
    cos_a, sin_a = math.cos(theta), math.sin(theta)
    dx_r = dx * cos_a + dy * sin_a
    dy_r = -dx * sin_a + dy * cos_a
    return dx_r, dy_r


def _ellipse_distance(h: int, w: int, center_x: float, center_y: float,
                       radius_x: float, radius_y: float, angle_deg: float) -> np.ndarray:
    dx_r, dy_r = _rotated_local_coords(h, w, center_x, center_y, angle_deg)
    rx, ry = max(radius_x * w, 1e-3), max(radius_y * h, 1e-3)
    return np.sqrt((dx_r / rx) ** 2 + (dy_r / ry) ** 2)


def radial_mask(h: int, w: int, center_x: float, center_y: float,
                 radius_x: float, radius_y: float, angle_deg: float = 0.0,
                 feather: float = 50.0) -> np.ndarray:
    """center/radii as fractions of (width, height). 1.0 inside the
    ellipse, smoothly fading to 0.0 over `feather` percent of the radius -
    Lightroom's Radial Filter, which defaults to a generous feather since
    it's meant as a soft vignette-style gradient rather than a hard
    cutout (see ellipse_mask for the shape-mask sibling with a harder
    default edge)."""
    distance = _ellipse_distance(h, w, center_x, center_y, radius_x, radius_y, angle_deg)
    return _distance_falloff(distance, feather).astype(np.float32)


def ellipse_mask(h: int, w: int, center_x: float, center_y: float,
                  radius_x: float, radius_y: float, angle_deg: float = 0.0,
                  feather: float = 0.0) -> np.ndarray:
    """Identical falloff math to radial_mask - kept as a distinct entry
    point (and distinct default feather=0, a crisp shape edge) since
    Radial and Ellipse are presented as separate tools with different
    default intents, matching the masking panel's own Radial vs. Ellipse
    distinction."""
    distance = _ellipse_distance(h, w, center_x, center_y, radius_x, radius_y, angle_deg)
    return _distance_falloff(distance, feather).astype(np.float32)


def rectangle_mask(h: int, w: int, center_x: float, center_y: float,
                    half_width: float, half_height: float, angle_deg: float = 0.0,
                    feather: float = 0.0) -> np.ndarray:
    """center/half-extents as fractions of (width, height). Same falloff
    shape as ellipse_mask, but using Chebyshev (max-axis) distance instead
    of Euclidean, which is what turns a circle into a rectangle."""
    dx_r, dy_r = _rotated_local_coords(h, w, center_x, center_y, angle_deg)
    hw, hh = max(half_width * w, 1e-3), max(half_height * h, 1e-3)
    distance = np.maximum(np.abs(dx_r) / hw, np.abs(dy_r) / hh)
    return _distance_falloff(distance, feather).astype(np.float32)


def linear_gradient_mask(h: int, w: int, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """(x0,y0)->(x1,y1), normalized: 0.0 at the start point, 1.0 at the
    end point, linearly interpolated along that line and clamped beyond
    either end - Lightroom's Graduated Filter. The transition's softness
    is simply the distance between the two points (drag them closer
    together for a sharper edge); an overall extra Blur is available as a
    Mask-level operation (see mask.py) for additional softening on top.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    px0, py0 = x0 * w, y0 * h
    px1, py1 = x1 * w, y1 * h
    dx, dy = px1 - px0, py1 - py0
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-9:
        return np.zeros((h, w), dtype=np.float32)
    t = ((xx - px0) * dx + (yy - py0) * dy) / length_sq
    return np.clip(t, 0.0, 1.0).astype(np.float32)


def polygon_mask(h: int, w: int, points, feather: float = 0.0) -> np.ndarray:
    """points: a list of (x, y) normalized [0, 1] vertices, >= 3 points.
    Rasterized with cv2.fillPoly for the hard interior, then (if
    feather > 0) softened with a signed distance transform - a true
    distance-based feather around the actual polygon boundary, half
    inward and half outward, rather than an approximate per-edge falloff.
    """
    if len(points) < 3:
        return np.zeros((h, w), dtype=np.float32)

    pts = np.array([[x * w, y * h] for x, y in points], dtype=np.int32)
    hard = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(hard, [pts], 1)

    if feather <= 0:
        return hard.astype(np.float32)

    feather_px = max(1.0, (feather / 100.0) * min(h, w) * 0.5)
    dist_inside = cv2.distanceTransform(hard, cv2.DIST_L2, 5).astype(np.float32)
    dist_outside = cv2.distanceTransform(1 - hard, cv2.DIST_L2, 5).astype(np.float32)
    signed_distance = dist_inside - dist_outside  # >0 inside, <0 outside, 0 at the boundary
    t = np.clip((signed_distance / feather_px + 1.0) / 2.0, 0.0, 1.0)
    return smoothstep(t).astype(np.float32)


# --- brush -----------------------------------------------------------------

def _soft_circle_kernel(radius_px: float, hardness: float) -> np.ndarray:
    r = max(int(round(radius_px)), 1)
    size = 2 * r + 1
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) - r
    distance = np.sqrt(xx ** 2 + yy ** 2) / max(r, 1e-6)
    return _distance_falloff(distance, 100.0 - np.clip(hardness, 0.0, 100.0)).astype(np.float32)


def _resample_polyline(points, spacing: float):
    """Walks a polyline and emits a point every `spacing` distance, so a
    fast mouse drag (few recorded points, large gaps) still stamps a
    continuous brush stroke instead of leaving holes."""
    if len(points) <= 1:
        return list(points)
    out = [points[0]]
    carry = 0.0
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        seg_len = math.hypot(x1 - x0, y1 - y0)
        if seg_len < 1e-9:
            continue
        d = spacing - carry
        while d < seg_len:
            t = d / seg_len
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
            d += spacing
        carry = seg_len - (d - spacing)
    out.append(points[-1])
    return out


def _stamp_dabs(h: int, w: int, dab_centers_px, kernel: np.ndarray) -> np.ndarray:
    canvas = np.zeros((h, w), dtype=np.float32)
    kh, kw = kernel.shape
    half_h, half_w = kh // 2, kw // 2

    for px, py in dab_centers_px:
        cx, cy = int(round(px)), int(round(py))
        x0, y0 = cx - half_w, cy - half_h
        x1, y1 = x0 + kw, y0 + kh

        src_x0, src_y0 = max(0, -x0), max(0, -y0)
        dst_x0, dst_y0 = max(0, x0), max(0, y0)
        dst_x1, dst_y1 = min(w, x1), min(h, y1)
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            continue
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        region = canvas[dst_y0:dst_y1, dst_x0:dst_x1]
        np.maximum(region, kernel[src_y0:src_y1, src_x0:src_x1], out=region)

    return canvas


def rasterize_stroke(h: int, w: int, points, radius: float = DEFAULT_BRUSH_RADIUS,
                      hardness: float = DEFAULT_BRUSH_HARDNESS, flow: float = DEFAULT_BRUSH_FLOW) -> np.ndarray:
    """points: a list of (x, y) normalized [0, 1] positions recorded along
    one continuous mouse drag. radius: fraction of the image diagonal.
    Returns a single stroke's own HxW float32 mask in [0, 1] (dabs within
    one stroke compose via max(), so a slow drag re-passing the same spot
    doesn't build up past the stroke's own flow - see brush_mask() for how
    separate strokes accumulate)."""
    if not points:
        return np.zeros((h, w), dtype=np.float32)

    diagonal = math.hypot(h, w)
    radius_px = max(radius * diagonal, 1.0)
    px_points = [(x * w, y * h) for x, y in points]
    dab_centers = _resample_polyline(px_points, spacing=max(radius_px * 0.2, 1.0))
    kernel = _soft_circle_kernel(radius_px, hardness)

    stroke = _stamp_dabs(h, w, dab_centers, kernel)
    stroke *= np.clip(flow, 0.0, 100.0) / 100.0
    return np.clip(stroke, 0.0, 1.0).astype(np.float32)


def brush_mask(h: int, w: int, strokes) -> np.ndarray:
    """strokes: a list of dicts, each:
        {"points": [(x, y), ...] normalized,
         "radius": float (fraction of image diagonal, default 0.04),
         "hardness": float 0..100 (default 80),
         "flow": float 0..100 (default 100),
         "mode": "add" | "subtract" (default "add")}
    Separate strokes accumulate (an "add" stroke over already-painted area
    builds up further, like repeated brush passes; a "subtract" stroke
    erases), each capped so the total never exceeds full strength -
    standard digital-painting compositing, not a simple overwrite.
    """
    result = np.zeros((h, w), dtype=np.float32)
    for stroke in strokes:
        stroke_mask = rasterize_stroke(
            h, w,
            stroke.get("points", []),
            radius=stroke.get("radius", DEFAULT_BRUSH_RADIUS),
            hardness=stroke.get("hardness", DEFAULT_BRUSH_HARDNESS),
            flow=stroke.get("flow", DEFAULT_BRUSH_FLOW),
        )
        if stroke.get("mode", "add") == "subtract":
            result = result * (1.0 - stroke_mask)
        else:
            result = 1.0 - (1.0 - result) * (1.0 - stroke_mask)
    return np.clip(result, 0.0, 1.0).astype(np.float32)
