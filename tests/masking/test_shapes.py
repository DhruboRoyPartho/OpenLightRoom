"""Tests for core/masking/shapes.py - the Masking panel's geometric mask
types (Radial, Ellipse, Linear Gradient, Rectangle, Polygon, Brush)."""

import numpy as np

from core.masking.shapes import (
    radial_mask, ellipse_mask, rectangle_mask, linear_gradient_mask, polygon_mask,
    brush_mask, rasterize_stroke, _resample_polyline,
)


# --- radial / ellipse -------------------------------------------------------

def test_radial_mask_is_full_strength_at_center():
    mask = radial_mask(200, 200, 0.5, 0.5, 0.3, 0.3, feather=50.0)
    assert mask[100, 100] > 0.99


def test_radial_mask_fades_to_zero_far_outside():
    mask = radial_mask(200, 200, 0.5, 0.5, 0.1, 0.1, feather=20.0)
    assert mask[0, 0] < 0.01


def test_ellipse_mask_zero_feather_is_a_hard_edge():
    mask = ellipse_mask(100, 100, 0.5, 0.5, 0.2, 0.2, feather=0.0)
    assert mask[50, 50] > 0.99          # center: inside
    assert mask[5, 5] < 0.01            # corner: well outside
    unique_ish = np.unique(np.round(mask, 2))
    assert len(unique_ish) <= 3         # ~binary: 0, 1, and a thin AA edge band


def test_ellipse_respects_aspect_ratio():
    """A wide, short ellipse should include a point far to the side but
    exclude the same distance vertically."""
    mask = ellipse_mask(200, 200, 0.5, 0.5, 0.4, 0.1, feather=0.0)
    assert mask[100, 170] > 0.5   # far right, within the wide radius
    assert mask[130, 100] < 0.5   # only moderately below center, outside the short radius


def test_ellipse_rotation_moves_the_long_axis():
    unrotated = ellipse_mask(200, 200, 0.5, 0.5, 0.35, 0.1, angle_deg=0.0, feather=0.0)
    rotated = ellipse_mask(200, 200, 0.5, 0.5, 0.35, 0.1, angle_deg=90.0, feather=0.0)
    # A point comfortably to the right of center (well inside the 70px
    # horizontal radius, not right at its boundary) is inside the
    # unrotated (wide) ellipse but outside the 90-degree-rotated (now
    # tall, 20px-wide) one.
    assert unrotated[100, 150] > 0.5
    assert rotated[100, 150] < 0.5


# --- rectangle ---------------------------------------------------------------

def test_rectangle_mask_hard_edge_shape():
    mask = rectangle_mask(200, 200, 0.5, 0.5, 0.2, 0.1, feather=0.0)
    assert mask[100, 100] > 0.99                # center
    assert mask[100, 100 + 30] > 0.99           # still within half_width*w=40px
    assert mask[100, 100 + 60] < 0.01           # beyond half_width
    assert mask[100 - 30, 100] < 0.01           # beyond half_height (0.1*200=20px)


def test_rectangle_mask_feather_softens_the_edge():
    hard = rectangle_mask(200, 200, 0.5, 0.5, 0.2, 0.2, feather=0.0)
    soft = rectangle_mask(200, 200, 0.5, 0.5, 0.2, 0.2, feather=50.0)
    edge_hard = hard[100, 139]
    edge_soft = soft[100, 139]
    assert 0.0 < edge_soft < 1.0
    assert edge_soft != edge_hard


# --- linear gradient -----------------------------------------------------

def test_linear_gradient_endpoints():
    mask = linear_gradient_mask(100, 100, x0=0.2, y0=0.5, x1=0.8, y1=0.5)
    assert mask[50, 0] < 0.05     # before the start, clamped to ~0
    assert mask[50, 99] > 0.95    # past the end, clamped to ~1


def test_linear_gradient_is_monotonic_along_its_axis():
    mask = linear_gradient_mask(10, 100, x0=0.0, y0=0.5, x1=1.0, y1=0.5)
    row = mask[5, :]
    assert np.all(np.diff(row) >= -1e-6)


def test_linear_gradient_degenerate_points_returns_zero_not_nan():
    mask = linear_gradient_mask(50, 50, x0=0.5, y0=0.5, x1=0.5, y1=0.5)
    assert np.isfinite(mask).all()
    assert np.allclose(mask, 0.0)


# --- polygon -----------------------------------------------------------------

def test_polygon_mask_requires_at_least_three_points():
    mask = polygon_mask(50, 50, [(0.1, 0.1), (0.9, 0.9)])
    assert np.allclose(mask, 0.0)


def test_polygon_mask_selects_its_interior():
    triangle = [(0.5, 0.1), (0.1, 0.9), (0.9, 0.9)]
    mask = polygon_mask(200, 200, triangle, feather=0.0)
    assert mask[170, 100] > 0.5   # near the triangle's base/centroid area
    assert mask[10, 10] < 0.5     # top-left corner, outside the triangle


def test_polygon_mask_feather_softens_the_boundary():
    square = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    hard = polygon_mask(200, 200, square, feather=0.0)
    soft = polygon_mask(200, 200, square, feather=20.0)
    assert hard[100, 40] in (0.0, 1.0)
    boundary_soft_value = soft[100, 40]
    assert 0.0 <= boundary_soft_value <= 1.0
    assert not np.array_equal(hard, soft)


# --- brush ---------------------------------------------------------------

def test_resample_polyline_fills_gaps():
    points = [(0.0, 0.0), (1.0, 0.0)]
    resampled = _resample_polyline(points, spacing=0.1)
    assert len(resampled) > 2
    assert resampled[0] == points[0]
    assert resampled[-1] == points[-1]


def test_rasterize_stroke_paints_along_a_dragged_path():
    points = [(0.2, 0.5), (0.5, 0.5), (0.8, 0.5)]
    mask = rasterize_stroke(200, 200, points, radius=0.05, hardness=100.0, flow=100.0)
    assert mask[100, 100] > 0.9   # midpoint of the stroke
    assert mask[10, 10] < 0.1     # far corner, untouched


def test_brush_mask_empty_strokes_is_all_zero():
    mask = brush_mask(100, 100, [])
    assert np.allclose(mask, 0.0)


def test_brush_mask_subtract_stroke_erases_previous_paint():
    add_stroke = {"points": [(0.5, 0.5)], "radius": 0.2, "hardness": 100.0, "flow": 100.0, "mode": "add"}
    subtract_stroke = {"points": [(0.5, 0.5)], "radius": 0.2, "hardness": 100.0, "flow": 100.0, "mode": "subtract"}

    painted = brush_mask(200, 200, [add_stroke])
    erased = brush_mask(200, 200, [add_stroke, subtract_stroke])

    assert painted[100, 100] > 0.9
    assert erased[100, 100] < 0.1


def test_brush_mask_low_flow_caps_strength_of_a_single_pass():
    stroke = {"points": [(0.5, 0.5)], "radius": 0.2, "hardness": 100.0, "flow": 30.0, "mode": "add"}
    mask = brush_mask(200, 200, [stroke])
    assert mask[100, 100] <= 0.31


def test_brush_mask_repeated_passes_build_up_strength():
    stroke = {"points": [(0.5, 0.5)], "radius": 0.2, "hardness": 100.0, "flow": 30.0, "mode": "add"}
    one_pass = brush_mask(200, 200, [stroke])
    two_passes = brush_mask(200, 200, [stroke, stroke])
    assert two_passes[100, 100] > one_pass[100, 100]


def test_brush_mask_output_bounded_and_finite():
    rng = np.random.default_rng(0)
    points = [(float(x), float(y)) for x, y in rng.random((15, 2))]
    stroke = {"points": points, "radius": 0.1, "hardness": 50.0, "flow": 80.0}
    mask = brush_mask(150, 150, [stroke])
    assert np.isfinite(mask).all()
    assert mask.min() >= 0.0 and mask.max() <= 1.0
