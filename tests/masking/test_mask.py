"""Tests for core/masking/mask.py - Mask/MaskComponent, the container that
combines the Masking panel's shape/selection types via
Add/Subtract/Intersect and applies the whole-mask Invert/Feather/Blur/
Density operations."""

import numpy as np
import pytest

from core.masking.mask import Mask, MaskComponent, COMBINE_OPS, COMPONENT_KINDS


def _rect(cx=0.5, cy=0.5, hw=0.3, hh=0.3, op="add", invert=False):
    return MaskComponent(
        kind="rectangle",
        params={"center_x": cx, "center_y": cy, "half_width": hw, "half_height": hh, "feather": 0.0},
        op=op, invert=invert,
    )


def test_component_rejects_unknown_kind():
    with pytest.raises(ValueError):
        MaskComponent(kind="not_a_real_kind")


def test_component_rejects_unknown_op():
    with pytest.raises(ValueError):
        MaskComponent(kind="rectangle", op="xor")


def test_empty_mask_evaluates_to_all_zero():
    mask = Mask()
    image = np.zeros((50, 50, 3), dtype=np.float32)
    result = mask.evaluate(image)
    assert result.shape == (50, 50)
    assert np.allclose(result, 0.0)
    assert mask.is_empty()


def test_single_component_mask_matches_the_shape_directly():
    mask = Mask(components=[_rect()])
    image = np.zeros((100, 100, 3), dtype=np.float32)
    result = mask.evaluate(image)
    assert result[50, 50] > 0.99   # center of the rectangle
    assert result[5, 5] < 0.01     # far corner


def test_add_combines_two_disjoint_shapes():
    left = MaskComponent(kind="rectangle", params={"center_x": 0.25, "center_y": 0.5, "half_width": 0.15, "half_height": 0.4})
    right = MaskComponent(kind="rectangle", params={"center_x": 0.75, "center_y": 0.5, "half_width": 0.15, "half_height": 0.4}, op="add")
    mask = Mask(components=[left, right])
    image = np.zeros((100, 100, 3), dtype=np.float32)
    result = mask.evaluate(image)
    assert result[50, 25] > 0.9   # inside left rect
    assert result[50, 75] > 0.9   # inside right rect
    assert result[50, 50] < 0.1   # gap between them


def test_subtract_removes_the_overlap():
    base = _rect(cx=0.5, cy=0.5, hw=0.3, hh=0.3)
    cutout = MaskComponent(kind="rectangle", params={"center_x": 0.5, "center_y": 0.5, "half_width": 0.1, "half_height": 0.1}, op="subtract")
    mask = Mask(components=[base, cutout])
    image = np.zeros((100, 100, 3), dtype=np.float32)
    result = mask.evaluate(image)
    assert result[50, 50] < 0.1     # the cut-out center
    assert result[50, 30] > 0.9     # still comfortably within the base rect, outside the cutout


def test_intersect_keeps_only_the_overlap():
    a = MaskComponent(kind="rectangle", params={"center_x": 0.4, "center_y": 0.5, "half_width": 0.3, "half_height": 0.3})
    b = MaskComponent(kind="rectangle", params={"center_x": 0.6, "center_y": 0.5, "half_width": 0.3, "half_height": 0.3}, op="intersect")
    mask = Mask(components=[a, b])
    image = np.zeros((100, 100, 3), dtype=np.float32)
    result = mask.evaluate(image)
    assert result[50, 50] > 0.9    # the overlapping middle
    assert result[50, 12] < 0.1    # only in "a", not "b"
    assert result[50, 88] < 0.1    # only in "b", not "a"


def test_first_components_op_is_ignored():
    """The very first component always seeds the mask outright - there's
    nothing to subtract/intersect against yet."""
    only = _rect(op="subtract")
    mask = Mask(components=[only])
    image = np.zeros((100, 100, 3), dtype=np.float32)
    result = mask.evaluate(image)
    assert result[50, 50] > 0.9   # behaves like "add", not an all-zero subtraction


def test_component_level_invert_flips_before_combining():
    normal = _rect()
    inverted = _rect(invert=True)
    image = np.zeros((100, 100, 3), dtype=np.float32)
    result_normal = Mask(components=[normal]).evaluate(image)
    result_inverted = Mask(components=[inverted]).evaluate(image)
    assert result_normal[50, 50] > 0.9
    assert result_inverted[50, 50] < 0.1
    assert result_inverted[5, 5] > 0.9


def test_mask_level_invert():
    mask = Mask(components=[_rect()], invert=True)
    image = np.zeros((100, 100, 3), dtype=np.float32)
    result = mask.evaluate(image)
    assert result[50, 50] < 0.1
    assert result[5, 5] > 0.9


def test_density_scales_the_whole_mask():
    full = Mask(components=[_rect()], density=100.0)
    half = Mask(components=[_rect()], density=50.0)
    image = np.zeros((100, 100, 3), dtype=np.float32)
    full_result = full.evaluate(image)
    half_result = half.evaluate(image)
    assert np.isclose(half_result[50, 50], full_result[50, 50] * 0.5, atol=1e-3)


def test_density_zero_makes_the_mask_inert():
    mask = Mask(components=[_rect()], density=0.0)
    image = np.zeros((100, 100, 3), dtype=np.float32)
    result = mask.evaluate(image)
    assert np.allclose(result, 0.0)


def test_feather_softens_the_boundary_without_changing_the_interior():
    hard_mask = Mask(components=[_rect(hw=0.3, hh=0.3)], feather=0.0)
    soft_mask = Mask(components=[_rect(hw=0.3, hh=0.3)], feather=30.0)
    image = np.zeros((100, 100, 3), dtype=np.float32)
    hard_result = hard_mask.evaluate(image)
    soft_result = soft_mask.evaluate(image)
    assert hard_result[50, 50] > 0.99
    assert soft_result[50, 50] > 0.9   # deep interior mostly unaffected
    assert not np.array_equal(hard_result, soft_result)


def test_blur_softens_edges_too():
    hard_mask = Mask(components=[_rect(hw=0.3, hh=0.3)], blur=0.0)
    blurred_mask = Mask(components=[_rect(hw=0.3, hh=0.3)], blur=20.0)
    image = np.zeros((100, 100, 3), dtype=np.float32)
    hard_result = hard_mask.evaluate(image)
    blurred_result = blurred_mask.evaluate(image)
    assert not np.array_equal(hard_result, blurred_result)
    assert np.isfinite(blurred_result).all()


def test_feather_and_blur_compose():
    mask = Mask(components=[_rect()], feather=20.0, blur=10.0)
    image = np.zeros((100, 100, 3), dtype=np.float32)
    result = mask.evaluate(image)
    assert np.isfinite(result).all()
    assert result.min() >= 0.0 and result.max() <= 1.0


def test_result_always_bounded_and_finite_for_a_complex_stack():
    rng = np.random.default_rng(0)
    image = rng.random((80, 80, 3)).astype(np.float32)
    components = [
        MaskComponent(kind="ellipse", params={"center_x": 0.4, "center_y": 0.4, "radius_x": 0.3, "radius_y": 0.2, "feather": 40}),
        MaskComponent(kind="luminance_range", params={"low": 0.2, "high": 0.8, "feather": 20}, op="intersect"),
        MaskComponent(kind="color_range", params={"sample_rgb": (0.5, 0.4, 0.3), "refine": 40}, op="subtract"),
    ]
    mask = Mask(components=components, feather=10.0, blur=5.0, density=80.0, invert=True)
    result = mask.evaluate(image)
    assert result.shape == (80, 80)
    assert np.isfinite(result).all()
    assert result.min() >= 0.0 and result.max() <= 1.0


def test_all_documented_kinds_and_ops_are_exposed():
    assert "add" in COMBINE_OPS and "subtract" in COMBINE_OPS and "intersect" in COMBINE_OPS
    for kind in ("brush", "radial", "ellipse", "linear_gradient", "rectangle", "polygon",
                 "color_range", "luminance_range", "subject", "sky", "skin"):
        assert kind in COMPONENT_KINDS
