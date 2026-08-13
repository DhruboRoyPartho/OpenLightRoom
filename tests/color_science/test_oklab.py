"""Tests for core/color_science/oklab.py."""

import numpy as np
import pytest

from core.color_science import oklab


def test_round_trip_random_linear_srgb():
    rng = np.random.default_rng(5)
    rgb = rng.random((200, 3))
    lab = oklab.linear_srgb_to_oklab(rgb)
    back = oklab.oklab_to_linear_srgb(lab)
    assert np.allclose(back, rgb, atol=1e-6)


def test_oklab_oklch_round_trip():
    rng = np.random.default_rng(6)
    rgb = rng.random((100, 3))
    lab = oklab.linear_srgb_to_oklab(rgb)
    lch = oklab.oklab_to_oklch(lab)
    back = oklab.oklch_to_oklab(lch)
    assert np.allclose(back, lab, atol=1e-9)


def test_full_round_trip_rgb_to_oklch_to_rgb():
    rng = np.random.default_rng(7)
    rgb = rng.random((100, 3))
    lch = oklab.linear_srgb_to_oklch(rgb)
    back = oklab.oklch_to_linear_srgb(lch)
    assert np.allclose(back, rgb, atol=1e-6)


def test_neutral_gray_has_zero_chroma():
    for v in [0.0, 0.18, 0.5, 1.0]:
        gray = np.array([v, v, v])
        lch = oklab.linear_srgb_to_oklch(gray)
        assert lch[1] < 1e-6, f"expected ~0 chroma for neutral gray, got {lch[1]}"


def test_black_has_zero_lightness():
    lab = oklab.linear_srgb_to_oklab(np.array([0.0, 0.0, 0.0]))
    assert np.isclose(lab[0], 0.0, atol=1e-9)


def test_white_has_higher_lightness_than_black():
    l_black = oklab.linear_srgb_to_oklab(np.array([0.0, 0.0, 0.0]))[0]
    l_white = oklab.linear_srgb_to_oklab(np.array([1.0, 1.0, 1.0]))[0]
    assert l_white > l_black


def test_hue_ordering_is_stable_for_primary_colors():
    """Red, green and blue should land at distinct, well-separated hue
    angles - a sanity check that the LMS/OKLab matrices are wired up in
    the correct orientation rather than transposed or mismatched."""
    red_h = oklab.linear_srgb_to_oklch(np.array([1.0, 0.0, 0.0]))[2]
    green_h = oklab.linear_srgb_to_oklch(np.array([0.0, 1.0, 0.0]))[2]
    blue_h = oklab.linear_srgb_to_oklch(np.array([0.0, 0.0, 1.0]))[2]
    hues = [red_h, green_h, blue_h]
    assert len(set(round(h) for h in hues)) == 3  # all distinct
    for h in hues:
        assert 0.0 <= h < 360.0


def test_cube_root_handles_negative_lms_without_nan():
    """A saturated/out-of-gamut linear RGB value can produce a negative
    LMS component; np.cbrt (used internally) must handle that correctly
    instead of producing NaN the way `x ** (1/3)` would for negative x."""
    saturated_out_of_gamut = np.array([-0.5, 1.5, -0.2])
    lab = oklab.linear_srgb_to_oklab(saturated_out_of_gamut)
    assert np.isfinite(lab).all()
    back = oklab.oklab_to_linear_srgb(lab)
    assert np.allclose(back, saturated_out_of_gamut, atol=1e-6)


def test_shape_preserved_for_image_like_arrays():
    rng = np.random.default_rng(8)
    rgb = rng.random((12, 16, 3))
    lch = oklab.linear_srgb_to_oklch(rgb)
    assert lch.shape == rgb.shape
    back = oklab.oklch_to_linear_srgb(lch)
    assert np.allclose(back, rgb, atol=1e-6)


def test_nan_and_inf_do_not_raise():
    values = np.array([[np.nan, 0.5, 0.5], [np.inf, 0.5, 0.5], [-np.inf, -np.inf, -np.inf]])
    lch = oklab.linear_srgb_to_oklch(values)
    assert lch.shape == values.shape
