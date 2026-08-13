"""Tests for core/masking/selective_color_mask.py - the combined RGB-image
-> mask entry point that HSL/Color Wheels/Selective Color are meant to
share."""

import numpy as np

from core.masking.selective_color_mask import SelectiveColorMask


def _pixel_image(*colors):
    """A 1 x len(colors) x 3 float32 image, one pixel per given RGB color,
    so each test pixel can be indexed and asserted on independently."""
    return np.array([colors], dtype=np.float32)


def test_no_constraints_selects_everything():
    image = _pixel_image((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.2, 0.2, 0.2))
    mask = SelectiveColorMask().evaluate(image)
    assert mask.shape == image.shape[:2]
    assert np.allclose(mask, 1.0)


def test_hue_only_selects_matching_hue_and_excludes_opposite_hue():
    red = (1.0, 0.0, 0.0)     # hue 0
    blue = (0.0, 0.0, 1.0)    # hue 240
    image = _pixel_image(red, blue)
    mask = SelectiveColorMask(hue_center_deg=0.0, hue_width_deg=20.0, hue_feather_deg=20.0).evaluate(image)
    assert mask[0, 0] > 0.9   # red selected
    assert mask[0, 1] < 0.1   # blue excluded


def test_saturation_only_selects_saturated_and_excludes_gray():
    saturated_red = (1.0, 0.0, 0.0)
    gray = (0.5, 0.5, 0.5)
    image = _pixel_image(saturated_red, gray)
    mask = SelectiveColorMask(sat_low=0.5, sat_high=1.0, sat_feather=0.1).evaluate(image)
    assert mask[0, 0] > 0.9
    assert mask[0, 1] < 0.1


def test_luminance_only_selects_dark_and_excludes_bright():
    black = (0.02, 0.02, 0.02)
    white = (0.98, 0.98, 0.98)
    image = _pixel_image(black, white)
    mask = SelectiveColorMask(luma_low=0.0, luma_high=0.2, luma_feather=0.1).evaluate(image)
    assert mask[0, 0] > 0.9
    assert mask[0, 1] < 0.1


def test_combined_hue_and_saturation_requires_both_to_match():
    saturated_red = (1.0, 0.0, 0.0)      # matches hue and saturation
    desaturated_red = (0.6, 0.5, 0.5)    # matches hue only (low saturation)
    saturated_blue = (0.0, 0.0, 1.0)     # matches saturation only (wrong hue)
    image = _pixel_image(saturated_red, desaturated_red, saturated_blue)
    mask = SelectiveColorMask(
        hue_center_deg=0.0, hue_width_deg=15.0, hue_feather_deg=10.0,
        sat_low=0.5, sat_high=1.0, sat_feather=0.1,
    ).evaluate(image)
    assert mask[0, 0] > 0.9
    assert mask[0, 1] < 0.2
    assert mask[0, 2] < 0.1


def test_output_bounded_and_finite_for_out_of_gamut_and_nan_input():
    weird = np.array([[[1.5, -0.3, 0.5], [np.nan, 0.5, 0.5], [np.inf, -np.inf, 0.5]]], dtype=np.float32)
    mask = SelectiveColorMask(hue_center_deg=30.0, hue_width_deg=20.0, hue_feather_deg=20.0,
                               sat_low=0.2, sat_high=0.8, sat_feather=0.1,
                               luma_low=0.1, luma_high=0.9, luma_feather=0.1).evaluate(weird)
    assert mask.shape == weird.shape[:2]
    finite = np.isfinite(mask)
    assert np.all(mask[finite] >= 0.0) and np.all(mask[finite] <= 1.0)
