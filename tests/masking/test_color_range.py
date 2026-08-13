"""Tests for core/masking/color_range.py."""

import numpy as np

from core.masking.color_range import color_range_mask


def test_selects_pixels_matching_the_sampled_color():
    image = np.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float32)  # red, blue
    mask = color_range_mask(image, sample_rgb=(1.0, 0.0, 0.0), refine=50.0)
    assert mask[0, 0] > 0.9
    assert mask[0, 1] < 0.1


def test_higher_refine_narrows_the_selection():
    image = np.array([[[1.0, 0.0, 0.0], [0.9, 0.3, 0.1]]], dtype=np.float32)  # red, orange-ish red
    loose = color_range_mask(image, sample_rgb=(1.0, 0.0, 0.0), refine=10.0)
    tight = color_range_mask(image, sample_rgb=(1.0, 0.0, 0.0), refine=95.0)
    assert loose[0, 1] >= tight[0, 1]


def test_out_of_gamut_and_nan_input_do_not_raise():
    # NaN may propagate through (as elsewhere in core/masking - see
    # SelectiveColorMask, which this delegates to); the requirement is
    # "does not raise" and that whatever IS finite stays in [0, 1].
    image = np.array([[[1.8, -0.4, np.nan]]], dtype=np.float32)
    mask = color_range_mask(image, sample_rgb=(0.5, 0.5, 0.5), refine=50.0)
    finite = np.isfinite(mask)
    assert mask[finite].min() >= 0.0 if finite.any() else True
    assert mask[finite].max() <= 1.0 if finite.any() else True
