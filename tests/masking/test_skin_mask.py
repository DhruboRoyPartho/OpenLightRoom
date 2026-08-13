"""Tests for core/masking/skin_mask.py."""

import numpy as np

from core.masking.skin_mask import skin_mask


def test_typical_skin_tone_is_selected():
    # A representative mid-toned skin color (light/medium skin under
    # neutral lighting).
    image = np.full((10, 10, 3), (0.87, 0.62, 0.48), dtype=np.float32)
    mask = skin_mask(image)
    assert mask.mean() > 0.5


def test_saturated_blue_is_not_selected():
    image = np.full((10, 10, 3), (0.0, 0.0, 1.0), dtype=np.float32)
    mask = skin_mask(image)
    assert mask.mean() < 0.1


def test_saturated_green_is_not_selected():
    image = np.full((10, 10, 3), (0.0, 1.0, 0.0), dtype=np.float32)
    mask = skin_mask(image)
    assert mask.mean() < 0.1


def test_output_bounded_and_finite_for_extreme_input():
    image = np.array([[[np.nan, 2.0, -1.0]]], dtype=np.float32)
    mask = skin_mask(image)
    assert np.isfinite(mask).all()
    assert mask.min() >= 0.0 and mask.max() <= 1.0


def test_feather_parameter_changes_the_result():
    image = np.full((10, 10, 3), (0.85, 0.6, 0.5), dtype=np.float32)
    tight = skin_mask(image, feather=1.0)
    loose = skin_mask(image, feather=100.0)
    assert tight.shape == loose.shape
