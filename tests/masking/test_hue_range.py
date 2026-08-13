"""Tests for core/masking/hue_range.py - circular hue distance and the
hue-range mask built on it."""

import numpy as np

from core.masking.hue_range import circular_hue_distance, hue_range_mask


def test_circular_distance_same_hue_is_zero():
    assert circular_hue_distance(np.array([120.0]), 120.0)[0] == 0.0


def test_circular_distance_wraps_around_360():
    """0 degrees (red) and 359 degrees are only 1 degree apart on the
    circle, not 359 - this is the whole point of circular_hue_distance
    over a plain abs(a - b)."""
    d = circular_hue_distance(np.array([359.0]), 0.0)[0]
    assert np.isclose(d, 1.0)


def test_circular_distance_opposite_hues_is_180():
    d = circular_hue_distance(np.array([180.0]), 0.0)[0]
    assert np.isclose(d, 180.0)


def test_circular_distance_is_symmetric():
    a = circular_hue_distance(np.array([300.0]), 40.0)[0]
    b = circular_hue_distance(np.array([40.0]), 300.0)[0]
    assert np.isclose(a, b)


def test_hue_mask_full_strength_at_and_near_center():
    hues = np.array([60.0, 60.0 + 5.0, 60.0 - 5.0])
    mask = hue_range_mask(hues, center_deg=60.0, width_deg=20.0, feather_deg=10.0)
    assert np.allclose(mask, 1.0)


def test_hue_mask_zero_far_from_center():
    hues = np.array([240.0])  # opposite side of the wheel from 60
    mask = hue_range_mask(hues, center_deg=60.0, width_deg=20.0, feather_deg=10.0)
    assert np.isclose(mask[0], 0.0, atol=1e-6)


def test_hue_mask_wraps_around_zero():
    """A mask centered at 0 degrees (red) should include hues just below
    360 as well as just above 0."""
    hues = np.array([358.0, 2.0])
    mask = hue_range_mask(hues, center_deg=0.0, width_deg=5.0, feather_deg=5.0)
    assert np.all(mask > 0.99)


def test_hue_mask_is_bounded_and_finite():
    hues = np.linspace(-720.0, 720.0, 100)
    mask = hue_range_mask(hues, center_deg=90.0, width_deg=15.0, feather_deg=15.0)
    assert np.isfinite(mask).all()
    assert np.all(mask >= 0.0) and np.all(mask <= 1.0)
