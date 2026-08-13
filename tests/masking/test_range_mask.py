"""Tests for core/masking/range_mask.py (and the thin saturation/luminance
wrappers built on it)."""

import numpy as np

from core.masking.range_mask import range_mask
from core.masking.saturation_range import saturation_range_mask
from core.masking.luminance_range import luminance_zone_mask


def test_value_inside_range_is_full_strength():
    value = np.array([0.5])
    assert np.isclose(range_mask(value, 0.3, 0.7, 0.1)[0], 1.0)


def test_value_far_outside_range_is_zero():
    value = np.array([0.0])
    assert np.isclose(range_mask(value, 0.5, 0.6, 0.05)[0], 0.0, atol=1e-6)


def test_value_beyond_feather_is_exactly_zero():
    value = np.array([1.0])
    assert range_mask(value, 0.0, 0.1, 0.05)[0] == 0.0


def test_mask_is_monotonic_falling_off_outside_the_range():
    values = np.linspace(0.0, 1.0, 200)
    mask = range_mask(values, 0.4, 0.6, 0.15)
    # Below the range, mask should be non-decreasing as value rises toward
    # the range; above it, non-increasing as value moves away.
    below = values < 0.4
    above = values > 0.6
    below_mask = mask[below]
    above_mask = mask[above]
    assert np.all(np.diff(below_mask) >= -1e-9)
    assert np.all(np.diff(above_mask) <= 1e-9)


def test_output_bounded_in_zero_one_for_extreme_inputs():
    values = np.array([-100.0, -1.0, 0.0, 0.5, 1.0, 2.0, 1000.0])
    mask = range_mask(values, 0.3, 0.7, 0.1)
    assert np.all(mask >= 0.0) and np.all(mask <= 1.0)


def test_saturation_and_luminance_wrappers_delegate_correctly():
    values = np.array([0.5])
    assert np.isclose(saturation_range_mask(values, 0.3, 0.7, 0.1), range_mask(values, 0.3, 0.7, 0.1))
    assert np.isclose(luminance_zone_mask(values, 0.3, 0.7, 0.1), range_mask(values, 0.3, 0.7, 0.1))


def test_nan_and_inf_do_not_raise():
    values = np.array([np.nan, np.inf, -np.inf])
    mask = range_mask(values, 0.3, 0.7, 0.1)
    assert mask.shape == values.shape
