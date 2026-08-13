"""Tests for core/masking/combine.py."""

import numpy as np
import pytest

from core.masking.combine import combine_masks


def test_single_mask_passthrough():
    m = np.array([0.0, 0.5, 1.0])
    assert np.allclose(combine_masks(m), m)


def test_two_full_masks_combine_to_full():
    ones = np.ones(4)
    assert np.allclose(combine_masks(ones, ones), 1.0)


def test_any_zero_mask_zeroes_the_result():
    ones = np.ones(4)
    zeros = np.zeros(4)
    assert np.allclose(combine_masks(ones, zeros, ones), 0.0)


def test_partial_masks_multiply():
    a = np.array([0.5])
    b = np.array([0.4])
    assert np.isclose(combine_masks(a, b)[0], 0.2)


def test_result_is_clipped_to_zero_one():
    over = np.array([1.5])
    assert combine_masks(over, over)[0] <= 1.0


def test_requires_at_least_one_mask():
    with pytest.raises(ValueError):
        combine_masks()
