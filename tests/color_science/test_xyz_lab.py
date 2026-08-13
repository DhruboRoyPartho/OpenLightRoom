"""Tests for core/color_science/xyz_lab.py."""

import numpy as np
import pytest

from core.color_science import xyz_lab


def test_d65_white_maps_to_l100_a0_b0():
    lab = xyz_lab.xyz_to_lab(xyz_lab.D65_WHITE_XYZ)
    assert np.allclose(lab, [100.0, 0.0, 0.0], atol=1e-6)


def test_black_maps_to_l0():
    lab = xyz_lab.xyz_to_lab(np.array([0.0, 0.0, 0.0]))
    assert np.allclose(lab, [0.0, 0.0, 0.0], atol=1e-6)


def test_round_trip_random_xyz():
    rng = np.random.default_rng(3)
    xyz = rng.random((100, 3))
    lab = xyz_lab.xyz_to_lab(xyz)
    back = xyz_lab.lab_to_xyz(lab)
    assert np.allclose(back, xyz, atol=1e-9)


def test_round_trip_near_black_uses_linear_toe_branch():
    """Values below the (6/29)^3 threshold exercise the linear segment of
    the piecewise f() function - a common source of bugs if the branches
    don't meet continuously at the threshold."""
    xyz = np.array([1e-6, 1e-6, 1e-6])
    lab = xyz_lab.xyz_to_lab(xyz)
    back = xyz_lab.lab_to_xyz(lab)
    assert np.allclose(back, xyz, atol=1e-12)


def test_d50_white_point_option():
    lab = xyz_lab.xyz_to_lab(xyz_lab.D50_WHITE_XYZ, white=xyz_lab.D50_WHITE_XYZ)
    assert np.allclose(lab, [100.0, 0.0, 0.0], atol=1e-6)


def test_shape_preserved_for_image_like_arrays():
    rng = np.random.default_rng(4)
    xyz = rng.random((8, 10, 3))
    lab = xyz_lab.xyz_to_lab(xyz)
    assert lab.shape == xyz.shape
    back = xyz_lab.lab_to_xyz(lab)
    assert np.allclose(back, xyz, atol=1e-9)


def test_negative_and_nan_do_not_raise():
    values = np.array([[-0.5, 0.5, 0.5], [np.nan, 0.5, 0.5]])
    lab = xyz_lab.xyz_to_lab(values)
    assert lab.shape == values.shape
