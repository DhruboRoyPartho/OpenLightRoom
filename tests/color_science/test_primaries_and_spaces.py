"""Tests for core/color_science/primaries.py and spaces.py: matrix
derivation correctness and round-trip conversion between color spaces."""

import numpy as np
import pytest

from core.color_science import primaries as prim
from core.color_science import spaces


# The widely-published reference sRGB -> XYZ (D65) matrix, e.g. from the
# sRGB spec / Bruce Lindbloom's reference tables. Used to sanity-check that
# build_rgb_to_xyz() derives matrices correctly from raw chromaticities
# rather than trusting the derivation blindly.
REFERENCE_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])


def test_srgb_matrix_matches_published_reference():
    # atol=5e-4, not tighter: different published sources round the D65
    # white point / final matrix slightly differently (the largest observed
    # element-wise difference here is ~2.3e-4, in the Z row), so a
    # from-first-principles derivation and a copied reference table agree
    # to about 3-4 decimal places, not bit-for-bit.
    m = prim.build_rgb_to_xyz(prim.SRGB_PRIMARIES)
    assert np.allclose(m, REFERENCE_SRGB_TO_XYZ, atol=5e-4)


def test_rgb_to_xyz_and_back_is_identity_for_each_space():
    rng = np.random.default_rng(0)
    linear_rgb = rng.random((50, 3))
    for space_primaries in [
        prim.SRGB_PRIMARIES, prim.REC709_PRIMARIES, prim.DISPLAY_P3_PRIMARIES,
        prim.ADOBE_RGB_PRIMARIES, prim.PROPHOTO_RGB_PRIMARIES,
    ]:
        to_xyz = prim.build_rgb_to_xyz(space_primaries)
        to_rgb = prim.build_xyz_to_rgb(space_primaries)
        xyz = linear_rgb @ to_xyz.T
        back = xyz @ to_rgb.T
        assert np.allclose(back, linear_rgb, atol=1e-9)


def test_white_point_maps_to_itself():
    """R=G=B=1 (linear) must map to exactly the space's own white point,
    by construction of build_rgb_to_xyz - this is the defining property
    the scaling step solves for."""
    for space in [spaces.SRGB, spaces.DISPLAY_P3, spaces.ADOBE_RGB, spaces.PROPHOTO_RGB]:
        m = prim.build_rgb_to_xyz(space.primaries)
        xyz_of_white = m @ np.array([1.0, 1.0, 1.0])
        assert np.allclose(xyz_of_white, space.white_xyz, atol=1e-9)


@pytest.mark.parametrize("space", [spaces.SRGB, spaces.REC709, spaces.DISPLAY_P3, spaces.ADOBE_RGB, spaces.PROPHOTO_RGB])
def test_encoded_rgb_round_trip_through_xyz(space):
    rng = np.random.default_rng(1)
    encoded = rng.random((30, 3))
    xyz = spaces.to_xyz(encoded, space)
    back = spaces.from_xyz(xyz, space)
    assert np.allclose(back, encoded, atol=1e-6)


def test_convert_same_space_is_identity():
    encoded = np.array([[0.2, 0.5, 0.8]])
    out = spaces.convert(encoded, spaces.SRGB, spaces.SRGB)
    assert np.allclose(out, encoded, atol=1e-10)


def test_convert_round_trip_between_different_white_points():
    """sRGB (D65) <-> ProPhoto RGB (D50): exercises the Bradford chromatic
    adaptation path, which only engages when white points differ."""
    rng = np.random.default_rng(2)
    encoded_srgb = rng.random((40, 3)) * 0.9 + 0.05  # avoid the extremes
    prophoto = spaces.convert(encoded_srgb, spaces.SRGB, spaces.PROPHOTO_RGB)
    back = spaces.convert(prophoto, spaces.PROPHOTO_RGB, spaces.SRGB)
    assert np.allclose(back, encoded_srgb, atol=1e-4)


def test_convert_srgb_to_display_p3_neutral_gray_stays_neutral():
    """A neutral gray has no color to be distorted by a gamut change -
    converting between two D65-native spaces should leave it exactly
    unchanged (same primaries' white, no adaptation needed)."""
    gray = np.array([[0.5, 0.5, 0.5]])
    out = spaces.convert(gray, spaces.SRGB, spaces.DISPLAY_P3)
    assert np.allclose(out, gray, atol=1e-6)


def test_get_color_space_by_name():
    assert spaces.get("sRGB") is spaces.SRGB
    with pytest.raises(ValueError):
        spaces.get("NotARealSpace")


# --- edge cases -----------------------------------------------------------

def test_pure_black_and_white_round_trip():
    for space in [spaces.SRGB, spaces.ADOBE_RGB, spaces.PROPHOTO_RGB]:
        black = np.array([[0.0, 0.0, 0.0]])
        white = np.array([[1.0, 1.0, 1.0]])
        assert np.allclose(spaces.from_xyz(spaces.to_xyz(black, space), space), black, atol=1e-8)
        assert np.allclose(spaces.from_xyz(spaces.to_xyz(white, space), space), white, atol=1e-5)


def test_out_of_gamut_values_do_not_crash_or_produce_nan():
    """Negative or >1 linear values (representing colors outside the
    space's gamut, which do occur transiently in a float pipeline) must
    still convert without raising or producing NaN/Inf."""
    weird = np.array([[-0.3, 1.8, 0.5], [2.0, -1.0, -0.5]])
    xyz = spaces.to_xyz(weird, spaces.SRGB)
    assert np.isfinite(xyz).all()
    back = spaces.from_xyz(xyz, spaces.SRGB)
    assert np.isfinite(back).all()


def test_nan_and_inf_input_do_not_raise():
    values = np.array([[np.nan, 0.5, 0.5], [np.inf, 0.5, 0.5], [-np.inf, 0.5, 0.5]])
    # Should not raise; NaN/Inf propagate through pure elementwise math,
    # which is expected/acceptable - the caller's render pipeline is
    # responsible for clipping before this point.
    xyz = spaces.to_xyz(values, spaces.SRGB)
    assert xyz.shape == values.shape
