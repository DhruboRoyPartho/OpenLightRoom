"""Tests for core/scopes/vectorscope.py."""

import numpy as np

from core.scopes.vectorscope import (
    compute_vectorscope, rgb_to_cb_cr, cb_cr_to_hue_angle_deg, SKIN_TONE_ANGLE_DEG,
)


def test_neutral_gray_maps_to_the_origin_regardless_of_brightness():
    for v in (0.0, 0.2, 0.5, 0.8, 1.0):
        gray = np.array([[v, v, v]], dtype=np.float32)
        cb, cr = rgb_to_cb_cr(gray)
        assert np.allclose(cb, 0.0, atol=1e-6)
        assert np.allclose(cr, 0.0, atol=1e-6)


def test_solid_neutral_image_concentrates_all_density_at_the_center():
    image = np.full((10, 10, 3), 0.4, dtype=np.float32)
    scope = compute_vectorscope(image, size=64)
    center = 64 // 2
    # Allow the two pixels nearest center for bin-edge rounding.
    window = scope[center - 1:center + 2, center - 1:center + 2]
    assert window.sum() == 100
    assert scope.sum() == 100


def test_primary_and_secondary_hues_land_at_distinct_angles():
    colors = {
        "red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0),
        "cyan": (0.0, 1.0, 1.0), "magenta": (1.0, 0.0, 1.0), "yellow": (1.0, 1.0, 0.0),
    }
    angles = {}
    for name, rgb in colors.items():
        pixel = np.array([rgb], dtype=np.float32)
        cb, cr = rgb_to_cb_cr(pixel)
        angles[name] = float(cb_cr_to_hue_angle_deg(cb, cr)[0])

    rounded = {name: round(angle) for name, angle in angles.items()}
    assert len(set(rounded.values())) == 6  # all six hues distinct
    for angle in angles.values():
        assert 0.0 <= angle < 360.0


def test_more_saturated_color_is_farther_from_center_than_less_saturated():
    muted_red = np.array([[0.6, 0.4, 0.4]], dtype=np.float32)
    vivid_red = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    cb_m, cr_m = rgb_to_cb_cr(muted_red)
    cb_v, cr_v = rgb_to_cb_cr(vivid_red)
    dist_muted = np.hypot(cb_m, cr_m)[0]
    dist_vivid = np.hypot(cb_v, cr_v)[0]
    assert dist_vivid > dist_muted


def test_skin_tone_angle_constant_is_a_valid_angle():
    assert 0.0 <= SKIN_TONE_ANGLE_DEG < 360.0


def test_output_shape_and_total_density():
    rng = np.random.default_rng(0)
    image = rng.random((6, 9, 3)).astype(np.float32)
    scope = compute_vectorscope(image, size=128)
    assert scope.shape == (128, 128)
    assert scope.sum() == 6 * 9


def test_nan_and_out_of_gamut_input_do_not_raise():
    weird = np.array([[[np.nan, 2.0, -1.0]]], dtype=np.float32)
    scope = compute_vectorscope(weird, size=16)
    assert np.isfinite(scope).all()
