"""Tests for core/processing/color_wheels.py."""

import numpy as np

from core.processing.color_wheels import apply_color_wheels, DEFAULT_WHEEL
from core.color_science.oklab import linear_srgb_to_oklab
from core.processing.color_space import srgb_to_linear


def _empty():
    return dict(DEFAULT_WHEEL)


def _mid_gray_image():
    return np.full((4, 4, 3), 0.5, dtype=np.float32)


def test_all_empty_wheels_is_identity():
    image = np.random.default_rng(0).random((4, 4, 3)).astype(np.float32)
    out = apply_color_wheels(image, _empty(), _empty(), _empty(), _empty())
    assert out is image


def test_global_wheel_affects_black_and_white_alike():
    """Global should push every pixel regardless of its tonal zone -
    unlike shadows/midtones/highlights, which only affect their own zone."""
    image = np.array([[[0.02, 0.02, 0.02], [0.98, 0.98, 0.98]]], dtype=np.float32)
    global_wheel = {"hue_deg": 30.0, "chroma": 50.0, "luminance": 0.0}
    out = apply_color_wheels(image, _empty(), _empty(), _empty(), global_wheel)
    assert not np.allclose(out[0, 0], image[0, 0], atol=1e-4)
    assert not np.allclose(out[0, 1], image[0, 1], atol=1e-4)


def test_shadows_wheel_affects_dark_pixel_not_bright_pixel():
    image = np.array([[[0.02, 0.02, 0.02], [0.98, 0.98, 0.98]]], dtype=np.float32)
    shadows = {"hue_deg": 200.0, "chroma": 80.0, "luminance": 0.0}
    out = apply_color_wheels(image, shadows, _empty(), _empty(), _empty())
    assert not np.allclose(out[0, 0], image[0, 0], atol=1e-4)
    assert np.allclose(out[0, 1], image[0, 1], atol=1e-3)


def test_highlights_wheel_affects_bright_pixel_not_dark_pixel():
    image = np.array([[[0.02, 0.02, 0.02], [0.98, 0.98, 0.98]]], dtype=np.float32)
    highlights = {"hue_deg": 40.0, "chroma": 80.0, "luminance": 0.0}
    out = apply_color_wheels(image, _empty(), _empty(), highlights, _empty())
    assert np.allclose(out[0, 0], image[0, 0], atol=1e-3)
    assert not np.allclose(out[0, 1], image[0, 1], atol=1e-4)


def test_luminance_only_shifts_lightness_without_needing_chroma():
    image = _mid_gray_image()
    global_wheel = {"hue_deg": 0.0, "chroma": 0.0, "luminance": 60.0}
    out = apply_color_wheels(image, _empty(), _empty(), _empty(), global_wheel)
    linear_before = srgb_to_linear(image)
    linear_after = srgb_to_linear(out)
    L_before = linear_srgb_to_oklab(linear_before)[..., 0]
    L_after = linear_srgb_to_oklab(linear_after)[..., 0]
    assert np.all(L_after > L_before)


def test_zero_chroma_zero_luminance_wheel_is_a_no_op_for_that_zone():
    image = _mid_gray_image()
    out = apply_color_wheels(image, _empty(), {"hue_deg": 90.0, "chroma": 0.0, "luminance": 0.0}, _empty(), _empty())
    assert np.allclose(out, image, atol=1e-6)


def test_result_bounded_and_finite_for_extreme_settings():
    rng = np.random.default_rng(1)
    image = rng.random((16, 16, 3)).astype(np.float32)
    wheel = {"hue_deg": 123.0, "chroma": 100.0, "luminance": 100.0}
    out = apply_color_wheels(image, wheel, wheel, wheel, wheel)
    assert np.isfinite(out).all()
    assert out.shape == image.shape


def test_out_of_gamut_and_nan_input_do_not_raise():
    weird = np.array([[[1.8, -0.4, 0.5], [np.nan, 0.5, 0.5]]], dtype=np.float32)
    wheel = {"hue_deg": 10.0, "chroma": 50.0, "luminance": 20.0}
    out = apply_color_wheels(weird, wheel, wheel, wheel, wheel)
    assert out.shape == weird.shape
