"""Tests for core/processing/white_balance.py."""

import numpy as np
import pytest

from core.processing.white_balance import (
    estimate_gray_world_white_balance,
    estimate_white_balance_from_sample,
    sample_scene_linear_pixel,
)
from core.processing.temperature import adjust_temperature
from core.processing.tint import adjust_tint


def _apply_wb(image, temp, tint):
    return adjust_tint(adjust_temperature(image, temp), tint)


def test_already_neutral_image_needs_no_correction():
    image = np.full((8, 8, 3), 0.4, dtype=np.float32)
    temp, tint = estimate_gray_world_white_balance(image)
    assert abs(temp) < 1e-3
    assert abs(tint) < 1e-3


def test_gray_world_correction_actually_neutralizes_a_color_cast():
    """The whole point: apply a known cast, recover a correction, verify
    applying it makes the channels equal again."""
    rng = np.random.default_rng(0)
    neutral_scene = rng.random((32, 32, 3)).astype(np.float32) * 0.5 + 0.1
    # Give it a strong warm (red/yellow) cast, as if shot under tungsten light.
    cast = neutral_scene.copy()
    cast[..., 0] *= 1.6   # more red
    cast[..., 2] *= 0.6   # less blue

    temp, tint = estimate_gray_world_white_balance(cast)
    corrected = _apply_wb(cast, temp, tint)

    avg_r = corrected[..., 0].mean()
    avg_g = corrected[..., 1].mean()
    avg_b = corrected[..., 2].mean()
    assert np.isclose(avg_r, avg_g, rtol=0.02)
    assert np.isclose(avg_g, avg_b, rtol=0.02)


def test_green_tint_cast_is_corrected_by_tint_alone():
    rng = np.random.default_rng(1)
    neutral_scene = rng.random((16, 16, 3)).astype(np.float32) * 0.4 + 0.2
    cast = neutral_scene.copy()
    cast[..., 1] *= 1.5  # green cast (e.g. fluorescent lighting)

    temp, tint = estimate_gray_world_white_balance(cast)
    corrected = _apply_wb(cast, temp, tint)
    avg_r = corrected[..., 0].mean()
    avg_g = corrected[..., 1].mean()
    avg_b = corrected[..., 2].mean()
    assert np.isclose(avg_r, avg_g, rtol=0.02)
    assert np.isclose(avg_g, avg_b, rtol=0.02)


def test_from_sample_matches_gray_world_for_a_flat_image():
    """A perfectly flat-colored image's gray-world average is exactly its
    one color, so both entry points must agree exactly."""
    image = np.full((4, 4, 3), (0.7, 0.5, 0.3), dtype=np.float32)
    temp_gw, tint_gw = estimate_gray_world_white_balance(image)
    temp_sample, tint_sample = estimate_white_balance_from_sample(0.7, 0.5, 0.3)
    assert np.isclose(temp_gw, temp_sample)
    assert np.isclose(tint_gw, tint_sample)


def test_from_sample_correction_neutralizes_that_exact_pixel():
    r, g, b = 0.8, 0.4, 0.2
    temp, tint = estimate_white_balance_from_sample(r, g, b)
    pixel = np.array([[[r, g, b]]], dtype=np.float32)
    corrected = _apply_wb(pixel, temp, tint)
    assert np.allclose(corrected[0, 0, 0], corrected[0, 0, 1], atol=1e-4)
    assert np.allclose(corrected[0, 0, 1], corrected[0, 0, 2], atol=1e-4)


def test_result_is_bounded_to_slider_range():
    # An extreme, physically-implausible cast should still clamp into
    # [-100, 100] rather than returning an out-of-range value.
    extreme = np.array([[[10.0, 0.001, 0.00001]]], dtype=np.float32)
    temp, tint = estimate_gray_world_white_balance(extreme)
    assert -100.0 <= temp <= 100.0
    assert -100.0 <= tint <= 100.0


def test_black_image_does_not_raise_or_produce_nan():
    black = np.zeros((4, 4, 3), dtype=np.float32)
    temp, tint = estimate_gray_world_white_balance(black)
    assert np.isfinite(temp) and np.isfinite(tint)


def test_negative_and_nan_input_do_not_raise():
    weird = np.array([[[np.nan, -0.5, 0.3], [np.inf, 0.2, -np.inf]]], dtype=np.float32)
    temp, tint = estimate_gray_world_white_balance(weird)
    # May legitimately be NaN if the average itself is NaN (garbage in,
    # garbage out) - the requirement is "does not raise", checked simply by
    # reaching this line.
    assert True


class _FakeCropLayer:
    def __str__(self):
        return "Crop"

    def apply(self, image):
        return image[2:6, 2:6]  # arbitrary crop for the test


class _FakeDocument:
    def __init__(self, base_image, layers):
        self.base_image = base_image
        self.layers = layers


def test_sample_scene_linear_pixel_without_crop_reads_base_image_directly():
    base = np.zeros((10, 10, 3), dtype=np.float32)
    base[5, 5] = (0.1, 0.2, 0.3)
    doc = _FakeDocument(base, layers=[])
    r, g, b = sample_scene_linear_pixel(doc, 5, 5)
    assert (r, g, b) == pytest.approx((0.1, 0.2, 0.3), abs=1e-6)


def test_sample_scene_linear_pixel_applies_crop_layer_first():
    base = np.zeros((10, 10, 3), dtype=np.float32)
    base[2 + 1, 2 + 1] = (0.4, 0.5, 0.6)  # lands at (1,1) in the cropped [2:6,2:6] view
    doc = _FakeDocument(base, layers=[_FakeCropLayer()])
    r, g, b = sample_scene_linear_pixel(doc, 1, 1)
    assert (r, g, b) == pytest.approx((0.4, 0.5, 0.6), abs=1e-6)


def test_sample_scene_linear_pixel_clamps_out_of_bounds_coordinates():
    base = np.full((4, 4, 3), (0.9, 0.9, 0.9), dtype=np.float32)
    doc = _FakeDocument(base, layers=[])
    r, g, b = sample_scene_linear_pixel(doc, 999, -50)
    assert (r, g, b) == pytest.approx((0.9, 0.9, 0.9), abs=1e-6)
