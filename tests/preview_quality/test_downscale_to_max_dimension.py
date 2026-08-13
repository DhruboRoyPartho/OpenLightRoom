"""Tests for core/processing/geometry.py:downscale_to_max_dimension - the
core operation behind the interactive preview-quality setting."""

import numpy as np

from core.processing.geometry import downscale_to_max_dimension


def test_none_max_dimension_returns_the_same_object_unchanged():
    image = np.random.default_rng(0).random((100, 200, 3)).astype(np.float32)
    out = downscale_to_max_dimension(image, None)
    assert out is image


def test_image_already_within_the_cap_is_returned_unchanged():
    image = np.random.default_rng(0).random((100, 200, 3)).astype(np.float32)
    out = downscale_to_max_dimension(image, 200)
    assert out is image
    out2 = downscale_to_max_dimension(image, 500)
    assert out2 is image


def test_downscales_the_longer_side_to_the_cap_preserving_aspect_ratio():
    image = np.random.default_rng(0).random((1000, 2000, 3)).astype(np.float32)
    out = downscale_to_max_dimension(image, 500)
    assert out.shape[1] == 500          # width was the longer side
    assert abs(out.shape[0] - 250) <= 1  # height scaled by the same factor


def test_downscales_a_tall_image_correctly():
    image = np.random.default_rng(0).random((2000, 1000, 3)).astype(np.float32)
    out = downscale_to_max_dimension(image, 500)
    assert out.shape[0] == 500
    assert abs(out.shape[1] - 250) <= 1


def test_never_upscales():
    image = np.random.default_rng(0).random((50, 80, 3)).astype(np.float32)
    out = downscale_to_max_dimension(image, 4000)
    assert out.shape == image.shape


def test_output_stays_in_valid_range_and_is_finite():
    image = np.random.default_rng(0).random((400, 400, 3)).astype(np.float32)
    out = downscale_to_max_dimension(image, 100)
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_channels_preserved():
    image = np.random.default_rng(0).random((400, 600, 3)).astype(np.float32)
    out = downscale_to_max_dimension(image, 150)
    assert out.shape[2] == 3


def test_dtype_preserved():
    image = np.random.default_rng(0).random((400, 600, 3)).astype(np.float32)
    out = downscale_to_max_dimension(image, 150)
    assert out.dtype == np.float32
