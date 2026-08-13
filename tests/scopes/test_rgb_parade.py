"""Tests for core/scopes/rgb_parade.py."""

import numpy as np

from core.scopes.rgb_parade import compute_rgb_parade


def test_solid_red_image_lights_up_only_the_red_channel_at_full_value():
    image = np.zeros((10, 10, 3), dtype=np.float32)
    image[..., 0] = 1.0  # pure red
    parade = compute_rgb_parade(image, out_width=10, out_height=50)

    assert parade["R"][0, :].sum() == 10 * 10   # row 0 = value 1.0 (top)
    assert parade["G"][-1, :].sum() == 10 * 10  # row -1 = value 0.0 (bottom)
    assert parade["B"][-1, :].sum() == 10 * 10


def test_each_channel_totals_the_full_pixel_count():
    rng = np.random.default_rng(0)
    image = rng.random((8, 12, 3)).astype(np.float32)
    parade = compute_rgb_parade(image, out_width=32, out_height=32)
    for ch in ("R", "G", "B"):
        assert parade[ch].sum() == 8 * 12


def test_output_shapes_match_requested_dimensions():
    image = np.zeros((4, 4, 3), dtype=np.float32)
    parade = compute_rgb_parade(image, out_width=16, out_height=8)
    for ch in ("R", "G", "B"):
        assert parade[ch].shape == (8, 16)


def test_nan_and_out_of_gamut_input_do_not_raise():
    weird = np.array([[[np.nan, 2.0, -1.0]]], dtype=np.float32)
    parade = compute_rgb_parade(weird, out_width=4, out_height=4)
    for ch in ("R", "G", "B"):
        assert np.isfinite(parade[ch]).all()
