"""Tests for core/scopes/histogram.py."""

import numpy as np

from core.scopes.histogram import compute_rgb_histogram, compute_luminance_histogram


def test_solid_color_produces_a_single_spike_per_channel():
    image = np.full((10, 10, 3), (0.2, 0.6, 0.9), dtype=np.float32)
    hist = compute_rgb_histogram(image, bins=256)

    for value, ch in zip((0.2, 0.6, 0.9), ("R", "G", "B")):
        counts = hist[ch]
        expected_bin = int(value * 256)
        assert counts[expected_bin] == 100
        assert counts.sum() == 100
        # every other bin is empty
        assert np.count_nonzero(counts) == 1


def test_histogram_counts_sum_to_pixel_count_for_random_image():
    rng = np.random.default_rng(0)
    image = rng.random((7, 13, 3)).astype(np.float32)
    hist = compute_rgb_histogram(image, bins=64)
    for ch in ("R", "G", "B"):
        assert hist[ch].sum() == 7 * 13


def test_luminance_histogram_spike_for_neutral_gray():
    gray = np.full((5, 5, 3), 0.5, dtype=np.float32)
    counts = compute_luminance_histogram(gray, bins=256)
    assert counts[128] == 25
    assert counts.sum() == 25


def test_black_and_white_are_bounded_and_land_at_the_range_edges():
    black = np.zeros((3, 3, 3), dtype=np.float32)
    white = np.ones((3, 3, 3), dtype=np.float32)
    hist_black = compute_rgb_histogram(black)
    hist_white = compute_rgb_histogram(white)
    for ch in ("R", "G", "B"):
        assert hist_black[ch][0] == 9
        assert hist_white[ch][-1] == 9


def test_out_of_gamut_and_nan_input_do_not_raise_and_stay_bounded():
    weird = np.array([[[1.8, -0.4, np.nan], [np.inf, -np.inf, 0.5]]], dtype=np.float32)
    hist = compute_rgb_histogram(weird)
    for ch in ("R", "G", "B"):
        assert hist[ch].sum() == 2
        assert hist[ch].shape == (256,)
    luma_hist = compute_luminance_histogram(weird)
    assert luma_hist.sum() == 2
