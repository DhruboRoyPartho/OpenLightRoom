"""Tests for core/masking/luminance_range.py's three_zone_luminance_masks
- the shadows/midtones/highlights partition-of-unity used for tonal
separation."""

import numpy as np

from core.masking.luminance_range import three_zone_luminance_masks


def test_zones_sum_to_one_everywhere():
    luma = np.linspace(0.0, 1.0, 500)
    shadows, midtones, highlights = three_zone_luminance_masks(luma, shadow_edge=0.33, highlight_edge=0.66, feather=0.1)
    total = shadows + midtones + highlights
    assert np.allclose(total, 1.0, atol=1e-6)


def test_pure_black_is_entirely_shadows():
    shadows, midtones, highlights = three_zone_luminance_masks(np.array([0.0]))
    assert shadows[0] > 0.99
    assert midtones[0] < 0.01
    assert highlights[0] < 0.01


def test_pure_white_is_entirely_highlights():
    shadows, midtones, highlights = three_zone_luminance_masks(np.array([1.0]))
    assert highlights[0] > 0.99
    assert shadows[0] < 0.01
    assert midtones[0] < 0.01


def test_middle_gray_is_mostly_midtones():
    shadows, midtones, highlights = three_zone_luminance_masks(
        np.array([0.5]), shadow_edge=0.33, highlight_edge=0.66, feather=0.1
    )
    assert midtones[0] > 0.9
    assert shadows[0] < 0.1
    assert highlights[0] < 0.1


def test_all_zones_stay_within_zero_one_including_edge_cases():
    luma = np.array([-1.0, 0.0, 0.33, 0.5, 0.66, 1.0, 2.0, np.nan])
    shadows, midtones, highlights = three_zone_luminance_masks(luma)
    for zone in (shadows, midtones, highlights):
        finite = np.isfinite(zone)
        assert np.all(zone[finite] >= 0.0) and np.all(zone[finite] <= 1.0)
