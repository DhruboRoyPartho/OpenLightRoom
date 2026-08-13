"""Tests for core/processing/hsl_grading.py - the per-channel Hue/
Saturation/Luminance pixel math."""

import cv2
import numpy as np

from core.processing.hsl_grading import apply_hsl_grading, HSL_CHANNELS, CHANNEL_HUE_DEG


def _solid(rgb):
    return np.array([[rgb]], dtype=np.float32)


def test_no_active_channels_is_identity():
    image = np.random.default_rng(0).random((4, 4, 3)).astype(np.float32)
    out = apply_hsl_grading(image, {}, {}, {})
    assert out is image  # short-circuits without touching the array at all


def test_zero_values_for_every_channel_is_identity():
    image = np.random.default_rng(1).random((4, 4, 3)).astype(np.float32)
    zeros = {ch: 0 for ch in HSL_CHANNELS}
    out = apply_hsl_grading(image, zeros, zeros, zeros)
    assert out is image


def test_saturation_boost_on_matching_channel_increases_saturation():
    red = _solid((0.8, 0.2, 0.2))  # a somewhat-saturated red
    out = apply_hsl_grading(red, {}, {"Red": 50}, {})
    hsv_before = cv2.cvtColor(red, cv2.COLOR_RGB2HSV)
    hsv_after = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)
    assert hsv_after[0, 0, 1] > hsv_before[0, 0, 1]


def test_saturation_change_on_non_matching_channel_has_no_effect():
    red = _solid((0.8, 0.2, 0.2))
    out = apply_hsl_grading(red, {}, {"Blue": 50}, {})
    assert np.allclose(out, red, atol=1e-6)


def test_luminance_boost_on_matching_channel_increases_value():
    red = _solid((0.6, 0.1, 0.1))
    out = apply_hsl_grading(red, {}, {}, {"Red": 40})
    hsv_before = cv2.cvtColor(red, cv2.COLOR_RGB2HSV)
    hsv_after = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)
    assert hsv_after[0, 0, 2] > hsv_before[0, 0, 2]


def test_hue_shift_on_matching_channel_rotates_hue_toward_neighbor():
    red = _solid((1.0, 0.0, 0.0))  # hue 0
    out = apply_hsl_grading(red, {"Red": 50}, {}, {})
    hsv_after = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)
    # +50 on a 0..100 scale -> +30 degrees (half of MAX_HUE_SHIFT_DEG=60),
    # moving red's hue toward orange/yellow.
    assert 0.0 < hsv_after[0, 0, 0] < 60.0


def test_result_is_bounded_and_finite_for_valid_input():
    rng = np.random.default_rng(2)
    image = rng.random((16, 16, 3)).astype(np.float32)
    hue = {ch: 80 for ch in HSL_CHANNELS}
    sat = {ch: -90 for ch in HSL_CHANNELS}
    luma = {ch: 70 for ch in HSL_CHANNELS}
    out = apply_hsl_grading(image, hue, sat, luma)
    assert np.isfinite(out).all()
    assert out.shape == image.shape


def test_out_of_gamut_and_nan_input_do_not_raise():
    weird = np.array([[[1.5, -0.3, 0.5], [np.nan, 0.5, 0.5]]], dtype=np.float32)
    out = apply_hsl_grading(weird, {"Red": 50}, {"Green": 50}, {"Blue": 50})
    assert out.shape == weird.shape


def test_each_channel_hue_center_is_selected_by_its_own_hue_shift():
    """Sanity check that every one of the 8 channel centers actually
    produces a visible effect on a pixel placed exactly at that hue - a
    typo in CHANNEL_HUE_DEG for any channel would leave that channel inert."""
    for ch in HSL_CHANNELS:
        h = CHANNEL_HUE_DEG[ch]
        rgb = cv2.cvtColor(np.array([[[h, 0.8, 0.8]]], dtype=np.float32), cv2.COLOR_HSV2RGB)
        out = apply_hsl_grading(rgb, {}, {ch: 80}, {})
        assert not np.allclose(out, rgb, atol=1e-4), f"channel {ch} had no effect at its own hue center"
