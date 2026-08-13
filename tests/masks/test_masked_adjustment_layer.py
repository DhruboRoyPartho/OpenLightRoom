"""Tests for core/adjustment_layers/masked_adjustment_layer.py."""

import numpy as np
import pytest

from core.adjustment_layers.masked_adjustment_layer import MaskedAdjustmentLayer, ADJUSTMENT_FIELDS
from core.masking.mask import Mask, MaskComponent
from core.processing.color_space import srgb_to_linear, linear_to_srgb
from core.processing.exposure import adjust_exposure
from core.processing.temperature import adjust_temperature
from core.processing.tint import adjust_tint


def _full_mask():
    return Mask(components=[MaskComponent(kind="rectangle", params={"center_x": 0.5, "center_y": 0.5, "half_width": 1.0, "half_height": 1.0, "feather": 0.0})])


def _left_half_mask():
    return Mask(components=[MaskComponent(kind="rectangle", params={"center_x": 0.25, "center_y": 0.5, "half_width": 0.25, "half_height": 1.0, "feather": 0.0})])


def test_default_layer_is_identity():
    layer = MaskedAdjustmentLayer("Mask 1")
    assert layer.is_identity()
    assert str(layer) == "Mask 1"
    assert layer.label == "Mask 1"


def test_no_adjustments_short_circuits_even_with_a_mask():
    layer = MaskedAdjustmentLayer("Mask 1", mask=_full_mask())
    image = np.random.default_rng(0).random((10, 10, 3)).astype(np.float32)
    assert layer.apply(image) is image


def test_empty_mask_short_circuits_even_with_adjustments():
    layer = MaskedAdjustmentLayer("Mask 1", mask=Mask(), exposure=50.0)
    image = np.random.default_rng(0).random((10, 10, 3)).astype(np.float32)
    assert layer.apply(image) is image


def test_invisible_layer_is_a_no_op():
    layer = MaskedAdjustmentLayer("Mask 1", mask=_full_mask(), exposure=50.0, visible=False)
    image = np.full((10, 10, 3), 0.3, dtype=np.float32)
    out = layer.apply(image)
    assert np.array_equal(out, image)
    assert layer.is_identity()


def test_full_mask_exposure_brightens_the_whole_image():
    layer = MaskedAdjustmentLayer("Mask 1", mask=_full_mask(), exposure=40.0)
    image = np.full((10, 10, 3), 0.3, dtype=np.float32)
    out = layer.apply(image)
    assert out.mean() > image.mean()


def test_adjustment_only_affects_the_masked_region():
    layer = MaskedAdjustmentLayer("Mask 1", mask=_left_half_mask(), exposure=80.0)
    image = np.full((10, 10, 3), 0.3, dtype=np.float32)
    out = layer.apply(image)
    left = out[:, :3]     # inside the mask (center_x=0.25, half_width=0.25 -> spans cols ~[0,5))
    right = out[:, 7:]    # outside the mask
    assert left.mean() > image.mean()
    assert np.allclose(right, image[:, 7:], atol=1e-5)


def test_with_adjustment_returns_a_new_layer_without_mutating_the_original():
    layer = MaskedAdjustmentLayer("Mask 1")
    updated = layer.with_adjustment("exposure", 25.0)
    assert updated.exposure == 25.0
    assert layer.exposure == 0.0
    assert str(updated) == "Mask 1"


def test_with_adjustment_rejects_unknown_field():
    layer = MaskedAdjustmentLayer("Mask 1")
    with pytest.raises(ValueError):
        layer.with_adjustment("sharpness", 10.0)


def test_with_mask_replaces_only_the_mask():
    layer = MaskedAdjustmentLayer("Mask 1", exposure=10.0)
    new_mask = _full_mask()
    updated = layer.with_mask(new_mask)
    assert updated.mask is new_mask
    assert updated.exposure == 10.0


def test_with_label_does_not_change_pipeline_identity():
    layer = MaskedAdjustmentLayer("Mask 1")
    renamed = layer.with_label("Sky")
    assert renamed.label == "Sky"
    assert str(renamed) == "Mask 1"   # pipeline name is immutable


def test_with_visible_toggles_independently_of_adjustments():
    layer = MaskedAdjustmentLayer("Mask 1", mask=_full_mask(), exposure=30.0)
    hidden = layer.with_visible(False)
    assert hidden.is_identity()
    assert layer.visible is True   # original untouched


def test_all_documented_fields_are_settable_and_affect_output():
    """Every locally-supported adjustment actually does something -
    catches a field silently missing from apply()'s dispatch. Whites/
    Highlights and Blacks/Shadows are luminance-gated (see
    core/processing/whites.py etc - only engage near white/black
    respectively, exactly like their global counterparts), so each field
    is tested against a base image in the luma range it actually affects.
    """
    base_by_field = {
        "whites": (0.9, 0.9, 0.9), "highlights": (0.9, 0.9, 0.9),
        "blacks": (0.1, 0.1, 0.1), "shadows": (0.1, 0.1, 0.1),
        # Saturation/Hue are no-ops on a perfectly neutral gray (nothing
        # to scale/rotate) - use a mildly saturated color for those two.
        "saturation": (0.6, 0.4, 0.3), "hue": (0.6, 0.4, 0.3),
    }
    for field in ADJUSTMENT_FIELDS:
        image = np.full((20, 20, 3), base_by_field.get(field, (0.4, 0.4, 0.4)), dtype=np.float32)
        value = 1.3 if field == "contrast" else (90.0 if field == "hue" else 40.0)
        layer = MaskedAdjustmentLayer("Mask 1", mask=_full_mask(), **{field: value})
        out = layer.apply(image)
        assert not np.allclose(out, image, atol=1e-4), f"field {field!r} had no effect"


# --- Exposure/Temperature/Tint must round-trip through linear light ------
#
# These three are physically meaningful only in linear light (a stop is a
# doubling of *linear* radiance) - globally they run before the
# linear->display transform for exactly that reason (see
# core/pipeline/default_pipeline.py). A masked adjustment only ever sees
# the already display-referred (gamma-encoded) image, so apply() must
# undo the encoding, apply the physical operation, then re-encode -
# applying "a stop doubles the value" straight to gamma-encoded bytes
# clips highlights far too early and looks harsh/blown-out rather than a
# natural, photographic grade (the exact complaint this fixes).

def test_local_exposure_matches_the_physically_correct_linear_math():
    layer = MaskedAdjustmentLayer("Mask 1", mask=_full_mask(), exposure=40.0)
    image = np.full((4, 4, 3), 0.5, dtype=np.float32)
    out = layer.apply(image)

    expected = linear_to_srgb(np.clip(
        adjust_exposure(srgb_to_linear(image), 40.0), 0.0, 16.0))
    assert np.allclose(out, np.clip(expected, 0.0, 1.0), atol=1e-5)


def test_local_exposure_does_not_clip_as_aggressively_as_naive_display_space_math():
    """The regression this guards against: applying adjust_exposure()
    directly to gamma-encoded bytes (the pre-fix behavior) blows a
    mid-gray pixel to a raw value of 2.0 (instantly clipped to solid
    white) for a 2-stop push. The physically correct, linear-round-
    tripped result stays comfortably under 1.0 for the same push."""
    image = np.full((4, 4, 3), 0.5, dtype=np.float32)

    naive_display_space_result = adjust_exposure(image, 40.0)[0, 0, 0]
    assert naive_display_space_result >= 1.999  # the old, too-harsh behavior

    layer = MaskedAdjustmentLayer("Mask 1", mask=_full_mask(), exposure=40.0)
    fixed_result = layer.apply(image)[0, 0, 0]
    assert fixed_result < 0.95  # bright, but not slammed straight to white


def test_local_temperature_and_tint_match_the_physically_correct_linear_math():
    layer = MaskedAdjustmentLayer("Mask 1", mask=_full_mask(), temperature=40.0, tint=-30.0)
    image = np.full((4, 4, 3), 0.5, dtype=np.float32)
    out = layer.apply(image)

    linear = srgb_to_linear(image)
    linear = adjust_temperature(linear, 40.0)
    linear = adjust_tint(linear, -30.0)
    expected = np.clip(linear_to_srgb(np.clip(linear, 0.0, 16.0)), 0.0, 1.0)
    assert np.allclose(out, expected, atol=1e-5)


def test_result_is_bounded_and_finite():
    layer = MaskedAdjustmentLayer(
        "Mask 1", mask=_full_mask(),
        exposure=90.0, contrast=2.5, highlights=-80.0, shadows=80.0,
        whites=-90.0, blacks=90.0, temperature=90.0, tint=-90.0,
        saturation=-90.0, hue=170.0,
    )
    rng = np.random.default_rng(1)
    image = rng.random((16, 16, 3)).astype(np.float32) * 1.5 - 0.2
    out = layer.apply(image)
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
