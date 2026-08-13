"""Tests for core/adjustment_layers/color_wheels_layer.py."""

import numpy as np
import pytest

from core.adjustment_layers.color_wheels_layer import ColorWheelsLayer


def test_default_layer_is_identity():
    layer = ColorWheelsLayer()
    assert layer.is_identity()
    assert str(layer) == "Color Wheels"


def test_with_wheel_updates_only_the_targeted_zone():
    layer = ColorWheelsLayer()
    updated = layer.with_wheel("shadows", hue_deg=200.0, chroma=40.0)
    assert updated.shadows["hue_deg"] == 200.0
    assert updated.shadows["chroma"] == 40.0
    assert updated.midtones == layer.midtones
    assert updated.highlights == layer.highlights
    assert updated.global_ == layer.global_
    assert not updated.is_identity()
    assert layer.is_identity()  # original untouched


def test_with_wheel_wraps_hue_into_0_360():
    layer = ColorWheelsLayer()
    updated = layer.with_wheel("global", hue_deg=400.0, chroma=10.0)
    assert updated.global_["hue_deg"] == 40.0


def test_with_wheel_rejects_unknown_zone():
    layer = ColorWheelsLayer()
    with pytest.raises(ValueError):
        layer.with_wheel("nonexistent", chroma=10.0)


def test_with_wheel_only_changes_passed_fields():
    layer = ColorWheelsLayer(shadows={"hue_deg": 10.0, "chroma": 20.0, "luminance": 5.0})
    updated = layer.with_wheel("shadows", chroma=30.0)
    assert updated.shadows["hue_deg"] == 10.0
    assert updated.shadows["chroma"] == 30.0
    assert updated.shadows["luminance"] == 5.0


def test_apply_is_identity_when_default_and_delegates_when_not():
    image = np.random.default_rng(0).random((3, 3, 3)).astype(np.float32)
    layer = ColorWheelsLayer()
    assert layer.apply(image) is image

    active = layer.with_wheel("global", hue_deg=100.0, chroma=50.0)
    assert not np.allclose(active.apply(image), image)
