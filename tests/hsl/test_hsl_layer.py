"""Tests for core/adjustment_layers/hsl_layer.py."""

import numpy as np
import pytest

from core.adjustment_layers.hsl_layer import HSLLayer


def test_default_layer_is_identity():
    layer = HSLLayer()
    assert layer.is_identity()
    assert str(layer) == "HSL"


def test_constructor_drops_zero_entries():
    layer = HSLLayer(hue={"Red": 0, "Blue": 20}, saturation={"Green": 0})
    assert layer.hue == {"Blue": 20}
    assert layer.saturation == {}
    assert not layer.is_identity()


def test_with_value_sets_and_clears_a_channel():
    layer = HSLLayer()
    with_red = layer.with_value("hue", "Red", 40)
    assert with_red.hue == {"Red": 40}
    assert layer.hue == {}  # original untouched (immutable-style update)

    cleared = with_red.with_value("hue", "Red", 0)
    assert cleared.hue == {}
    assert cleared.is_identity()


def test_with_value_rejects_unknown_axis_or_channel():
    layer = HSLLayer()
    with pytest.raises(ValueError):
        layer.with_value("brightness", "Red", 10)
    with pytest.raises(ValueError):
        layer.with_value("hue", "Chartreuse", 10)


def test_apply_delegates_to_processing_function_and_is_identity_when_empty():
    image = np.random.default_rng(0).random((3, 3, 3)).astype(np.float32)
    layer = HSLLayer()
    out = layer.apply(image)
    assert out is image


def test_apply_with_values_changes_the_image():
    image = np.full((2, 2, 3), (0.8, 0.2, 0.2), dtype=np.float32)
    layer = HSLLayer(saturation={"Red": 60})
    out = layer.apply(image)
    assert not np.allclose(out, image)
