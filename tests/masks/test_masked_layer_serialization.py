"""Tests for project_io serialize_layer/deserialize_layer support for
MaskedAdjustmentLayer, and preset_io's exclusion of mask layers."""

import json

from core.io.project_io import serialize_layer, deserialize_layer, serialize_layers, deserialize_layers
from core.io.preset_io import layers_for_preset
from core.adjustment_layers.masked_adjustment_layer import MaskedAdjustmentLayer
from core.masking.mask import Mask, MaskComponent


def _sample_layer():
    mask = Mask(
        components=[
            MaskComponent(kind="radial", params={"center_x": 0.3, "center_y": 0.4, "radius_x": 0.2, "radius_y": 0.25, "feather": 40.0}),
            MaskComponent(kind="luminance_range", params={"low": 0.2, "high": 0.9, "feather": 15.0}, op="intersect"),
        ],
        feather=10.0, blur=5.0, density=85.0, invert=True,
    )
    return MaskedAdjustmentLayer(
        "Mask 2", mask=mask, label="Sky", visible=True,
        exposure=-20.0, contrast=1.1, highlights=-30.0, shadows=15.0,
        whites=0.0, blacks=0.0, temperature=10.0, tint=-5.0,
        saturation=20.0, hue=0.0,
    )


def test_serialize_produces_a_json_safe_dict():
    entry = _sample_layer()
    data = serialize_layer(entry)
    json.dumps(data)  # must not raise - every value has to be JSON-serializable
    assert data["type"] == "Mask 2"
    assert data["label"] == "Sky"
    assert data["visible"] is True
    assert data["exposure"] == -20.0
    assert data["mask"]["feather"] == 10.0
    assert data["mask"]["invert"] is True
    assert len(data["mask"]["components"]) == 2
    assert data["mask"]["components"][0]["kind"] == "radial"
    assert data["mask"]["components"][1]["op"] == "intersect"


def test_round_trip_preserves_everything():
    original = _sample_layer()
    entry = serialize_layer(original)
    restored = deserialize_layer(entry)

    assert str(restored) == "Mask 2"
    assert restored.label == "Sky"
    assert restored.visible is True
    assert restored.exposure == -20.0
    assert restored.contrast == 1.1
    assert restored.temperature == 10.0
    assert restored.saturation == 20.0

    assert restored.mask.feather == 10.0
    assert restored.mask.blur == 5.0
    assert restored.mask.density == 85.0
    assert restored.mask.invert is True
    assert len(restored.mask.components) == 2
    assert restored.mask.components[0].kind == "radial"
    assert restored.mask.components[0].params["center_x"] == 0.3
    assert restored.mask.components[1].op == "intersect"


def test_round_trip_through_a_full_json_dump_and_load(tmp_path):
    original = _sample_layer()
    path = tmp_path / "layer.json"
    path.write_text(json.dumps(serialize_layer(original)))

    entry = json.loads(path.read_text())
    restored = deserialize_layer(entry)

    assert str(restored) == "Mask 2"
    assert restored.mask.components[0].params["center_x"] == 0.3


def test_multiple_mask_layers_round_trip_via_serialize_layers():
    layer1 = MaskedAdjustmentLayer("Mask 1", mask=Mask([MaskComponent(kind="ellipse")]), exposure=10.0)
    layer2 = MaskedAdjustmentLayer("Mask 2", mask=Mask([MaskComponent(kind="rectangle")]), contrast=1.2)
    entries = serialize_layers([layer1, layer2])
    restored = deserialize_layers(entries)

    assert [str(l) for l in restored] == ["Mask 1", "Mask 2"]
    assert restored[0].exposure == 10.0
    assert restored[1].contrast == 1.2


def test_brush_stroke_points_round_trip_through_json():
    """Tuples become lists through a JSON round trip - confirms the mask
    evaluators tolerate that (they only ever unpack points, never require
    an exact tuple type)."""
    mask = Mask(components=[MaskComponent(
        kind="brush",
        params={"strokes": [{"points": [(0.1, 0.2), (0.3, 0.4)], "radius": 0.05, "hardness": 80, "flow": 100, "mode": "add"}]},
    )])
    layer = MaskedAdjustmentLayer("Mask 1", mask=mask, exposure=10.0)
    entry = json.loads(json.dumps(serialize_layer(layer)))
    restored = deserialize_layer(entry)

    points = restored.mask.components[0].params["strokes"][0]["points"]
    assert points[0] == [0.1, 0.2]  # JSON round trip: tuple -> list

    import numpy as np
    image = np.full((20, 20, 3), 0.3, dtype=np.float32)
    out = restored.apply(image)  # must not raise on list-of-lists points
    assert np.isfinite(out).all()


def test_mask_layer_with_missing_mask_data_defaults_to_an_empty_mask():
    entry = {"type": "Mask 7"}  # missing "mask" entirely - a hand-edited/corrupt file
    restored = deserialize_layer(entry)
    assert str(restored) == "Mask 7"
    assert restored.mask.is_empty()


def test_layers_for_preset_excludes_mask_layers():
    mask_layer = MaskedAdjustmentLayer("Mask 1", exposure=10.0)  # excluded regardless of content

    from core.adjustment_layers.exposure_layer import ExposureLayer
    layers = [ExposureLayer(20.0), mask_layer]
    filtered = layers_for_preset(layers)

    names = {str(l) for l in filtered}
    assert names == {"Exposure"}
