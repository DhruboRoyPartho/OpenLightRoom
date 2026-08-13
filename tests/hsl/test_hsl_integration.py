"""Integration tests: HSLLayer wired into ImageDocument.render() (via the
default pipeline) and round-tripped through project_io serialization."""

import numpy as np

from core.image_model.image_document import ImageDocument
from core.adjustment_layers.hsl_layer import HSLLayer
from core.io.project_io import serialize_layer


def test_hsl_layer_is_applied_by_image_document_render():
    base = np.full((4, 4, 3), (0.8, 0.2, 0.2), dtype=np.float32)
    doc = ImageDocument(base)

    without_hsl = doc.render()

    doc.add_layer(HSLLayer(saturation={"Red": 80}))
    with_hsl = doc.render()

    assert not np.allclose(without_hsl, with_hsl)


def test_hsl_registered_in_default_pipeline_display_stage():
    doc = ImageDocument(np.zeros((2, 2, 3), dtype=np.float32))
    assert "HSL" in doc.pipeline.stage("Display").layer_order


def test_serialize_round_trip_via_project_io():
    layer = HSLLayer(hue={"Red": 10}, saturation={"Blue": -20}, luminance={"Green": 30})
    entry = serialize_layer(layer)
    assert entry["type"] == "HSL"
    assert entry["hue"] == {"Red": 10}
    assert entry["saturation"] == {"Blue": -20}
    assert entry["luminance"] == {"Green": 30}

    restored = HSLLayer(hue=entry["hue"], saturation=entry["saturation"], luminance=entry["luminance"])
    assert restored.hue == layer.hue
    assert restored.saturation == layer.saturation
    assert restored.luminance == layer.luminance
