"""Integration tests: ColorWheelsLayer wired into ImageDocument.render()
and round-tripped through project_io serialization."""

import numpy as np

from core.image_model.image_document import ImageDocument
from core.adjustment_layers.color_wheels_layer import ColorWheelsLayer
from core.io.project_io import serialize_layer


def test_color_wheels_layer_is_applied_by_image_document_render():
    base = np.full((4, 4, 3), 0.5, dtype=np.float32)
    doc = ImageDocument(base)

    without = doc.render()

    doc.add_layer(ColorWheelsLayer(global_={"hue_deg": 30.0, "chroma": 60.0, "luminance": 0.0}))
    with_wheels = doc.render()

    assert not np.allclose(without, with_wheels)


def test_color_wheels_registered_in_default_pipeline_display_stage():
    doc = ImageDocument(np.zeros((2, 2, 3), dtype=np.float32))
    assert "Color Wheels" in doc.pipeline.stage("Display").layer_order


def test_serialize_round_trip_via_project_io():
    layer = ColorWheelsLayer(
        shadows={"hue_deg": 200.0, "chroma": 20.0, "luminance": -10.0},
        highlights={"hue_deg": 40.0, "chroma": 15.0, "luminance": 5.0},
    )
    entry = serialize_layer(layer)
    assert entry["type"] == "Color Wheels"
    assert entry["shadows"] == layer.shadows
    assert entry["highlights"] == layer.highlights
    assert entry["midtones"] == layer.midtones
    assert entry["global"] == layer.global_

    restored = ColorWheelsLayer(
        shadows=entry["shadows"], midtones=entry["midtones"],
        highlights=entry["highlights"], global_=entry["global"],
    )
    assert restored.shadows == layer.shadows
    assert restored.highlights == layer.highlights
