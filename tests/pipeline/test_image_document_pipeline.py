"""Regression tests that ImageDocument.render() still composes layers in
the same linear-then-display order after the hardcoded LINEAR_STAGE_ORDER /
DISPLAY_STAGE_ORDER lists were replaced by a Pipeline of Stages, and that
the pipeline can be extended by registering a new layer name (the point of
the refactor) without editing ImageDocument itself."""

import numpy as np

from core.image_model.image_document import ImageDocument
from core.adjustment_layers.exposure_layer import ExposureLayer
from core.adjustment_layers.brightness_layer import BrightnessLayer
from core.adjustment_layers.geometry_layer import GeometryLayer
from core.processing.color_space import linear_to_display
from core.processing.exposure import adjust_exposure
from core.processing.brightness import adjust_brightness


def _base_image():
    rng = np.random.default_rng(0)
    return rng.random((4, 4, 3), dtype=np.float32)


def test_exposure_runs_in_linear_light_before_brightness_runs_in_display_space():
    """Exposure (a linear-stage tool) and Brightness (a display-stage tool)
    must compose in that order: exposure multiplies scene-linear light,
    then the image is gamma-encoded, then brightness's display-space
    weighting is applied. Verified by reproducing that exact sequence by
    hand with the underlying processing functions and comparing pixel-for-
    pixel against ImageDocument.render()."""
    base = _base_image()
    doc = ImageDocument(base)
    doc.add_layer(ExposureLayer(20.0))   # +1 stop
    doc.add_layer(BrightnessLayer(0.2))

    expected = adjust_exposure(base.copy(), 20.0)
    expected = linear_to_display(expected)
    expected = adjust_brightness(expected, 0.2)
    expected = np.clip(expected, 0.0, 1.0).astype(np.float32)

    actual = doc.render()
    assert np.allclose(actual, expected, atol=1e-6)


def test_wrong_order_would_have_produced_a_different_result():
    """Sanity check that the two orderings actually diverge for this input
    (otherwise the previous test could pass by coincidence)."""
    base = _base_image()
    correct = adjust_exposure(base.copy(), 20.0)
    correct = linear_to_display(correct)
    correct = adjust_brightness(correct, 0.2)

    wrong = adjust_brightness(base.copy(), 0.2)
    wrong = adjust_exposure(wrong, 20.0)
    wrong = linear_to_display(wrong)

    assert not np.allclose(correct, wrong, atol=1e-3)


def test_geometry_crop_layer_applies_before_color_stages():
    base = np.zeros((4, 4, 3), dtype=np.float32)
    base[:, :, 0] = 1.0  # solid red, so a crop is visible as a shape change
    doc = ImageDocument(base)
    doc.add_layer(GeometryLayer(crop_rect=(0.25, 0.25, 0.75, 0.75)))
    doc.add_layer(ExposureLayer(20.0))

    out = doc.render()
    assert out.shape[0] < base.shape[0]
    assert out.shape[1] < base.shape[1]


def test_geometry_override_replaces_committed_crop_layer_for_one_render():
    base = np.zeros((4, 4, 3), dtype=np.float32)
    doc = ImageDocument(base)
    doc.add_layer(GeometryLayer(crop_rect=(0.0, 0.0, 1.0, 1.0)))  # full frame

    override = GeometryLayer(crop_rect=(0.25, 0.25, 0.75, 0.75))
    out = doc.render(geometry_override=override)
    assert out.shape[0] == 2 and out.shape[1] == 2

    # Without the override, the committed (full-frame) layer is used again.
    out_committed = doc.render()
    assert out_committed.shape[0] == 4 and out_committed.shape[1] == 4


def test_pipeline_can_be_extended_with_a_new_layer_name_without_touching_image_document():
    """The actual point of the refactor: a future tool (e.g. HSL, color
    wheels) registers its layer's string name into an existing stage on
    document.pipeline, and render() picks it up automatically."""
    base = np.zeros((2, 2, 3), dtype=np.float32)
    doc = ImageDocument(base)
    doc.pipeline.register_layer("Display", "DoubleIt")

    class _DoubleLayer:
        def __str__(self):
            return "DoubleIt"

        def apply(self, image):
            return image * 2.0

    doc.add_layer(_DoubleLayer())
    out = doc.render()
    assert np.allclose(out, 0.0)  # black stays black either way, but...

    base_gray = np.full((2, 2, 3), 0.25, dtype=np.float32)
    doc2 = ImageDocument(base_gray)
    doc2.pipeline.register_layer("Display", "DoubleIt")
    doc2.add_layer(_DoubleLayer())
    out2 = doc2.render()
    expected = np.clip(linear_to_display(base_gray) * 2.0, 0.0, 1.0)
    assert np.allclose(out2, expected, atol=1e-6)
