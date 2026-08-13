"""Tests for ImageDocument.render(max_dimension=...) - the preview-quality
render path, and the guarantee that base_image itself is never touched by
it (export always reads full-resolution base_image regardless of whatever
preview quality was last used on screen)."""

import numpy as np

from core.image_model.image_document import ImageDocument
from core.adjustment_layers.exposure_layer import ExposureLayer


def test_max_dimension_none_renders_at_full_resolution():
    base = np.random.default_rng(0).random((300, 500, 3)).astype(np.float32)
    doc = ImageDocument(base)
    out = doc.render()
    assert out.shape == (300, 500, 3)


def test_max_dimension_downscales_the_render_output():
    base = np.random.default_rng(0).random((300, 500, 3)).astype(np.float32)
    doc = ImageDocument(base)
    out = doc.render(max_dimension=100)
    assert out.shape[1] == 100
    assert out.shape[0] < 300


def test_base_image_is_never_mutated_by_a_preview_render():
    base = np.random.default_rng(0).random((300, 500, 3)).astype(np.float32)
    original = base.copy()
    doc = ImageDocument(base)
    doc.render(max_dimension=64)
    doc.render()  # full-res render afterward too
    assert np.array_equal(doc.base_image, original)


def test_adjustment_layers_still_apply_correctly_at_preview_resolution():
    """The whole point of downscaling before the pipeline instead of after:
    a color/tone edit should look the same (just softer/smaller), not be
    skipped or distorted, at preview resolution."""
    base = np.full((200, 200, 3), 0.3, dtype=np.float32)
    doc = ImageDocument(base)
    doc.add_layer(ExposureLayer(40.0))  # +2 stops, brightens noticeably

    full = doc.render()
    preview = doc.render(max_dimension=50)

    assert full.mean() > 0.3  # exposure did brighten the full render
    assert preview.mean() > 0.3  # ...and the preview render too
    assert abs(float(full.mean()) - float(preview.mean())) < 0.02  # closely matches


def test_geometry_override_and_max_dimension_compose_correctly():
    from core.adjustment_layers.geometry_layer import GeometryLayer
    base = np.random.default_rng(0).random((400, 400, 3)).astype(np.float32)
    doc = ImageDocument(base)
    override = GeometryLayer(crop_rect=(0.25, 0.25, 0.75, 0.75))

    out = doc.render(geometry_override=override, max_dimension=100)
    # Cropped to 50% of a (downscaled to <=100px) frame.
    assert out.shape[0] <= 50 and out.shape[1] <= 50


def test_result_is_bounded_and_finite_at_preview_resolution():
    base = np.random.default_rng(1).random((500, 500, 3)).astype(np.float32) * 3.0 - 1.0
    doc = ImageDocument(base)
    doc.add_layer(ExposureLayer(90.0))
    out = doc.render(max_dimension=64)
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
