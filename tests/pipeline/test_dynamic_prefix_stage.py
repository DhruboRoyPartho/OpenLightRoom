"""Tests for Stage's dynamic_prefix mode and Pipeline/ImageDocument's
plumbing of ordered_layers to it - the mechanism that lets a document hold
any number of "Mask N" layers at once, unlike every other (single-
instance, fixed-name) layer type."""

import numpy as np
import pytest

from core.pipeline.stage import Stage
from core.pipeline.pipeline import Pipeline
from core.image_model.image_document import ImageDocument


class _TaggingLayer:
    """Adds a constant and records its own name into a shared log, so
    tests can assert both which layers ran and in what order."""

    def __init__(self, name, amount, log):
        self._name = name
        self.amount = amount
        self.log = log

    def __str__(self):
        return self._name

    def apply(self, image):
        self.log.append(self._name)
        return image + self.amount


def test_stage_requires_exactly_one_mode():
    with pytest.raises(ValueError):
        Stage("Bad")
    with pytest.raises(ValueError):
        Stage("Bad", layer_order=["A"], dynamic_prefix="Mask ")
    with pytest.raises(ValueError):
        Stage("Bad", transform=lambda x: x, dynamic_prefix="Mask ")
    with pytest.raises(ValueError):
        Stage("Bad", layer_order=["A"], transform=lambda x: x, dynamic_prefix="Mask ")


def test_dynamic_prefix_applies_all_matching_layers_in_order():
    log = []
    a = _TaggingLayer("Mask 1", 1.0, log)
    b = _TaggingLayer("Mask 2", 10.0, log)
    unrelated = _TaggingLayer("Exposure", 100.0, log)
    stage = Stage("Masks", dynamic_prefix="Mask ")

    by_name = {"Mask 1": a, "Mask 2": b, "Exposure": unrelated}
    ordered = [a, b, unrelated]
    image = np.zeros((1, 1, 3), dtype=np.float32)
    out = stage.apply(image, by_name, ordered)

    assert log == ["Mask 1", "Mask 2"]  # "Exposure" is not a "Mask " layer
    assert np.allclose(out, 11.0)


def test_dynamic_prefix_falls_back_to_by_name_order_without_ordered_layers():
    log = []
    a = _TaggingLayer("Mask 1", 1.0, log)
    stage = Stage("Masks", dynamic_prefix="Mask ")
    image = np.zeros((1, 1, 3), dtype=np.float32)
    out = stage.apply(image, {"Mask 1": a})  # no ordered_layers passed
    assert log == ["Mask 1"]
    assert np.allclose(out, 1.0)


def test_pipeline_render_threads_ordered_layers_through_to_a_dynamic_stage():
    log = []
    a = _TaggingLayer("Mask 1", 1.0, log)
    b = _TaggingLayer("Mask 2", 1.0, log)
    pipeline = Pipeline([Stage("Masks", dynamic_prefix="Mask ")])
    by_name = {"Mask 1": a, "Mask 2": b}
    image = np.zeros((1, 1, 3), dtype=np.float32)
    out = pipeline.render(image, by_name, ordered_layers=[b, a])  # deliberately reversed
    assert log == ["Mask 2", "Mask 1"]
    assert np.allclose(out, 2.0)


def test_image_document_applies_any_number_of_mask_layers():
    base = np.zeros((2, 2, 3), dtype=np.float32)
    doc = ImageDocument(base)
    log = []
    doc.add_layer(_TaggingLayer("Mask 1", 0.1, log))
    doc.add_layer(_TaggingLayer("Mask 2", 0.2, log))
    doc.add_layer(_TaggingLayer("Mask 3", 0.3, log))

    out = doc.render()

    assert log == ["Mask 1", "Mask 2", "Mask 3"]
    assert np.allclose(out, 0.6, atol=1e-6)


def test_image_document_mask_layers_still_dedupe_by_exact_name():
    base = np.zeros((2, 2, 3), dtype=np.float32)
    doc = ImageDocument(base)
    log = []
    doc.add_layer(_TaggingLayer("Mask 1", 0.1, log))
    doc.add_layer(_TaggingLayer("Mask 1", 0.9, log))  # replaces the first "Mask 1"

    out = doc.render()

    assert log == ["Mask 1"]
    assert np.allclose(out, 0.9, atol=1e-6)
