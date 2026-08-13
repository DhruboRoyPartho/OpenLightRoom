"""Tests for ImageDocument.next_mask_name()."""

import numpy as np

from core.image_model.image_document import ImageDocument


class _Named:
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return self._name

    def apply(self, image):
        return image


def _doc():
    return ImageDocument(np.zeros((2, 2, 3), dtype=np.float32))


def test_first_mask_is_mask_1():
    doc = _doc()
    assert doc.next_mask_name() == "Mask 1"


def test_increments_past_existing_mask_layers():
    doc = _doc()
    doc.layers = [_Named("Mask 1"), _Named("Mask 2")]
    assert doc.next_mask_name() == "Mask 3"


def test_numbers_are_not_reused_after_deletion():
    doc = _doc()
    doc.layers = [_Named("Mask 1"), _Named("Mask 3")]  # "Mask 2" was deleted
    assert doc.next_mask_name() == "Mask 4"


def test_ignores_unrelated_layer_names():
    doc = _doc()
    doc.layers = [_Named("Exposure"), _Named("Mask 1"), _Named("Masking Something Else")]
    assert doc.next_mask_name() == "Mask 2"


def test_empty_document_starts_at_one_even_with_other_layers_present():
    doc = _doc()
    doc.layers = [_Named("Exposure"), _Named("Contrast")]
    assert doc.next_mask_name() == "Mask 1"
