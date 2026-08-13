"""Tests for core/threads/render_worker.py's max_dimension pass-through."""

import numpy as np
import pytest

from core.image_model.image_document import ImageDocument
from core.threads.render_worker import RenderWorker


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_stores_max_dimension():
    doc = ImageDocument(np.zeros((10, 10, 3), dtype=np.float32))
    worker = RenderWorker(doc, max_dimension=512)
    assert worker.max_dimension == 512


def test_defaults_to_none():
    doc = ImageDocument(np.zeros((10, 10, 3), dtype=np.float32))
    worker = RenderWorker(doc)
    assert worker.max_dimension is None


def test_run_emits_a_downscaled_image_when_max_dimension_is_set(app):
    doc = ImageDocument(np.random.default_rng(0).random((400, 200, 3)).astype(np.float32))
    worker = RenderWorker(doc, max_dimension=100)

    results = []
    worker.rendered.connect(lambda img: results.append(img))
    worker.start()
    worker.wait(5000)
    app.processEvents()

    assert len(results) == 1
    assert results[0].shape == (100, 50, 3)


def test_run_emits_full_resolution_when_max_dimension_is_none(app):
    doc = ImageDocument(np.random.default_rng(0).random((80, 60, 3)).astype(np.float32))
    worker = RenderWorker(doc)

    results = []
    worker.rendered.connect(lambda img: results.append(img))
    worker.start()
    worker.wait(5000)
    app.processEvents()

    assert results[0].shape == (80, 60, 3)
