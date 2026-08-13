"""Tests for RenderQueue/RenderWorker's preview_max_dimension plumbing.

Driven with direct calls (_start_render(), worker.wait()) rather than
waiting on the real 50ms debounce timer via the Qt event loop - faster and
avoids the timing flakiness documented elsewhere in this project's GUI
tests under sandbox memory pressure.
"""

import numpy as np
import pytest

from core.image_model.image_document import ImageDocument
from core.threads.render_queue import RenderQueue


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_defaults_to_full_resolution():
    doc = ImageDocument(np.zeros((10, 10, 3), dtype=np.float32))
    queue = RenderQueue(doc)
    assert queue.preview_max_dimension is None


def test_set_preview_max_dimension_updates_state():
    doc = ImageDocument(np.zeros((10, 10, 3), dtype=np.float32))
    queue = RenderQueue(doc)
    queue.set_preview_max_dimension(800)
    assert queue.preview_max_dimension == 800


def test_launching_a_worker_passes_the_current_preview_max_dimension():
    doc = ImageDocument(np.zeros((10, 10, 3), dtype=np.float32))
    queue = RenderQueue(doc)
    queue.set_preview_max_dimension(640)
    queue._launch_worker()
    assert queue.worker.max_dimension == 640
    queue.worker.wait(5000)


def test_full_resolution_round_trip_produces_full_size_output(app):
    doc = ImageDocument(np.random.default_rng(0).random((300, 500, 3)).astype(np.float32))
    queue = RenderQueue(doc)

    results = []
    queue.image_rendered.connect(lambda img: results.append(img))
    queue._start_render()
    queue.worker.wait(5000)
    app.processEvents()

    assert len(results) == 1
    assert results[0].shape == (300, 500, 3)


def test_preview_resolution_round_trip_produces_downscaled_output(app):
    doc = ImageDocument(np.random.default_rng(0).random((300, 500, 3)).astype(np.float32))
    queue = RenderQueue(doc)
    queue.set_preview_max_dimension(100)

    results = []
    queue.image_rendered.connect(lambda img: results.append(img))
    queue._start_render()
    queue.worker.wait(5000)
    app.processEvents()

    assert len(results) == 1
    assert results[0].shape[1] == 100
    assert results[0].shape[0] < 300


def test_render_started_fires_once_per_launch(app):
    doc = ImageDocument(np.zeros((10, 10, 3), dtype=np.float32))
    queue = RenderQueue(doc)

    starts = []
    queue.render_started.connect(lambda: starts.append(1))
    queue._start_render()
    queue.worker.wait(5000)
    app.processEvents()

    assert starts == [1]
