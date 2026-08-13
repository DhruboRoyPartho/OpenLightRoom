"""GUI-level test for interface/gui/presets_panel.py - built and driven
directly (no Qt event loop polling; see the project's established pattern
for GUI tests under this sandbox), pointed at an isolated presets folder."""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from core.io import preset_io
from core.image_model.image_document import ImageDocument
from core.adjustment_layers.exposure_layer import ExposureLayer
from interface.gui.image_viewer import ImageViewer
from interface.gui.layer_stack_panel import LayerStackPanel
from interface.gui.presets_panel import PresetsPanel


@pytest.fixture(autouse=True)
def isolated_presets_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(preset_io, "PRESETS_DIR", str(tmp_path / "presets"))
    yield tmp_path


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _build_panel(app):
    doc = ImageDocument(np.zeros((4, 4, 3), dtype=np.float32))
    viewer = ImageViewer(doc)
    stack = LayerStackPanel(doc, viewer)
    sync_calls = []
    panel = PresetsPanel(doc, viewer, stack, on_layers_changed=lambda: sync_calls.append(1))
    return doc, panel, sync_calls


def test_refresh_lists_presets_from_disk(app):
    preset_io.save_preset("Alpha", [ExposureLayer(10.0)])
    preset_io.save_preset("Beta", [ExposureLayer(20.0)])
    doc, panel, _ = _build_panel(app)
    items = [panel.list_widget.item(i).text() for i in range(panel.list_widget.count())]
    assert items == ["Alpha", "Beta"]


def test_apply_with_no_selection_is_a_no_op(app):
    doc, panel, _ = _build_panel(app)
    panel._on_apply()
    assert doc.layers == []
    assert doc.history == []


def test_apply_adds_layer_as_single_undo_step_and_calls_sync_callback(app):
    preset_io.save_preset("Warm", [ExposureLayer(30.0)])
    doc, panel, sync_calls = _build_panel(app)
    panel.refresh()
    panel.list_widget.setCurrentRow(0)

    panel._on_apply()

    layer = next((l for l in doc.layers if str(l) == "Exposure"), None)
    assert layer is not None and layer.exposure_factor == 30.0
    assert len(doc.history) == 1
    assert sync_calls == [1]

    doc.undo()
    assert doc.layers == []


def test_apply_replaces_existing_layer_of_the_same_type(app):
    preset_io.save_preset("Warm", [ExposureLayer(30.0)])
    doc, panel, _ = _build_panel(app)
    doc.add_layer(ExposureLayer(-50.0))
    panel.refresh()
    panel.list_widget.setCurrentRow(0)

    panel._on_apply()

    layers_named_exposure = [l for l in doc.layers if str(l) == "Exposure"]
    assert len(layers_named_exposure) == 1
    assert layers_named_exposure[0].exposure_factor == 30.0


def test_apply_missing_preset_refreshes_list_without_raising(app, monkeypatch):
    # QMessageBox.warning() would otherwise open a real modal dialog and
    # block waiting for a click - not safe to trigger in an automated,
    # headless test, so it's replaced with a no-op for this one assertion.
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)

    doc, panel, _ = _build_panel(app)
    panel.list_widget.addItem("Ghost")
    panel.list_widget.setCurrentRow(0)
    panel._on_apply()  # "Ghost" isn't a real file - should warn, not raise
    assert panel.list_widget.count() == 0
