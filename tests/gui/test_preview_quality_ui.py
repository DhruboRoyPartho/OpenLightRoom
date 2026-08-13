"""Tests for the Preview Quality control: ImageViewer.set_preview_quality,
the CanvasToolbar dropdown wired to it, and the guarantee that export is
never affected by whatever preview quality is currently selected."""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QFileDialog

from core.image_model.image_document import ImageDocument
from interface.gui.image_viewer import ImageViewer
from interface.gui.layer_stack_panel import LayerStackPanel
from interface.gui.canvas_toolbar import CanvasToolbar, PREVIEW_QUALITY_OPTIONS
from interface.gui.main_window import MainWindow


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_viewer_defaults_to_full_quality(app):
    doc = ImageDocument(np.zeros((10, 10, 3), dtype=np.float32))
    viewer = ImageViewer(doc)
    assert viewer.preview_quality() is None


def test_viewer_set_preview_quality_updates_render_queue(app):
    doc = ImageDocument(np.zeros((10, 10, 3), dtype=np.float32))
    viewer = ImageViewer(doc)
    viewer.set_preview_quality(800)
    assert viewer.preview_quality() == 800
    assert viewer.render_queue.preview_max_dimension == 800


def test_viewer_set_preview_quality_triggers_a_new_render(app):
    doc = ImageDocument(np.zeros((10, 10, 3), dtype=np.float32))
    viewer = ImageViewer(doc)
    viewer.set_preview_quality(800)
    # update_view() -> request_render() starts the debounce timer.
    assert viewer.render_queue.timer.isActive() is True


def test_before_view_respects_the_selected_preview_quality(app):
    doc = ImageDocument(np.random.default_rng(0).random((300, 500, 3)).astype(np.float32))
    viewer = ImageViewer(doc)
    viewer.set_preview_quality(100)
    viewer.set_show_before(True)
    assert viewer._base_pixmap.width() == 100


def test_toolbar_combo_has_the_documented_options(app):
    doc = ImageDocument(np.zeros((10, 10, 3), dtype=np.float32))
    viewer = ImageViewer(doc)
    stack = LayerStackPanel(doc, viewer)
    toolbar = CanvasToolbar(doc, viewer, stack)

    labels = [toolbar.preview_quality_combo.itemText(i) for i in range(toolbar.preview_quality_combo.count())]
    assert labels == [label for label, _ in PREVIEW_QUALITY_OPTIONS]
    assert toolbar.preview_quality_combo.currentText() == "Full Quality"


def test_selecting_a_toolbar_option_updates_the_viewer(app):
    doc = ImageDocument(np.zeros((10, 10, 3), dtype=np.float32))
    viewer = ImageViewer(doc)
    stack = LayerStackPanel(doc, viewer)
    toolbar = CanvasToolbar(doc, viewer, stack)

    balanced_index = [label for label, _ in PREVIEW_QUALITY_OPTIONS].index("Balanced (1280px)")
    toolbar.preview_quality_combo.setCurrentIndex(balanced_index)
    assert viewer.preview_quality() == 1280


def test_export_image_renders_full_resolution_end_to_end(app, monkeypatch, tmp_path):
    """The actual guarantee the user asked for: whatever the on-screen
    preview quality is set to must never leak into the exported file."""
    from PySide6.QtWidgets import QDialog
    from interface.gui.import_export_dialog import ExportDialog

    w = MainWindow()
    doc = ImageDocument(np.random.default_rng(0).random((40, 60, 3)).astype(np.float32))
    w._set_document(doc, image_path="fake.jpg", project_path=None)
    w.image_viewer.set_preview_quality(20)  # low preview quality selected
    # That call queued a debounced preview render on a 50ms QTimer, purely
    # for the on-screen canvas - irrelevant to this test, which only needs
    # preview_quality() to already read 20 (true immediately, above) by
    # the time export runs. Left pending, that timer can fire at some
    # arbitrary later point (whenever the shared QApplication next pumps
    # events, e.g. during an unrelated later test) and land a stray
    # render for this exact document, flakily making the final assertion
    # depend on total test-suite timing. Stopping it removes that race.
    w.image_viewer.render_queue.timer.stop()

    # Scoped to this test's own document: ImageDocument.render is patched
    # at the class level, so a background RenderWorker QThread left over
    # from an earlier test's own (unrelated) document can still land a
    # call here otherwise - filtering by identity keeps this deterministic
    # regardless of any such stray cross-test thread.
    captured = []
    original_render = ImageDocument.render

    def spy_render(self, *args, **kwargs):
        if self is doc:
            captured.append(kwargs.get("max_dimension"))
        return original_render(self, *args, **kwargs)

    monkeypatch.setattr(ImageDocument, "render", spy_render)

    save_path = str(tmp_path / "out.jpg")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (save_path, "")))

    # Drive ExportDialog.exec() to immediately click Export, as if the
    # user accepted the dialog with its default settings.
    def fake_exec(self):
        self.export_btn.click()
        return QDialog.Accepted

    monkeypatch.setattr(ExportDialog, "exec", fake_exec)

    w.export_image()

    assert captured, "export_image() never called document.render()"
    # The export's own render() call is the last one issued in this
    # synchronous flow - the only one that matters for the guarantee under
    # test - and it must be full resolution regardless of the 20px preview
    # quality selected above.
    assert captured[-1] is None
