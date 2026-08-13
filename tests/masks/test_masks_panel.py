"""GUI-level tests for interface/gui/masks_panel.py - driven via direct
method calls and signal emission on the real Qt objects, matching this
project's established pattern for GUI tests (avoids polling the Qt event
loop, which has been unreliable under sandbox memory pressure)."""

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from core.image_model.image_document import ImageDocument
from interface.gui.image_viewer import ImageViewer
from interface.gui.layer_stack_panel import LayerStackPanel
from interface.gui.masks_panel import MasksPanel, KIND_LABELS, MASK_TYPE_OPTIONS


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _build(app, size=(20, 20, 3), value=0.3):
    doc = ImageDocument(np.full(size, value, dtype=np.float32))
    viewer = ImageViewer(doc)
    stack = LayerStackPanel(doc, viewer)
    panel = MasksPanel(doc, viewer, stack)
    panel.show()
    return doc, panel


def test_kind_labels_map_every_documented_kind_to_a_display_label():
    for label, kind in MASK_TYPE_OPTIONS:
        assert KIND_LABELS[kind] == label


def test_starts_empty(app):
    doc, panel = _build(app)
    assert panel.mask_list.count() == 0
    assert panel.editor_container.isVisible() is False
    assert panel.empty_hint.isVisible() is True


def test_add_mask_creates_a_layer_and_selects_it(app):
    doc, panel = _build(app)
    panel._add_mask("radial")

    assert [str(l) for l in doc.layers] == ["Mask 1"]
    assert len(doc.history) == 1
    assert panel._selected_mask_name == "Mask 1"
    assert panel.mask_list.count() == 1
    assert panel.editor_container.isVisible() is True

    layer = panel._current_mask_layer()
    assert [(c.kind, c.op) for c in layer.mask.components] == [("radial", "add")]


def test_add_mask_undo_removes_it(app):
    doc, panel = _build(app)
    panel._add_mask("rectangle")
    doc.undo()
    assert doc.layers == []


def test_adding_a_second_mask_gets_a_distinct_name(app):
    doc, panel = _build(app)
    panel._add_mask("radial")
    panel._add_mask("ellipse")
    assert {str(l) for l in doc.layers} == {"Mask 1", "Mask 2"}


def test_exposure_slider_drag_commits_a_single_undo_step(app):
    doc, panel = _build(app)
    panel._add_mask("radial")
    history_after_add = len(doc.history)

    slider, spinbox, reset_btn = panel._adjustment_rows["exposure"]
    slider.sliderPressed.emit()
    slider.setValue(30)
    slider.setValue(50)
    slider.sliderReleased.emit()

    layer = panel._current_mask_layer()
    assert layer.exposure == 50.0
    assert len(doc.history) == history_after_add + 1

    doc.undo()
    layer = panel._current_mask_layer()
    assert layer is None or layer.exposure == 0.0


def test_exposure_actually_affects_the_masked_region_on_render(app):
    doc, panel = _build(app, size=(40, 40, 3), value=0.3)
    panel._add_mask("radial")  # default: centered, feathered ellipse
    slider, _sb, _rb = panel._adjustment_rows["exposure"]
    slider.sliderPressed.emit()
    slider.setValue(80)
    slider.sliderReleased.emit()

    out = doc.render()
    center = out[20, 20]
    corner = out[0, 0]
    assert center.mean() > corner.mean()  # brightened in the masked center, not at the untouched corner


def test_reset_button_restores_default_and_commits(app):
    doc, panel = _build(app)
    panel._add_mask("radial")
    slider, spinbox, reset_btn = panel._adjustment_rows["exposure"]
    slider.sliderPressed.emit()
    slider.setValue(60)
    slider.sliderReleased.emit()
    assert reset_btn.isEnabled()

    reset_btn.click()
    layer = panel._current_mask_layer()
    assert layer is None or layer.exposure == 0.0


def test_feather_blur_density_sliders_update_the_mask(app):
    doc, panel = _build(app)
    panel._add_mask("radial")

    panel.feather_slider.sliderPressed.emit()
    panel.feather_slider.setValue(70)
    panel.feather_slider.sliderReleased.emit()

    panel.blur_slider.sliderPressed.emit()
    panel.blur_slider.setValue(20)
    panel.blur_slider.sliderReleased.emit()

    panel.density_slider.sliderPressed.emit()
    panel.density_slider.setValue(50)
    panel.density_slider.sliderReleased.emit()

    layer = panel._current_mask_layer()
    assert layer.mask.feather == 70.0
    assert layer.mask.blur == 20.0
    assert layer.mask.density == 50.0


def test_invert_checkbox_toggles_the_whole_mask(app):
    doc, panel = _build(app)
    panel._add_mask("radial")
    panel.invert_check.setChecked(True)
    layer = panel._current_mask_layer()
    assert layer.mask.invert is True

    doc.undo()
    panel.refresh()
    layer = panel._current_mask_layer()
    assert layer.mask.invert is False


def test_add_component_appends_and_selects_it(app):
    doc, panel = _build(app)
    panel._add_mask("radial")
    panel._add_component("rectangle")

    layer = panel._current_mask_layer()
    assert [c.kind for c in layer.mask.components] == ["radial", "rectangle"]
    assert panel._selected_component_index == 1
    assert panel.component_list.count() == 2


def test_component_op_change_updates_the_second_components_operator(app):
    doc, panel = _build(app)
    panel._add_mask("radial")
    panel._add_component("rectangle")
    panel._selected_component_index = 1
    panel._sync_editor()

    panel.component_op_combo.setCurrentIndex(1)  # "Subtract"
    layer = panel._current_mask_layer()
    assert layer.mask.components[1].op == "subtract"


def test_component_invert_checkbox_flips_only_that_component(app):
    doc, panel = _build(app)
    panel._add_mask("radial")
    panel._add_component("rectangle")
    panel._selected_component_index = 1
    panel._sync_editor()

    panel.component_invert_check.setChecked(True)
    layer = panel._current_mask_layer()
    assert layer.mask.components[1].invert is True
    assert layer.mask.components[0].invert is False


def test_delete_component_removes_only_that_one(app):
    doc, panel = _build(app)
    panel._add_mask("radial")
    panel._add_component("rectangle")
    panel._selected_component_index = 1
    panel._sync_editor()

    panel._delete_selected_component()
    layer = panel._current_mask_layer()
    assert [c.kind for c in layer.mask.components] == ["radial"]


def test_cannot_delete_the_last_component_down_to_zero_is_allowed_but_mask_becomes_empty(app):
    doc, panel = _build(app)
    panel._add_mask("radial")
    panel._selected_component_index = 0
    panel._sync_editor()
    panel._delete_selected_component()

    layer = panel._current_mask_layer()
    assert layer.mask.is_empty()


def test_shape_param_slider_updates_component_geometry(app):
    doc, panel = _build(app)
    panel._add_mask("radial")
    panel._rebuild_param_fields()

    # First field for "radial" is Center X (0..100 -> 0.0..1.0 fraction).
    row_layout = panel.param_fields_layout.itemAt(0).layout()
    # Locate the slider within that row (label, slider, spinbox, reset).
    slider = None
    for i in range(row_layout.count()):
        w = row_layout.itemAt(i).widget()
        if hasattr(w, "sliderPressed"):
            slider = w
            break
    assert slider is not None

    slider.sliderPressed.emit()
    slider.setValue(75)
    slider.sliderReleased.emit()

    layer = panel._current_mask_layer()
    assert layer.mask.components[0].params["center_x"] == pytest.approx(0.75)


def test_visibility_checkbox_toggle_updates_layer_and_commits(app):
    doc, panel = _build(app)
    panel._add_mask("radial")
    item = panel.mask_list.item(0)
    item.setCheckState(Qt.Unchecked)
    panel._on_mask_item_changed(item)

    layer = panel._current_mask_layer()
    assert layer.visible is False

    doc.undo()
    panel.refresh()
    layer = panel._current_mask_layer()
    assert layer.visible is True


def test_rename_updates_label_not_pipeline_identity(app, monkeypatch):
    from PySide6.QtWidgets import QInputDialog
    doc, panel = _build(app)
    panel._add_mask("radial")
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Sky", True)))

    panel._rename_selected_mask()
    layer = panel._current_mask_layer()
    assert layer.label == "Sky"
    assert str(layer) == "Mask 1"


def test_duplicate_creates_an_independent_copy(app):
    doc, panel = _build(app)
    panel._add_mask("radial")
    slider, _sb, _rb = panel._adjustment_rows["exposure"]
    slider.sliderPressed.emit()
    slider.setValue(30)
    slider.sliderReleased.emit()

    panel._duplicate_selected_mask()

    assert {str(l) for l in doc.layers} == {"Mask 1", "Mask 2"}
    original = next(l for l in doc.layers if str(l) == "Mask 1")
    copy = next(l for l in doc.layers if str(l) == "Mask 2")
    assert copy.exposure == original.exposure == 30.0
    assert copy.mask is not original.mask  # independent objects, not shared


def test_delete_mask_removes_it_and_deselects(app):
    doc, panel = _build(app)
    panel._add_mask("radial")
    panel._delete_selected_mask()
    assert doc.layers == []
    assert panel._selected_mask_name is None
    assert panel.editor_container.isVisible() is False


def test_move_mask_swaps_pipeline_order(app):
    doc, panel = _build(app)
    panel._add_mask("radial")   # Mask 1
    panel._add_mask("ellipse")  # Mask 2, currently selected

    names_before = [str(l) for l in doc.layers]
    assert names_before == ["Mask 1", "Mask 2"]

    panel._move_selected_mask(-1)  # move "Mask 2" up, ahead of "Mask 1"
    names_after = [str(l) for l in doc.layers]
    assert names_after == ["Mask 2", "Mask 1"]


def test_refresh_after_undo_reflects_document_state(app):
    doc, panel = _build(app)
    panel._add_mask("radial")
    panel._add_mask("ellipse")
    doc.undo()  # removes "Mask 2"
    panel.refresh()
    assert panel.mask_list.count() == 1
