"""Tests for the interactive canvas mask-placement system:
ImageViewer's mask-edit-mode signals/handlers (interface/gui/image_viewer.py)
and MasksPanel's auto-interactivity wiring (interface/gui/masks_panel.py) -
selecting a mask/component with a geometric (interactive) kind arms canvas
dragging immediately, with no separate toggle.

Driven via direct method calls (_handle_mask_mouse_press/_move/_release)
rather than real Qt mouse events or event-loop waits, matching this
project's established GUI-testing pattern. A render is waited on (via
worker.wait()) once per test that needs the viewer's actual pixel size,
since the widget only takes on its real (pixmap-derived) size after the
first async render completes - before that it sits at its 300x300
minimum size placeholder.
"""

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from core.image_model.image_document import ImageDocument
from interface.gui.image_viewer import ImageViewer
from interface.gui.masks_panel import MasksPanel
from interface.gui.layer_stack_panel import LayerStackPanel


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _build(app, size=(40, 40, 3)):
    doc = ImageDocument(np.full(size, 0.3, dtype=np.float32))
    viewer = ImageViewer(doc)
    stack = LayerStackPanel(doc, viewer)
    panel = MasksPanel(doc, viewer, stack)
    panel.show()
    viewer.show()
    if viewer.render_queue.worker is not None:
        viewer.render_queue.worker.wait(5000)
    app.processEvents()
    return doc, viewer, panel


# --- ImageViewer mask-edit-mode basics --------------------------------

def test_enter_and_exit_mask_edit_mode(app):
    doc, viewer, panel = _build(app)
    assert viewer.is_mask_edit_mode() is False

    viewer.enter_mask_edit_mode("radial", {"center_x": 0.5, "center_y": 0.5, "radius_x": 0.25, "radius_y": 0.25})
    assert viewer.is_mask_edit_mode() is True
    assert viewer._mask_edit_kind == "radial"

    viewer.exit_mask_edit_mode()
    assert viewer.is_mask_edit_mode() is False
    assert viewer._mask_edit_kind is None


def test_set_mask_edit_params_updates_the_live_overlay(app):
    doc, viewer, panel = _build(app)
    viewer.enter_mask_edit_mode("radial", {"center_x": 0.5, "center_y": 0.5, "radius_x": 0.25, "radius_y": 0.25})
    viewer.set_mask_edit_params({"center_x": 0.2, "center_y": 0.8, "radius_x": 0.1, "radius_y": 0.1})
    assert viewer._mask_edit_params["center_x"] == 0.2


# --- ellipse/radial/rectangle drag --------------------------------------

def test_dragging_the_center_handle_moves_the_shape_and_commits_one_undo_step(app):
    doc, viewer, panel = _build(app)
    panel._add_mask("radial")
    history_before = len(doc.history)

    w, h = viewer.width(), viewer.height()
    viewer._handle_mask_mouse_press(QPointF(w / 2, h / 2))
    assert viewer._mask_drag_handle == "center"
    viewer._handle_mask_mouse_move(QPointF(w / 2 - w * 0.1, h / 2 - h * 0.1))
    viewer._handle_mask_mouse_release()

    layer = panel._current_mask_layer()
    params = layer.mask.components[0].params
    assert params["center_x"] == pytest.approx(0.4, abs=1e-6)
    assert params["center_y"] == pytest.approx(0.4, abs=1e-6)
    assert len(doc.history) == history_before + 1


def test_dragging_the_corner_handle_resizes_the_shape(app):
    doc, viewer, panel = _build(app)
    panel._add_mask("ellipse")

    w, h = viewer.width(), viewer.height()
    params = panel._current_mask_layer().mask.components[0].params
    corner = QPointF((params["center_x"] + params["radius_x"]) * w, (params["center_y"] + params["radius_y"]) * h)

    viewer._handle_mask_mouse_press(corner)
    assert viewer._mask_drag_handle == "corner"
    viewer._handle_mask_mouse_move(QPointF(corner.x() + w * 0.1, corner.y()))
    viewer._handle_mask_mouse_release()

    new_params = panel._current_mask_layer().mask.components[0].params
    assert new_params["radius_x"] > params["radius_x"]


def test_rectangle_corner_drag_resizes_half_width_and_half_height(app):
    doc, viewer, panel = _build(app)
    panel._add_mask("rectangle")

    w, h = viewer.width(), viewer.height()
    params = panel._current_mask_layer().mask.components[0].params
    corner = QPointF((params["center_x"] + params["half_width"]) * w, (params["center_y"] + params["half_height"]) * h)

    viewer._handle_mask_mouse_press(corner)
    viewer._handle_mask_mouse_move(QPointF(corner.x() + w * 0.1, corner.y() + h * 0.1))
    viewer._handle_mask_mouse_release()

    new_params = panel._current_mask_layer().mask.components[0].params
    assert new_params["half_width"] > params["half_width"]
    assert new_params["half_height"] > params["half_height"]


def test_shape_drag_is_a_no_op_for_non_interactive_kinds(app):
    doc, viewer, panel = _build(app)
    panel._add_mask("luminance_range")  # not a geometric/interactive kind
    assert viewer.is_mask_edit_mode() is False

    w, h = viewer.width(), viewer.height()
    viewer._handle_mask_mouse_press(QPointF(w / 2, h / 2))  # no-op: mask edit mode never entered
    assert viewer._mask_drag_handle is None


def test_center_drag_undo_redo_round_trips(app):
    doc, viewer, panel = _build(app)
    panel._add_mask("radial")
    original_params = dict(panel._current_mask_layer().mask.components[0].params)

    w, h = viewer.width(), viewer.height()
    viewer._handle_mask_mouse_press(QPointF(w / 2, h / 2))
    viewer._handle_mask_mouse_move(QPointF(w * 0.2, h * 0.2))
    viewer._handle_mask_mouse_release()

    doc.undo()
    panel.refresh()
    assert panel._current_mask_layer().mask.components[0].params == original_params

    doc.redo()
    panel.refresh()
    assert panel._current_mask_layer().mask.components[0].params["center_x"] == pytest.approx(0.2, abs=0.05)


# --- linear gradient -----------------------------------------------------

def test_linear_gradient_endpoint_drag(app):
    doc, viewer, panel = _build(app)
    panel._add_mask("linear_gradient")

    w, h = viewer.width(), viewer.height()
    params = panel._current_mask_layer().mask.components[0].params
    start = QPointF(params["x0"] * w, params["y0"] * h)

    viewer._handle_mask_mouse_press(start)
    assert viewer._mask_drag_handle == "start"
    viewer._handle_mask_mouse_move(QPointF(w * 0.1, h * 0.9))
    viewer._handle_mask_mouse_release()

    new_params = panel._current_mask_layer().mask.components[0].params
    assert new_params["x0"] == pytest.approx(0.1, abs=1e-6)
    assert new_params["y0"] == pytest.approx(0.9, abs=1e-6)
    assert new_params["x1"] == params["x1"]  # the other endpoint is untouched


def test_linear_gradient_end_handle_is_independent_of_start(app):
    doc, viewer, panel = _build(app)
    panel._add_mask("linear_gradient")

    w, h = viewer.width(), viewer.height()
    params = panel._current_mask_layer().mask.components[0].params
    end = QPointF(params["x1"] * w, params["y1"] * h)

    viewer._handle_mask_mouse_press(end)
    assert viewer._mask_drag_handle == "end"
    viewer._handle_mask_mouse_move(QPointF(w * 0.9, h * 0.1))
    viewer._handle_mask_mouse_release()

    new_params = panel._current_mask_layer().mask.components[0].params
    assert new_params["x1"] == pytest.approx(0.9, abs=1e-6)
    assert new_params["x0"] == params["x0"]


# --- brush -----------------------------------------------------------------

def test_brush_settings_container_shows_only_for_brush_components(app):
    doc, viewer, panel = _build(app)
    panel._add_mask("radial")
    assert panel.brush_settings_container.isVisible() is False

    panel._add_mask("brush")
    assert panel.brush_settings_container.isVisible() is True


def test_painting_a_stroke_adds_it_to_the_component_and_commits_once(app):
    doc, viewer, panel = _build(app)
    panel._add_mask("brush")
    history_before = len(doc.history)

    w, h = viewer.width(), viewer.height()
    viewer._handle_mask_mouse_press(QPointF(w * 0.3, h * 0.5))
    viewer._handle_mask_mouse_move(QPointF(w * 0.5, h * 0.5))
    viewer._handle_mask_mouse_move(QPointF(w * 0.7, h * 0.5))
    viewer._handle_mask_mouse_release()

    layer = panel._current_mask_layer()
    strokes = layer.mask.components[0].params["strokes"]
    assert len(strokes) == 1
    assert len(strokes[0]["points"]) == 3
    assert len(doc.history) == history_before + 1


def test_painting_two_separate_strokes_appends_both(app):
    doc, viewer, panel = _build(app)
    panel._add_mask("brush")
    w, h = viewer.width(), viewer.height()

    for _ in range(2):
        viewer._handle_mask_mouse_press(QPointF(w * 0.3, h * 0.5))
        viewer._handle_mask_mouse_move(QPointF(w * 0.5, h * 0.5))
        viewer._handle_mask_mouse_release()

    layer = panel._current_mask_layer()
    assert len(layer.mask.components[0].params["strokes"]) == 2


def test_brush_stroke_actually_paints_pixels_on_render(app):
    doc, viewer, panel = _build(app, size=(40, 40, 3))
    panel._add_mask("brush")

    slider, _sb, _rb = panel._adjustment_rows["exposure"]
    slider.sliderPressed.emit()
    slider.setValue(80)
    slider.sliderReleased.emit()

    w, h = viewer.width(), viewer.height()
    viewer._handle_mask_mouse_press(QPointF(w * 0.5, h * 0.5))
    viewer._handle_mask_mouse_move(QPointF(w * 0.5, h * 0.5))
    viewer._handle_mask_mouse_release()

    out = doc.render()
    assert np.isfinite(out).all()
    center_px = out[out.shape[0] // 2, out.shape[1] // 2]
    corner_px = out[0, 0]
    assert center_px.mean() > corner_px.mean()


def test_empty_stroke_release_does_not_crash_or_commit(app):
    doc, viewer, panel = _build(app)
    panel._add_mask("brush")
    history_before = len(doc.history)

    viewer._handle_mask_mouse_release()  # release with no prior press
    assert len(doc.history) == history_before


# --- brush cursor preview shouldn't linger on the canvas -----------------
#
# _mouse_pos drives the brush tool's hover-preview circle (and the
# polygon tool's rubber-band line) in _paint_mask_brush/_paint_mask_polygon.
# It used to only ever get set (in mouseMoveEvent), never cleared, so the
# circle stayed painted at its last position after the mouse left the
# canvas - e.g. moving over to a slider in the Masks panel - reading as a
# stuck mark rather than a live cursor.

def test_mouse_leaving_the_canvas_clears_the_brush_cursor_preview(app):
    doc, viewer, panel = _build(app)
    panel._add_mask("brush")

    viewer._mouse_pos = QPointF(20, 20)
    assert viewer._mouse_pos is not None

    viewer.leaveEvent(None)
    assert viewer._mouse_pos is None


def test_leaving_the_canvas_mid_stroke_does_not_lose_the_recorded_points(app):
    """The preview circle should disappear, but an in-progress stroke's
    already-recorded path must survive - only the hover cursor is
    transient state."""
    doc, viewer, panel = _build(app)
    panel._add_mask("brush")

    w, h = viewer.width(), viewer.height()
    viewer._handle_mask_mouse_press(QPointF(w * 0.3, h * 0.5))
    viewer._handle_mask_mouse_move(QPointF(w * 0.5, h * 0.5))
    assert len(viewer._brush_stroke_points) == 2

    viewer.leaveEvent(None)
    assert viewer._mouse_pos is None
    assert len(viewer._brush_stroke_points) == 2  # untouched


def test_exiting_mask_edit_mode_also_clears_the_stale_cursor_position(app):
    doc, viewer, panel = _build(app)
    panel._add_mask("brush")
    viewer._mouse_pos = QPointF(20, 20)

    viewer.exit_mask_edit_mode()
    assert viewer._mouse_pos is None


# --- polygon -----------------------------------------------------------

def test_polygon_click_path_and_close(app):
    doc, viewer, panel = _build(app)
    panel._add_mask("polygon")
    history_before = len(doc.history)

    w, h = viewer.width(), viewer.height()
    points = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    for x, y in points:
        viewer._handle_mask_mouse_press(QPointF(x * w, y * h))
    assert len(viewer._polygon_points) == 4  # not yet closed

    # click near the first point again to close
    viewer._handle_mask_mouse_press(QPointF(points[0][0] * w + 2, points[0][1] * h + 2))

    layer = panel._current_mask_layer()
    stored_points = layer.mask.components[0].params["points"]
    assert stored_points == points
    assert viewer._polygon_points == []  # cleared after closing
    assert len(doc.history) == history_before + 1


def test_polygon_with_fewer_than_three_points_does_not_close(app):
    doc, viewer, panel = _build(app)
    panel._add_mask("polygon")

    w, h = viewer.width(), viewer.height()
    viewer._handle_mask_mouse_press(QPointF(w * 0.2, h * 0.2))
    viewer._handle_mask_mouse_press(QPointF(w * 0.8, h * 0.2))
    # clicking back near the first point with only 2 points placed should
    # just add a third point, not close (closing requires >= 3 already placed)
    viewer._handle_mask_mouse_press(QPointF(w * 0.2 + 1, h * 0.2 + 1))
    assert len(viewer._polygon_points) == 3


# --- auto-interactivity & component-kind gating -----------------------

def test_selecting_a_non_geometric_kind_does_not_arm_canvas_editing(app):
    doc, viewer, panel = _build(app)
    panel._add_mask("luminance_range")
    assert viewer.is_mask_edit_mode() is False


def test_selecting_a_geometric_kind_arms_canvas_editing(app):
    doc, viewer, panel = _build(app)
    panel._add_mask("radial")
    assert viewer.is_mask_edit_mode() is True


def test_switching_selected_component_updates_the_live_overlay_kind(app):
    doc, viewer, panel = _build(app)
    panel._add_mask("radial")
    panel._add_component("rectangle")
    assert viewer._mask_edit_kind == "rectangle"  # the newly-added, now-selected component

    panel._selected_component_index = 0
    panel._sync_editor()
    assert viewer._mask_edit_kind == "radial"


def test_switching_to_a_non_geometric_component_exits_viewer_mask_edit_mode(app):
    doc, viewer, panel = _build(app)
    panel._add_mask("radial")
    assert viewer.is_mask_edit_mode() is True

    panel._add_component("luminance_range")  # newly-added component becomes selected
    assert viewer.is_mask_edit_mode() is False

    panel._selected_component_index = 0  # back to the geometric radial component
    panel._sync_editor()
    assert viewer.is_mask_edit_mode() is True


def test_deselecting_the_mask_exits_viewer_mask_edit_mode(app):
    doc, viewer, panel = _build(app)
    panel._add_mask("radial")
    assert viewer.is_mask_edit_mode() is True

    panel.mask_list.setCurrentRow(-1)
    assert viewer.is_mask_edit_mode() is False


def test_deleting_the_selected_mask_exits_canvas_edit_mode(app):
    doc, viewer, panel = _build(app)
    panel._add_mask("radial")
    assert viewer.is_mask_edit_mode() is True

    panel._delete_selected_mask()
    assert viewer.is_mask_edit_mode() is False
