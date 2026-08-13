"""Tests for the Lightroom-style mask redesign:

- The red "Overlay" tint (ImageViewer._composite_mask_overlay /
  set_mask_overlay_provider) that shows where a mask actually applies,
  recomposited from a cached render rather than requiring a full
  pipeline re-render.
- Mutual exclusivity between crop mode, mask-edit mode and eyedropper
  mode (ImageViewer.cropModeChanged and the cross-mode force-exits),
  and CanvasToolbar staying in sync when crop is force-exited by
  something else (e.g. selecting a mask).
- MasksPanel automatically arming canvas interactivity/overlay on
  selection, with no separate "Edit on Canvas" toggle.
"""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from core.image_model.image_document import ImageDocument
from interface.gui.canvas_toolbar import CanvasToolbar
from interface.gui.image_viewer import ImageViewer, MASK_OVERLAY_COLOR, MASK_OVERLAY_STRENGTH
from interface.gui.layer_stack_panel import LayerStackPanel
from interface.gui.masks_panel import MasksPanel


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _force_render(viewer, app):
    viewer.render_queue._start_render()
    if viewer.render_queue.worker is not None:
        viewer.render_queue.worker.wait(5000)
    app.processEvents()


def _build(app, size=(20, 20, 3)):
    doc = ImageDocument(np.full(size, 0.3, dtype=np.float32))
    viewer = ImageViewer(doc)
    stack = LayerStackPanel(doc, viewer)
    panel = MasksPanel(doc, viewer, stack)
    toolbar = CanvasToolbar(doc, viewer, stack)
    panel.show()
    viewer.show()
    toolbar.show()
    _force_render(viewer, app)
    return doc, viewer, panel, toolbar


# --- overlay compositing math --------------------------------------------

def test_composite_mask_overlay_leaves_image_unchanged_where_alpha_is_zero(app):
    doc, viewer, panel, toolbar = _build(app)
    image = np.full((10, 10, 3), 0.5, dtype=np.float32)
    alpha = np.zeros((10, 10), dtype=np.float32)
    out = viewer._composite_mask_overlay(image, alpha)
    assert np.allclose(out, image)


def test_composite_mask_overlay_tints_fully_where_alpha_is_one(app):
    doc, viewer, panel, toolbar = _build(app)
    image = np.full((10, 10, 3), 0.5, dtype=np.float32)
    alpha = np.ones((10, 10), dtype=np.float32)
    out = viewer._composite_mask_overlay(image, alpha)
    expected = image * (1.0 - MASK_OVERLAY_STRENGTH) + MASK_OVERLAY_COLOR * MASK_OVERLAY_STRENGTH
    assert np.allclose(out, expected, atol=1e-6)


def test_composite_mask_overlay_resizes_a_mismatched_alpha(app):
    doc, viewer, panel, toolbar = _build(app)
    image = np.full((10, 10, 3), 0.5, dtype=np.float32)
    alpha = np.ones((5, 5), dtype=np.float32)  # deliberately smaller than image
    out = viewer._composite_mask_overlay(image, alpha)
    assert out.shape == image.shape
    assert not np.allclose(out, image)  # actually tinted, not silently skipped


# --- set_mask_overlay_provider / caching ---------------------------------

def test_set_mask_overlay_provider_recomposites_without_a_new_render(app):
    doc, viewer, panel, toolbar = _build(app)
    render_calls = []
    original_start_render = viewer.render_queue._start_render

    def spy(*a, **k):
        render_calls.append(1)
        return original_start_render(*a, **k)

    viewer.render_queue._start_render = spy
    viewer.set_mask_overlay_provider(lambda image: np.ones(image.shape[:2], dtype=np.float32))
    assert not render_calls
    assert viewer.is_mask_overlay_active() is True


def test_clearing_the_overlay_provider_restores_the_unmodified_render(app):
    doc, viewer, panel, toolbar = _build(app)
    plain_pixmap = viewer._base_pixmap.toImage()
    viewer.set_mask_overlay_provider(lambda image: np.ones(image.shape[:2], dtype=np.float32))
    tinted_pixmap = viewer._base_pixmap.toImage()
    assert tinted_pixmap != plain_pixmap

    viewer.set_mask_overlay_provider(None)
    assert viewer.is_mask_overlay_active() is False
    restored_pixmap = viewer._base_pixmap.toImage()
    assert restored_pixmap == plain_pixmap


def test_a_provider_returning_none_leaves_the_image_untinted(app):
    doc, viewer, panel, toolbar = _build(app)
    plain_pixmap = viewer._base_pixmap.toImage()
    viewer.set_mask_overlay_provider(lambda image: None)
    assert viewer._base_pixmap.toImage() == plain_pixmap


# --- MasksPanel: selection arms interactivity; overlay stays opt-in -----
#
# The overlay defaults OFF so the canvas always shows the actual graded
# image (every local adjustment visibly applied), not a translucent red
# wash sitting on top of it that makes it look like "nothing is really
# happening to the photo". It's still available as an explicit toggle,
# and flashes on automatically while a shape is actively being dragged/
# painted (see the drag-triggered overlay tests further down).

def test_selecting_a_mask_does_not_force_the_overlay_on(app):
    doc, viewer, panel, toolbar = _build(app)
    assert viewer.is_mask_overlay_active() is False

    panel._add_mask("radial")
    assert viewer.is_mask_overlay_active() is False
    assert panel.show_overlay_checkbox.isChecked() is False


def test_checking_show_overlay_arms_it_for_the_selected_mask(app):
    doc, viewer, panel, toolbar = _build(app)
    panel._add_mask("radial")
    assert viewer.is_mask_overlay_active() is False

    panel.show_overlay_checkbox.setChecked(True)
    assert viewer.is_mask_overlay_active() is True


def test_unchecking_show_overlay_clears_it_even_with_a_mask_selected(app):
    doc, viewer, panel, toolbar = _build(app)
    panel._add_mask("radial")
    panel.show_overlay_checkbox.setChecked(True)
    assert viewer.is_mask_overlay_active() is True

    panel.show_overlay_checkbox.setChecked(False)
    assert viewer.is_mask_overlay_active() is False


def test_deselecting_all_masks_clears_a_pinned_overlay(app):
    doc, viewer, panel, toolbar = _build(app)
    panel._add_mask("radial")
    panel.show_overlay_checkbox.setChecked(True)
    assert viewer.is_mask_overlay_active() is True

    panel._delete_selected_mask()
    assert viewer.is_mask_overlay_active() is False


def test_dragging_a_shape_temporarily_shows_the_overlay_then_hides_it_again(app):
    from PySide6.QtCore import QPointF
    doc, viewer, panel, toolbar = _build(app)
    panel._add_mask("radial")
    assert viewer.is_mask_overlay_active() is False  # not pinned on

    w, h = viewer.width(), viewer.height()
    viewer._handle_mask_mouse_press(QPointF(w / 2, h / 2))
    assert viewer.is_mask_overlay_active() is True  # visible while actively dragging

    viewer._handle_mask_mouse_move(QPointF(w / 2 - w * 0.1, h / 2 - h * 0.1))
    assert viewer.is_mask_overlay_active() is True

    viewer._handle_mask_mouse_release()
    assert viewer.is_mask_overlay_active() is False  # back off once the drag ends


def test_the_real_adjustment_is_visible_on_the_canvas_with_overlay_off(app):
    """The actual bug report this addresses: a local adjustment (e.g.
    Exposure) must show up as a real, clearly visible pixel change in
    the on-screen preview - not be masked by a permanent red tint that
    makes it look like only a "marker" is reacting, not the photo."""
    doc, viewer, panel, toolbar = _build(app, size=(20, 20, 3))
    panel._add_mask("radial")
    assert viewer.is_mask_overlay_active() is False

    slider, _sb, _rb = panel._adjustment_rows["exposure"]
    slider.sliderPressed.emit()
    slider.setValue(80)
    slider.sliderReleased.emit()
    _force_render(viewer, app)

    qimg = viewer._base_pixmap.toImage()
    w, h = qimg.width(), qimg.height()
    center = qimg.pixelColor(w // 2, h // 2)
    corner = qimg.pixelColor(2, 2)
    # A strong exposure boost inside the mask must read as a large,
    # unambiguous brightness difference from the untouched corner - not
    # a faint shift hidden under a permanent overlay tint.
    assert center.red() > corner.red() + 50
    assert center.green() > corner.green() + 50
    assert center.blue() > corner.blue() + 50


# --- crop / mask-edit / eyedropper mutual exclusivity --------------------

def test_entering_crop_mode_exits_mask_edit_mode_and_clears_overlay(app):
    doc, viewer, panel, toolbar = _build(app)
    panel._add_mask("radial")
    panel.show_overlay_checkbox.setChecked(True)
    assert viewer.is_mask_edit_mode() is True
    assert viewer.is_mask_overlay_active() is True

    viewer.enter_crop_mode()
    assert viewer.is_mask_edit_mode() is False
    assert viewer.is_mask_overlay_active() is False
    assert viewer.is_crop_mode() is True


def test_selecting_a_mask_while_cropping_exits_crop_mode(app):
    doc, viewer, panel, toolbar = _build(app)
    viewer.enter_crop_mode()
    assert viewer.is_crop_mode() is True

    panel._add_mask("radial")
    assert viewer.is_crop_mode() is False
    assert viewer.is_mask_edit_mode() is True


def test_entering_crop_mode_exits_eyedropper_mode(app):
    doc, viewer, panel, toolbar = _build(app)
    viewer.set_eyedropper_mode(True, purpose="white_balance")
    assert viewer.is_eyedropper_mode() is True

    viewer.enter_crop_mode()
    assert viewer.is_eyedropper_mode() is False


def test_eyedropper_mode_exits_crop_mode(app):
    doc, viewer, panel, toolbar = _build(app)
    viewer.enter_crop_mode()

    viewer.set_eyedropper_mode(True, purpose="white_balance")
    assert viewer.is_crop_mode() is False
    assert viewer.is_eyedropper_mode() is True


def test_crop_mode_changed_signal_fires_on_enter_and_exit(app):
    doc, viewer, panel, toolbar = _build(app)
    seen = []
    viewer.cropModeChanged.connect(seen.append)

    viewer.enter_crop_mode()
    assert seen == [True]

    viewer.exit_crop_mode()
    assert seen == [True, False]


def test_canvas_toolbar_crop_button_syncs_when_crop_is_force_exited_externally(app):
    doc, viewer, panel, toolbar = _build(app)
    toolbar.crop_btn.setChecked(True)
    assert viewer.is_crop_mode() is True
    assert toolbar.crop_row.isVisible() is True

    # Selecting a mask force-exits crop mode from underneath the toolbar -
    # its button/row must follow, not stay stuck showing "Crop" as active.
    panel._add_mask("radial")
    assert viewer.is_crop_mode() is False
    assert toolbar.crop_btn.isChecked() is False
    assert toolbar.crop_row.isVisible() is False


def test_canvas_toolbar_crop_button_click_still_drives_crop_mode_normally(app):
    doc, viewer, panel, toolbar = _build(app)
    toolbar.crop_btn.setChecked(True)
    assert viewer.is_crop_mode() is True

    toolbar.crop_btn.setChecked(False)
    assert viewer.is_crop_mode() is False
