"""Tests for the Color Range mask's eyedropper (MasksPanel.pick_color_button)
and its mutual exclusivity with ControlsPanel's White Balance eyedropper -
both share ImageViewer.pixelPicked/eyedropperOwnerChanged, so only one can
be "armed" at a time.

Renders are forced synchronously (queue._start_render() + worker.wait())
rather than waiting on the real 50ms debounce QTimer, which needs actual
elapsed wall-clock time the offscreen test harness won't naturally give it
between statements.
"""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from core.image_model.image_document import ImageDocument
from interface.gui.controls_panel import ControlsPanel
from interface.gui.image_viewer import ImageViewer
from interface.gui.layer_stack_panel import LayerStackPanel


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _force_render(viewer, app):
    viewer.render_queue._start_render()
    if viewer.render_queue.worker is not None:
        viewer.render_queue.worker.wait(5000)
    app.processEvents()


def _build(app):
    base = np.zeros((20, 20, 3), dtype=np.float32)
    base[:, :10] = (0.8, 0.1, 0.1)   # left half: red
    base[:, 10:] = (0.1, 0.1, 0.8)   # right half: blue
    doc = ImageDocument(base)
    viewer = ImageViewer(doc)
    stack = LayerStackPanel(doc, viewer)
    panel = ControlsPanel(doc, viewer, stack)
    panel.show()
    viewer.show()
    _force_render(viewer, app)
    return doc, viewer, panel


def test_pick_color_button_only_appears_for_color_range_components(app):
    doc, viewer, panel = _build(app)
    mp = panel.masks_panel

    mp._add_mask("radial")
    assert mp.pick_color_button is None

    mp._add_mask("color_range")
    assert mp.pick_color_button is not None


def test_toggling_pick_color_arms_the_shared_eyedropper(app):
    doc, viewer, panel = _build(app)
    mp = panel.masks_panel
    mp._add_mask("color_range")

    mp.pick_color_button.setChecked(True)
    assert viewer.is_eyedropper_mode() is True
    assert viewer.eyedropper_purpose() == "color_range"


def test_picking_samples_the_rendered_pixel_and_commits_once(app):
    doc, viewer, panel = _build(app)
    mp = panel.masks_panel
    mp._add_mask("color_range")
    _force_render(viewer, app)
    history_before = len(doc.history)

    mp.pick_color_button.setChecked(True)
    pw, ph = viewer.current_pixmap_size()
    viewer.pixelPicked.emit(int(pw * 0.25), int(ph * 0.5), "color_range")

    layer = mp._current_mask_layer()
    r, g, b = layer.mask.components[0].params["sample_rgb"]
    assert r > b  # sampled from the red (left) half
    assert len(doc.history) == history_before + 1


def test_picking_a_different_region_gives_a_different_color(app):
    doc, viewer, panel = _build(app)
    mp = panel.masks_panel
    mp._add_mask("color_range")
    _force_render(viewer, app)
    pw, ph = viewer.current_pixmap_size()

    mp.pick_color_button.setChecked(True)
    viewer.pixelPicked.emit(int(pw * 0.75), int(ph * 0.5), "color_range")
    layer = mp._current_mask_layer()
    r_blue, g_blue, b_blue = layer.mask.components[0].params["sample_rgb"]
    assert b_blue > r_blue  # sampled from the blue (right) half


def test_pick_button_unchecks_itself_after_one_pick(app):
    doc, viewer, panel = _build(app)
    mp = panel.masks_panel
    mp._add_mask("color_range")
    _force_render(viewer, app)

    mp.pick_color_button.setChecked(True)
    pw, ph = viewer.current_pixmap_size()
    viewer.pixelPicked.emit(int(pw * 0.5), int(ph * 0.5), "color_range")

    assert mp.pick_color_button.isChecked() is False
    assert viewer.is_eyedropper_mode() is False


def test_pixel_picked_with_an_unrelated_purpose_is_ignored(app):
    """A purpose string neither this handler nor White Balance's
    recognizes (not a real achievable state via the actual toggle buttons,
    but a safety check that an unmatched purpose is inert either way)."""
    doc, viewer, panel = _build(app)
    mp = panel.masks_panel
    mp._add_mask("color_range")
    _force_render(viewer, app)
    original = dict(mp._current_mask_layer().mask.components[0].params)
    history_before = len(doc.history)

    pw, ph = viewer.current_pixmap_size()
    viewer.pixelPicked.emit(int(pw * 0.5), int(ph * 0.5), "some_other_tool")

    layer = mp._current_mask_layer()
    assert layer.mask.components[0].params == original
    assert len(doc.history) == history_before


# --- mutual exclusivity with the White Balance eyedropper ---------------

def test_enabling_masks_eyedropper_disarms_white_balance_eyedropper(app):
    doc, viewer, panel = _build(app)
    mp = panel.masks_panel
    mp._add_mask("color_range")

    panel.eyedropper_button.setChecked(True)
    assert panel.eyedropper_button.isChecked() is True

    mp.pick_color_button.setChecked(True)
    assert panel.eyedropper_button.isChecked() is False
    assert viewer.eyedropper_purpose() == "color_range"


def test_enabling_white_balance_eyedropper_disarms_masks_eyedropper(app):
    doc, viewer, panel = _build(app)
    mp = panel.masks_panel
    mp._add_mask("color_range")

    mp.pick_color_button.setChecked(True)
    assert mp.pick_color_button.isChecked() is True

    panel.eyedropper_button.setChecked(True)
    assert mp.pick_color_button.isChecked() is False
    assert viewer.eyedropper_purpose() == "white_balance"


def test_white_balance_pixel_pick_is_ignored_by_the_color_range_handler(app):
    """A White Balance pick must never accidentally set a Color Range
    component's sample_rgb, even though both listen on the same signal."""
    doc, viewer, panel = _build(app)
    mp = panel.masks_panel
    mp._add_mask("color_range")
    _force_render(viewer, app)
    original = dict(mp._current_mask_layer().mask.components[0].params)

    panel.eyedropper_button.setChecked(True)
    pw, ph = viewer.current_pixmap_size()
    viewer.pixelPicked.emit(int(pw * 0.5), int(ph * 0.5), viewer.eyedropper_purpose())

    layer = mp._current_mask_layer()
    assert layer.mask.components[0].params == original


def test_color_range_pixel_pick_is_ignored_by_the_white_balance_handler(app):
    """The inverse: a Color Range pick must never accidentally overwrite
    Temperature/Tint."""
    doc, viewer, panel = _build(app)
    mp = panel.masks_panel
    mp._add_mask("color_range")
    _force_render(viewer, app)

    mp.pick_color_button.setChecked(True)
    pw, ph = viewer.current_pixmap_size()
    viewer.pixelPicked.emit(int(pw * 0.25), int(ph * 0.5), viewer.eyedropper_purpose())

    assert next((l for l in doc.layers if str(l) == "Temperature"), None) is None
    assert next((l for l in doc.layers if str(l) == "Tint"), None) is None
