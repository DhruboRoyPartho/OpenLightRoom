"""Tests for interface/gui/import_export_dialog.py's ExportDialog - the
per-format dynamic enable/disable logic that's the actual polish being
tested here (see core/io/image_io.py:save_image for what each format
really supports)."""

import pytest

from interface.gui.import_export_dialog import ExportDialog


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_default_format_is_jpg_with_quality_enabled_and_bit_depth_disabled(app):
    dialog = ExportDialog()
    assert dialog.format_box.currentText() == "JPG"
    assert dialog.quality_slider.isEnabled() is True
    assert dialog.bit_depth_box.isEnabled() is False
    assert dialog.bit_depth_box.currentText() == "8-bit"


def test_png_enables_both_quality_and_bit_depth(app):
    dialog = ExportDialog()
    dialog.format_box.setCurrentText("PNG")
    assert dialog.quality_slider.isEnabled() is True
    assert dialog.bit_depth_box.isEnabled() is True


def test_tiff_disables_quality_but_enables_bit_depth(app):
    dialog = ExportDialog()
    dialog.format_box.setCurrentText("TIFF")
    assert dialog.quality_slider.isEnabled() is False
    assert dialog.bit_depth_box.isEnabled() is True
    assert dialog.quality_hint.text() != ""


def test_webp_enables_quality_but_disables_bit_depth(app):
    dialog = ExportDialog()
    dialog.format_box.setCurrentText("WEBP")
    assert dialog.quality_slider.isEnabled() is True
    assert dialog.bit_depth_box.isEnabled() is False


def test_switching_to_an_8bit_only_format_resets_bit_depth_selection(app):
    dialog = ExportDialog()
    dialog.format_box.setCurrentText("PNG")
    dialog.bit_depth_box.setCurrentText("16-bit")

    dialog.format_box.setCurrentText("JPG")

    assert dialog.bit_depth_box.currentText() == "8-bit"
    assert dialog.bit_depth_box.isEnabled() is False


def test_quality_value_label_tracks_the_slider(app):
    dialog = ExportDialog()
    dialog.quality_slider.setValue(42)
    assert dialog.quality_value_label.text() == "42"


def test_cancel_button_rejects_the_dialog(app):
    from PySide6.QtWidgets import QDialog
    dialog = ExportDialog()
    dialog.cancel_btn.click()
    assert dialog.result() == QDialog.Rejected
