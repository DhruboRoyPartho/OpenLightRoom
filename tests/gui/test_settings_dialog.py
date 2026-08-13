"""Tests for interface/gui/settings_dialog.py (Edit > Preferences...).

Uses an AppSettings backed by a temp .ini file (never the real, on-disk
store) so test runs stay hermetic."""

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QDialog

from interface.gui.app_settings import AppSettings
from interface.gui.settings_dialog import SettingsDialog


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def app_settings(app, tmp_path):
    backing = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    return AppSettings(backing)


def test_dialog_loads_current_settings_on_open(app, app_settings):
    app_settings.set_default_preview_quality_label("Fast (800px)")
    app_settings.set_confirm_before_exit(False)
    app_settings.set_default_export_format("PNG")
    app_settings.set_default_export_bit_depth("16-bit")
    app_settings.set_default_export_quality(60)

    dialog = SettingsDialog(app_settings)

    assert dialog.preview_quality_box.currentText() == "Fast (800px)"
    assert dialog.confirm_exit_check.isChecked() is False
    assert dialog.format_box.currentText() == "PNG"
    assert dialog.bit_depth_box.currentText() == "16-bit"
    assert dialog.quality_slider.value() == 60


def test_save_persists_changes_to_app_settings(app, app_settings):
    dialog = SettingsDialog(app_settings)
    dialog.preview_quality_box.setCurrentText("Balanced (1280px)")
    dialog.confirm_exit_check.setChecked(False)
    dialog.format_box.setCurrentText("TIFF")
    dialog.quality_slider.setValue(88)

    dialog.save_btn.click()

    assert dialog.result() == QDialog.Accepted
    assert app_settings.default_preview_quality_label() == "Balanced (1280px)"
    assert app_settings.confirm_before_exit() is False
    assert app_settings.default_export_format() == "TIFF"


def test_cancel_discards_changes(app, app_settings):
    app_settings.set_confirm_before_exit(True)
    dialog = SettingsDialog(app_settings)
    dialog.confirm_exit_check.setChecked(False)

    dialog.cancel_btn.click()

    assert dialog.result() == QDialog.Rejected
    assert app_settings.confirm_before_exit() is True  # untouched


def test_tiff_format_disables_quality_slider_in_export_defaults(app, app_settings):
    dialog = SettingsDialog(app_settings)
    dialog.format_box.setCurrentText("TIFF")
    assert dialog.quality_slider.isEnabled() is False
    assert dialog.bit_depth_box.isEnabled() is True


def test_jpg_format_disables_bit_depth_and_resets_it_to_8_bit(app, app_settings):
    app_settings.set_default_export_format("PNG")
    app_settings.set_default_export_bit_depth("16-bit")
    dialog = SettingsDialog(app_settings)
    assert dialog.bit_depth_box.currentText() == "16-bit"

    dialog.format_box.setCurrentText("JPG")

    assert dialog.bit_depth_box.isEnabled() is False
    assert dialog.bit_depth_box.currentText() == "8-bit"
