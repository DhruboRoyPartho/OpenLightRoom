"""Tests for MainWindow's enriched menu bar: Edit (Undo/Redo/Preferences),
View (zoom/fit/before-after), Help (About/GitHub/Report an Issue), and
File's new "Open Recent" submenu.

Every MainWindow here is built with an explicit AppSettings backed by a
temp .ini file (never the real, on-disk store), and every dialog's
exec() is monkeypatched - actually executing one would open a real modal
event loop (or, for the Help links, a real browser window) and hang/
disrupt a headless test run.
"""

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QDialog

from interface.gui.main_window import MainWindow
from interface.gui.app_settings import AppSettings
from interface.gui.about_dialog import AboutDialog
from interface.gui.settings_dialog import SettingsDialog
from interface.gui.app_info import GITHUB_URL, GITHUB_ISSUES_URL
from core.image_model.image_document import ImageDocument
from core.adjustment_layers.exposure_layer import ExposureLayer


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def app_settings(app, tmp_path):
    backing = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    return AppSettings(backing)


def _menu(window, title):
    for action in window.menuBar().actions():
        if action.text() == title:
            return action.menu()
    return None


def test_menu_bar_has_the_four_professional_menus(app, app_settings):
    w = MainWindow(settings=app_settings)
    titles = [a.text() for a in w.menuBar().actions()]
    assert titles == ["File", "Edit", "View", "Help"]


# --- Edit: Undo/Redo ----------------------------------------------------

def test_undo_redo_actions_start_disabled_on_a_fresh_document(app, app_settings):
    w = MainWindow(settings=app_settings)
    w._sync_edit_menu_state()
    assert w.undo_action.isEnabled() is False
    assert w.redo_action.isEnabled() is False


def test_undo_action_becomes_enabled_after_a_change_and_calls_document_undo(app, app_settings):
    w = MainWindow(settings=app_settings)
    doc = ImageDocument(np.full((4, 4, 3), 0.5, dtype=np.float32))
    w._set_document(doc, image_path="fake.jpg", project_path=None)

    old_layer = None
    new_layer = ExposureLayer(exposure_factor=20.0)
    from core.commands.change_layer_command import ChangeLayerCommand
    doc.execute_command(ChangeLayerCommand(doc, "Exposure", old_layer, new_layer))

    w._sync_edit_menu_state()
    assert w.undo_action.isEnabled() is True
    assert w.redo_action.isEnabled() is False

    w.undo_action.trigger()
    assert doc.history == []

    w._sync_edit_menu_state()
    assert w.redo_action.isEnabled() is True


def test_edit_menu_contains_preferences(app, app_settings):
    w = MainWindow(settings=app_settings)
    edit_menu = _menu(w, "Edit")
    assert "Preferences..." in [a.text() for a in edit_menu.actions()]


# --- View -----------------------------------------------------------------

def test_view_menu_has_the_expected_actions(app, app_settings):
    w = MainWindow(settings=app_settings)
    view_menu = _menu(w, "View")
    texts = [a.text() for a in view_menu.actions() if a.text()]
    assert texts == ["Zoom In", "Zoom Out", "Fit to Window", "Actual Size (100%)", "Toggle Before / After"]


def test_toggle_before_after_action_toggles_the_canvas_toolbar(app, app_settings):
    w = MainWindow(settings=app_settings)
    view_menu = _menu(w, "View")
    action = next(a for a in view_menu.actions() if a.text() == "Toggle Before / After")

    assert w.canvas_toolbar.before_after_btn.isChecked() is False
    action.trigger()
    assert w.canvas_toolbar.before_after_btn.isChecked() is True


def test_zoom_in_action_calls_the_viewers_zoom_in(app, app_settings, monkeypatch):
    w = MainWindow(settings=app_settings)
    calls = []
    monkeypatch.setattr(w.image_viewer, "zoom_in", lambda: calls.append("zoom_in"))
    view_menu = _menu(w, "View")
    next(a for a in view_menu.actions() if a.text() == "Zoom In").trigger()
    assert calls == ["zoom_in"]


# --- Help -----------------------------------------------------------------

def test_help_menu_has_about_and_link_actions(app, app_settings):
    w = MainWindow(settings=app_settings)
    help_menu = _menu(w, "Help")
    texts = [a.text() for a in help_menu.actions() if a.text()]
    assert "About Open LightRoom..." in texts
    assert "View on GitHub" in texts
    assert "Report an Issue..." in texts


def test_about_action_opens_the_about_dialog(app, app_settings, monkeypatch):
    opened = []
    monkeypatch.setattr(AboutDialog, "exec", lambda self: opened.append(1) or QDialog.Accepted)
    w = MainWindow(settings=app_settings)
    help_menu = _menu(w, "Help")
    next(a for a in help_menu.actions() if a.text() == "About Open LightRoom...").trigger()
    assert opened == [1]


def test_github_action_opens_the_correct_url(app, app_settings, monkeypatch):
    opened_urls = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened_urls.append(url.toString()))
    w = MainWindow(settings=app_settings)
    help_menu = _menu(w, "Help")
    next(a for a in help_menu.actions() if a.text() == "View on GitHub").trigger()
    assert opened_urls == [GITHUB_URL]


def test_report_issue_action_opens_the_issues_url(app, app_settings, monkeypatch):
    opened_urls = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened_urls.append(url.toString()))
    w = MainWindow(settings=app_settings)
    help_menu = _menu(w, "Help")
    next(a for a in help_menu.actions() if a.text() == "Report an Issue...").trigger()
    assert opened_urls == [GITHUB_ISSUES_URL]


# --- Preferences ------------------------------------------------------

def test_open_preferences_applies_new_preview_quality_on_accept(app, app_settings, monkeypatch):
    def fake_exec(self):
        self.preview_quality_box.setCurrentText("Fast (800px)")
        self._save()
        return QDialog.Accepted

    monkeypatch.setattr(SettingsDialog, "exec", fake_exec)
    w = MainWindow(settings=app_settings)

    w.open_preferences()

    assert app_settings.default_preview_quality_label() == "Fast (800px)"
    assert w.canvas_toolbar.preview_quality_combo.currentText() == "Fast (800px)"
    assert w.image_viewer.preview_quality() == 800


def test_open_preferences_does_nothing_on_cancel(app, app_settings, monkeypatch):
    monkeypatch.setattr(SettingsDialog, "exec", lambda self: QDialog.Rejected)
    w = MainWindow(settings=app_settings)
    original = w.canvas_toolbar.preview_quality_combo.currentText()

    w.open_preferences()

    assert w.canvas_toolbar.preview_quality_combo.currentText() == original


def test_new_documents_start_at_the_saved_default_preview_quality(app, app_settings):
    app_settings.set_default_preview_quality_label("High (2048px)")
    w = MainWindow(settings=app_settings)
    doc = ImageDocument(np.full((4, 4, 3), 0.5, dtype=np.float32))

    w._set_document(doc, image_path="fake.jpg", project_path=None)

    assert w.canvas_toolbar.preview_quality_combo.currentText() == "High (2048px)"
    assert w.image_viewer.preview_quality() == 2048


# --- confirm_before_exit ------------------------------------------------

def test_close_event_skips_the_dialog_when_confirm_before_exit_is_disabled(app, app_settings):
    app_settings.set_confirm_before_exit(False)
    w = MainWindow(settings=app_settings)

    class _FakeCloseEvent:
        def __init__(self):
            self.accepted = None

        def accept(self):
            self.accepted = True

        def ignore(self):
            self.accepted = False

    event = _FakeCloseEvent()
    w.closeEvent(event)
    assert event.accepted is True


# --- recent projects --------------------------------------------------

def test_recent_projects_menu_starts_empty(app, app_settings):
    w = MainWindow(settings=app_settings)
    texts = [a.text() for a in w.recent_projects_menu.actions()]
    assert texts == ["No Recent Projects"]
    assert w.recent_projects_menu.actions()[0].isEnabled() is False


def test_recent_projects_menu_populates_after_a_project_write(app, app_settings, tmp_path):
    w = MainWindow(settings=app_settings)
    doc = ImageDocument(np.full((4, 4, 3), 0.5, dtype=np.float32))
    w._set_document(doc, image_path="fake.jpg", project_path=None)

    project_path = str(tmp_path / "test.olrproj")
    from core.io import project_io
    project_io.save_project(project_path, "fake.jpg", doc)
    w.current_project_path = project_path
    w.settings.add_recent_project(project_path)
    w._rebuild_recent_projects_menu()

    texts = [a.text() for a in w.recent_projects_menu.actions()]
    assert texts[0] == "test.olrproj"
    assert "Clear Recent Projects" in texts


def test_clear_recent_projects_empties_the_menu(app, app_settings):
    w = MainWindow(settings=app_settings)
    app_settings.add_recent_project("a.olrproj")
    w._rebuild_recent_projects_menu()
    assert len(w.recent_projects_menu.actions()) > 1

    w._clear_recent_projects()

    texts = [a.text() for a in w.recent_projects_menu.actions()]
    assert texts == ["No Recent Projects"]


def test_opening_a_missing_recent_project_warns_and_removes_it(app, app_settings, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(1))
    w = MainWindow(settings=app_settings)
    app_settings.add_recent_project("does/not/exist.olrproj")

    w._open_recent_project("does/not/exist.olrproj")

    assert warned == [1]
    assert "does/not/exist.olrproj" not in app_settings.recent_projects()
