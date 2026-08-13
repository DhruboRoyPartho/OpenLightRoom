"""Tests for MainWindow's close confirmation (closeEvent, covering both the
File > Exit menu path and the native window-X path) and the new
File > Close Project action.

ExitDialog/ConfirmDialog.exec() is monkeypatched everywhere here - actually
executing one would open a real modal event loop and hang a headless test
run waiting for a click that will never come.
"""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QDialog

from interface.gui.main_window import MainWindow
from interface.gui.exit_dialog import ExitDialog
from interface.gui.confirm_dialog import ConfirmDialog
from core.image_model.image_document import ImageDocument


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


class _FakeCloseEvent:
    def __init__(self):
        self.accepted = None

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


def test_close_event_accepts_when_dialog_confirmed(app, monkeypatch):
    monkeypatch.setattr(ExitDialog, "exec", lambda self: QDialog.Accepted)
    w = MainWindow()
    event = _FakeCloseEvent()
    w.closeEvent(event)
    assert event.accepted is True


def test_close_event_ignores_when_dialog_cancelled(app, monkeypatch):
    monkeypatch.setattr(ExitDialog, "exec", lambda self: QDialog.Rejected)
    w = MainWindow()
    event = _FakeCloseEvent()
    w.closeEvent(event)
    assert event.accepted is False


def test_exit_program_routes_through_close_not_a_second_dialog(app, monkeypatch):
    """exit_program() must not show its own confirmation - closeEvent() is
    the single place that happens, or File > Exit would show the dialog
    twice (once from exit_program, once from the close() it triggers)."""
    calls = []
    monkeypatch.setattr(MainWindow, "close", lambda self: calls.append("close"))
    w = MainWindow()
    w.exit_program()
    assert calls == ["close"]


def test_close_project_is_a_no_op_with_nothing_open(app, monkeypatch):
    exec_calls = []
    monkeypatch.setattr(ConfirmDialog, "exec", lambda self: exec_calls.append(1) or QDialog.Accepted)
    w = MainWindow()
    assert w.current_image_path is None
    w.close_project()
    assert exec_calls == []  # dialog never even shown - nothing to confirm


def test_close_project_resets_to_blank_when_confirmed(app, monkeypatch):
    monkeypatch.setattr(ConfirmDialog, "exec", lambda self: QDialog.Accepted)
    w = MainWindow()
    doc = ImageDocument(np.full((4, 4, 3), 0.5, dtype=np.float32))
    w._set_document(doc, image_path="fake/path.jpg", project_path=None)
    assert w.current_image_path == "fake/path.jpg"

    w.close_project()

    assert w.current_image_path is None
    assert w.current_project_path is None


def test_close_project_does_nothing_when_cancelled(app, monkeypatch):
    monkeypatch.setattr(ConfirmDialog, "exec", lambda self: QDialog.Rejected)
    w = MainWindow()
    doc = ImageDocument(np.full((4, 4, 3), 0.5, dtype=np.float32))
    w._set_document(doc, image_path="fake/path.jpg", project_path=None)

    w.close_project()

    assert w.current_image_path == "fake/path.jpg"


def test_file_menu_has_close_project_action(app):
    w = MainWindow()
    menu_bar = w.menuBar()
    file_menu = menu_bar.actions()[0].menu()
    action_texts = [a.text() for a in file_menu.actions()]
    assert "Close Project" in action_texts
