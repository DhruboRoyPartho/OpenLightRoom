"""Tests for interface/gui/confirm_dialog.py and exit_dialog.py."""

import pytest
from PySide6.QtWidgets import QDialog

from interface.gui.confirm_dialog import ConfirmDialog
from interface.gui.exit_dialog import ExitDialog


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_confirm_button_accepts(app):
    dialog = ConfirmDialog("Close Project", "Close the current project?", confirm_text="Close Project")
    dialog.confirm_btn.click()
    assert dialog.result() == QDialog.Accepted


def test_cancel_button_rejects(app):
    dialog = ConfirmDialog("Close Project", "Close the current project?")
    dialog.cancel_btn.click()
    assert dialog.result() == QDialog.Rejected


def test_confirm_text_is_applied_to_the_button(app):
    dialog = ConfirmDialog("Title", "Message", confirm_text="Delete Forever")
    assert dialog.confirm_btn.text() == "Delete Forever"


def test_yes_no_aliases_point_at_the_same_buttons(app):
    dialog = ConfirmDialog("Title", "Message")
    assert dialog.yes_btn is dialog.confirm_btn
    assert dialog.no_btn is dialog.cancel_btn


def test_exit_dialog_uses_the_literal_requested_phrasing(app):
    from PySide6.QtWidgets import QLabel
    dialog = ExitDialog()
    texts = [label.text() for label in dialog.findChildren(QLabel)]
    assert any("Do you want to close the app?" in t for t in texts)


def test_exit_dialog_confirm_accepts_and_cancel_rejects(app):
    dialog = ExitDialog()
    assert dialog.confirm_btn.text() == "Close"
    dialog.cancel_btn.click()
    assert dialog.result() == QDialog.Rejected

    dialog2 = ExitDialog()
    dialog2.confirm_btn.click()
    assert dialog2.result() == QDialog.Accepted
