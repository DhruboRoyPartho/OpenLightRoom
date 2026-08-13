"""Tests for interface/gui/about_dialog.py - the app's Credits/About page.
Must actually surface the author's name, contact email, the open-source
project statement, and a working link to the GitHub repository, since
that's the entire point of this dialog."""

import pytest
from PySide6.QtWidgets import QApplication

from interface.gui.about_dialog import AboutDialog
from interface.gui.app_info import APP_NAME, APP_VERSION, AUTHOR_NAME, AUTHOR_EMAIL, GITHUB_URL


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_window_title_names_the_app(app):
    dialog = AboutDialog()
    assert APP_NAME in dialog.windowTitle()


def test_shows_the_author_name(app):
    dialog = AboutDialog()
    assert AUTHOR_NAME in dialog.credit_label.text()


def test_shows_a_working_mailto_link_for_the_authors_email(app):
    dialog = AboutDialog()
    assert f"mailto:{AUTHOR_EMAIL}" in dialog.credit_label.text()
    assert AUTHOR_EMAIL in dialog.credit_label.text()


def test_shows_the_github_repository_link(app):
    dialog = AboutDialog()
    assert GITHUB_URL in dialog.github_label.text()
    assert f'href="{GITHUB_URL}"' in dialog.github_label.text()


def test_links_are_clickable_without_extra_wiring(app):
    """openExternalLinks hands off to QDesktopServices automatically -
    this is what actually makes the mailto:/https:// anchors clickable."""
    dialog = AboutDialog()
    assert dialog.github_label.openExternalLinks() is True
    assert dialog.credit_label.openExternalLinks() is True


def test_shows_the_version(app):
    dialog = AboutDialog()
    # Rendered inside the dialog somewhere - walk all QLabel children.
    from PySide6.QtWidgets import QLabel
    all_text = " ".join(l.text() for l in dialog.findChildren(QLabel))
    assert APP_VERSION in all_text


def test_close_button_accepts_the_dialog(app):
    from PySide6.QtWidgets import QDialog
    dialog = AboutDialog()
    dialog.close_btn.click()
    assert dialog.result() == QDialog.Accepted
