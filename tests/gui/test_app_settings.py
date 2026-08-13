"""Tests for interface/gui/app_settings.py's AppSettings.

Every test here constructs AppSettings with an explicit QSettings pointed
at a temp .ini file (via the `app_settings` fixture) - never the real,
on-disk store - so test runs can never read or write the actual user's
saved preferences.
"""

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from interface.gui.app_settings import (
    AppSettings, MAX_RECENT_PROJECTS,
    DEFAULT_PREVIEW_QUALITY_LABEL, DEFAULT_CONFIRM_BEFORE_EXIT,
    DEFAULT_EXPORT_FORMAT, DEFAULT_EXPORT_QUALITY, DEFAULT_EXPORT_BIT_DEPTH,
)


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def app_settings(app, tmp_path):
    backing = QSettings(str(tmp_path / "test_settings.ini"), QSettings.IniFormat)
    return AppSettings(backing)


def test_defaults_before_anything_is_saved(app_settings):
    assert app_settings.default_preview_quality_label() == DEFAULT_PREVIEW_QUALITY_LABEL
    assert app_settings.confirm_before_exit() == DEFAULT_CONFIRM_BEFORE_EXIT
    assert app_settings.default_export_format() == DEFAULT_EXPORT_FORMAT
    assert app_settings.default_export_quality() == DEFAULT_EXPORT_QUALITY
    assert app_settings.default_export_bit_depth() == DEFAULT_EXPORT_BIT_DEPTH
    assert app_settings.recent_projects() == []


def test_preview_quality_round_trips(app_settings):
    app_settings.set_default_preview_quality_label("Balanced (1280px)")
    assert app_settings.default_preview_quality_label() == "Balanced (1280px)"


def test_confirm_before_exit_round_trips_as_a_real_bool(app_settings):
    app_settings.set_confirm_before_exit(False)
    result = app_settings.confirm_before_exit()
    assert result is False

    app_settings.set_confirm_before_exit(True)
    assert app_settings.confirm_before_exit() is True


def test_export_defaults_round_trip(app_settings):
    app_settings.set_default_export_format("PNG")
    app_settings.set_default_export_quality(72)
    app_settings.set_default_export_bit_depth("16-bit")

    assert app_settings.default_export_format() == "PNG"
    assert app_settings.default_export_quality() == 72
    assert isinstance(app_settings.default_export_quality(), int)
    assert app_settings.default_export_bit_depth() == "16-bit"


def test_settings_persist_across_separate_appsettings_instances_on_the_same_file(app, tmp_path):
    ini_path = str(tmp_path / "shared.ini")
    first = AppSettings(QSettings(ini_path, QSettings.IniFormat))
    first.set_confirm_before_exit(False)
    first.set_default_export_format("TIFF")
    first.sync()

    second = AppSettings(QSettings(ini_path, QSettings.IniFormat))
    assert second.confirm_before_exit() is False
    assert second.default_export_format() == "TIFF"


# --- recent projects --------------------------------------------------

def test_add_recent_project_puts_it_first(app_settings):
    app_settings.add_recent_project("a.olrproj")
    app_settings.add_recent_project("b.olrproj")
    assert app_settings.recent_projects() == ["b.olrproj", "a.olrproj"]


def test_re_adding_an_existing_recent_project_moves_it_to_front_without_duplicating(app_settings):
    app_settings.add_recent_project("a.olrproj")
    app_settings.add_recent_project("b.olrproj")
    app_settings.add_recent_project("a.olrproj")
    assert app_settings.recent_projects() == ["a.olrproj", "b.olrproj"]


def test_recent_projects_list_is_capped(app_settings):
    for i in range(MAX_RECENT_PROJECTS + 5):
        app_settings.add_recent_project(f"project_{i}.olrproj")
    projects = app_settings.recent_projects()
    assert len(projects) == MAX_RECENT_PROJECTS
    # Most recently added stays first.
    assert projects[0] == f"project_{MAX_RECENT_PROJECTS + 4}.olrproj"


def test_remove_recent_project(app_settings):
    app_settings.add_recent_project("a.olrproj")
    app_settings.add_recent_project("b.olrproj")
    app_settings.remove_recent_project("a.olrproj")
    assert app_settings.recent_projects() == ["b.olrproj"]


def test_clear_recent_projects(app_settings):
    app_settings.add_recent_project("a.olrproj")
    app_settings.add_recent_project("b.olrproj")
    app_settings.clear_recent_projects()
    assert app_settings.recent_projects() == []


def test_a_single_recent_project_survives_the_round_trip_as_a_list(app, tmp_path):
    """Some QSettings backends collapse a one-item QVariantList back to a
    bare string on read - recent_projects() must still return a list."""
    ini_path = str(tmp_path / "single.ini")
    first = AppSettings(QSettings(ini_path, QSettings.IniFormat))
    first.add_recent_project("only.olrproj")
    first.sync()

    second = AppSettings(QSettings(ini_path, QSettings.IniFormat))
    result = second.recent_projects()
    assert result == ["only.olrproj"]
    assert isinstance(result, list)
