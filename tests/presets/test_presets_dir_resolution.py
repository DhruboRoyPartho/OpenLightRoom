"""Tests for core/io/preset_io.py's _default_presets_dir() - the actual
bug this guards against: a PyInstaller-frozen build resolved PRESETS_DIR
to a folder under the app's own install directory (e.g. Program Files),
which a non-admin user can't write to, crashing on startup with
PermissionError as soon as the Presets panel tried to list/create it.
"""

import os
import sys

from core.io import preset_io


def test_running_from_source_uses_the_repo_presets_folder(monkeypatch):
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    result = preset_io._default_presets_dir()
    assert os.path.basename(result) == "presets"
    # Not under a user-profile app-data directory - the repo's own folder.
    appdata = os.environ.get("APPDATA", "")
    assert not (appdata and result.startswith(appdata))


def test_frozen_build_uses_the_per_user_appdata_directory(monkeypatch, tmp_path):
    fake_appdata = str(tmp_path / "AppData" / "Roaming")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("APPDATA", fake_appdata)

    result = preset_io._default_presets_dir()

    assert result == os.path.join(fake_appdata, "OpenLightRoom", "presets")


def test_frozen_build_never_resolves_under_the_install_directory(monkeypatch, tmp_path):
    """The actual regression: presets must never end up as a sibling of
    preset_io.py's own (frozen, install-relative, read-only) location."""
    fake_appdata = str(tmp_path / "AppData" / "Roaming")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("APPDATA", fake_appdata)

    result = preset_io._default_presets_dir()

    install_relative = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(preset_io.__file__))))
    assert not result.startswith(install_relative)


def test_frozen_build_falls_back_when_appdata_is_unset(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.delenv("APPDATA", raising=False)

    result = preset_io._default_presets_dir()

    assert "OpenLightRoom" in result
    assert os.path.basename(result) == "presets"
