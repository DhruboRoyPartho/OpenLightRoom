"""Tests for main.py's module-level helpers (everything outside the
`if __name__ == "__main__":` guard, which is the only importable part)."""

import os
import pytest

import main as main_module


def test_returns_none_with_no_args(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py"])
    assert main_module._requested_project_path() is None


def test_returns_none_for_a_non_project_argument(monkeypatch, tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"not a real jpeg, just needs to exist")
    monkeypatch.setattr("sys.argv", ["main.py", str(image)])
    assert main_module._requested_project_path() is None


def test_returns_none_for_a_olrproj_path_that_does_not_exist(monkeypatch, tmp_path):
    missing = tmp_path / "missing.olrproj"
    monkeypatch.setattr("sys.argv", ["main.py", str(missing)])
    assert main_module._requested_project_path() is None


def test_returns_the_path_for_an_existing_olrproj_argument(monkeypatch, tmp_path):
    project = tmp_path / "shoot.olrproj"
    project.write_text("{}")
    monkeypatch.setattr("sys.argv", ["main.py", str(project)])
    assert main_module._requested_project_path() == str(project)


def test_is_case_insensitive_about_the_extension(monkeypatch, tmp_path):
    project = tmp_path / "shoot.OLRPROJ"
    project.write_text("{}")
    monkeypatch.setattr("sys.argv", ["main.py", str(project)])
    assert main_module._requested_project_path() == str(project)
