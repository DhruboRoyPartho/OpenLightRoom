"""Tests for interface/gui/assets.py (bundled asset path resolution) and
interface/gui/splash.py (responsive startup-splash sizing)."""

import os
import pytest
from PySide6.QtWidgets import QApplication

from interface.gui.assets import ASSETS_DIR, LOGO_PATH, LOADING_BANNER_PATH
from interface.gui.splash import build_splash_pixmap, MIN_WIDTH, MAX_WIDTH, SCREEN_WIDTH_FRACTION


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_logo_path_resolves_to_a_real_file():
    assert os.path.isfile(LOGO_PATH)


def test_loading_banner_path_resolves_to_a_real_file():
    assert os.path.isfile(LOADING_BANNER_PATH)


def test_assets_dir_is_the_project_level_assets_folder():
    assert os.path.basename(ASSETS_DIR) == "assets"
    assert os.path.isdir(ASSETS_DIR)


def test_build_splash_pixmap_is_not_null(app):
    pixmap = build_splash_pixmap(app.primaryScreen())
    assert not pixmap.isNull()


def test_build_splash_pixmap_preserves_the_banner_aspect_ratio(app):
    from PySide6.QtGui import QPixmap
    from interface.gui.assets import LOADING_BANNER_PATH

    source = QPixmap(LOADING_BANNER_PATH)
    source_ratio = source.width() / source.height()

    pixmap = build_splash_pixmap(app.primaryScreen())
    result_ratio = pixmap.width() / pixmap.height()
    assert result_ratio == pytest.approx(source_ratio, rel=1e-2)


def test_build_splash_pixmap_scales_with_a_larger_screen(app, monkeypatch):
    from PySide6.QtCore import QRect

    class FakeScreen:
        def __init__(self, width, ratio=1.0):
            self._width = width
            self._ratio = ratio

        def availableGeometry(self):
            return QRect(0, 0, self._width, round(self._width * 0.6))

        def devicePixelRatio(self):
            return self._ratio

    small = build_splash_pixmap(FakeScreen(1000))
    large = build_splash_pixmap(FakeScreen(2400))
    # Both clamp inside [MIN_WIDTH, MAX_WIDTH] * device ratio, but the
    # larger screen must still produce a wider (or equal, once clamped
    # at MAX_WIDTH) splash than the smaller one - never smaller.
    assert large.width() >= small.width()


def test_build_splash_pixmap_respects_the_minimum_width_floor(app):
    from PySide6.QtCore import QRect

    class TinyScreen:
        def availableGeometry(self):
            return QRect(0, 0, 200, 150)  # far below MIN_WIDTH / fraction

        def devicePixelRatio(self):
            return 1.0

    pixmap = build_splash_pixmap(TinyScreen())
    assert pixmap.width() >= MIN_WIDTH


def test_build_splash_pixmap_respects_the_maximum_width_cap(app):
    from PySide6.QtCore import QRect

    class HugeScreen:
        def availableGeometry(self):
            return QRect(0, 0, 8000, 4000)

        def devicePixelRatio(self):
            return 1.0

    pixmap = build_splash_pixmap(HugeScreen())
    assert pixmap.width() <= MAX_WIDTH


def test_build_splash_pixmap_scales_pixel_data_for_high_dpi_screens(app):
    from PySide6.QtCore import QRect

    class HiDpiScreen:
        def availableGeometry(self):
            return QRect(0, 0, 1600, 1000)

        def devicePixelRatio(self):
            return 2.0

    pixmap = build_splash_pixmap(HiDpiScreen())
    # Logical size (width() at the reported devicePixelRatio) stays sane...
    logical_width = pixmap.width() / pixmap.devicePixelRatio()
    assert MIN_WIDTH <= logical_width <= MAX_WIDTH * 1.01
    # ...while the underlying pixel data is rendered at 2x for sharpness.
    assert pixmap.devicePixelRatio() == pytest.approx(2.0)


def test_build_splash_pixmap_returns_null_when_asset_is_missing(app, monkeypatch):
    import interface.gui.splash as splash_module
    monkeypatch.setattr(splash_module, "LOADING_BANNER_PATH", "does/not/exist.png")
    pixmap = build_splash_pixmap(app.primaryScreen())
    assert pixmap.isNull()
