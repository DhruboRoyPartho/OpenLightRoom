"""
Project: Open LightRoom
Author: Dhrubo Roy Partho
Date: 15-05-2025
Version: see interface/gui/app_info.py:APP_VERSION
"""

import os
import sys
import time

from PySide6.QtWidgets import QApplication, QSplashScreen
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt

from interface.gui.main_window import MainWindow
from interface.gui.theme import apply_theme
from interface.gui.assets import LOGO_PATH
from interface.gui.splash import build_splash_pixmap, MIN_VISIBLE_MS


def _fix_windows_taskbar_icon():
    """Without this, Windows groups the app under python.exe's own
    generic icon in the taskbar - it identifies pinned/running apps by
    the hosting executable unless told otherwise via an explicit App
    User Model ID. No-op (and harmless) on any other platform."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("OpenLightRoom.PhotoEditor.1")
    except Exception:
        pass


def _requested_project_path():
    """A .olrproj path passed on the command line - e.g. Explorer
    double-clicking a project file registered to open with this app via
    an installer's file association - or None."""
    for arg in sys.argv[1:]:
        if arg.lower().endswith(".olrproj") and os.path.isfile(arg):
            return arg
    return None


if __name__ == "__main__":
    _fix_windows_taskbar_icon()

    app = QApplication(sys.argv)
    apply_theme(app)

    app_icon = QIcon(LOGO_PATH)
    app.setWindowIcon(app_icon)

    # Startup splash: assets/loading_banner.png, scaled to the current
    # display (see interface/gui/splash.py) rather than one fixed size,
    # so it looks intentional on anything from a small laptop panel to a
    # large or high-DPI monitor.
    splash = None
    splash_shown_at = None
    splash_pixmap = build_splash_pixmap(app.primaryScreen())
    if not splash_pixmap.isNull():
        splash = QSplashScreen(splash_pixmap, Qt.WindowStaysOnTopHint)
        splash.show()
        app.processEvents()
        splash_shown_at = time.monotonic()

    window = MainWindow()
    window.setWindowIcon(app_icon)

    project_path = _requested_project_path()
    if project_path:
        window._load_project_file(project_path)

    if splash is not None:
        # Hold the splash up for a minimum stretch even if the window
        # finished building almost instantly, so it reads as a
        # deliberate brand moment rather than a flicker.
        elapsed_ms = (time.monotonic() - splash_shown_at) * 1000
        remaining_ms = MIN_VISIBLE_MS - elapsed_ms
        if remaining_ms > 0:
            time.sleep(remaining_ms / 1000)

    window.show()
    if splash is not None:
        splash.finish(window)

    sys.exit(app.exec())
