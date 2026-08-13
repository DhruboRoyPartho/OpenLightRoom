"""Resolves paths to the app's bundled assets (logo, startup banner)
relative to the project root, independent of the process's current
working directory (so `python main.py` works the same whether launched
from the project root or anywhere else).

Also frozen-app aware: a PyInstaller build doesn't preserve this file's
real on-disk location the way a normal Python source tree does, so
`__file__`-relative resolution only applies when running from source.
When frozen, `sys.frozen` is set and the assets live next to the
executable (onedir) or get unpacked to `sys._MEIPASS` (onefile) instead -
see packaging/OpenLightRoom.spec's `datas` entry, which copies the
`assets/` folder to the bundle root under either mode.
"""

import os
import sys


def _project_root() -> str:
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    gui_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(gui_dir))


PROJECT_ROOT = _project_root()
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")

LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
LOADING_BANNER_PATH = os.path.join(ASSETS_DIR, "loading_banner.png")
