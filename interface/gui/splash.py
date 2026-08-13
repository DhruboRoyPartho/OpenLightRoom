"""Startup splash screen sizing: scales assets/loading_banner.png
proportionally to the current display instead of showing it at one fixed
size, so it reads correctly on a small laptop panel, a large desktop
monitor, or a high-DPI/4K screen alike.
"""

from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

from interface.gui.assets import LOADING_BANNER_PATH

MIN_WIDTH = 480     # never render smaller than this, even on a tiny display
MAX_WIDTH = 900     # never render larger than this, even on an ultrawide/4K display
SCREEN_WIDTH_FRACTION = 0.42
MIN_VISIBLE_MS = 5000   # keep the splash up at least this long, even on a fast local launch


def build_splash_pixmap(screen) -> QPixmap:
    """screen: a QScreen (e.g. QApplication.primaryScreen()). Returns a
    null QPixmap if the banner asset is missing, so callers can skip
    showing a splash entirely rather than displaying a blank window."""
    source = QPixmap(LOADING_BANNER_PATH)
    if source.isNull() or screen is None:
        return source

    available = screen.availableGeometry()
    device_ratio = screen.devicePixelRatio() or 1.0

    logical_width = max(MIN_WIDTH, min(MAX_WIDTH, round(available.width() * SCREEN_WIDTH_FRACTION)))
    physical_width = max(1, round(logical_width * device_ratio))

    # Scale at the display's actual pixel density (not just its logical
    # size) so the banner stays crisp on HiDPI screens instead of being
    # upscaled from a lower-resolution render.
    pixmap = source.scaledToWidth(physical_width, Qt.SmoothTransformation)
    pixmap.setDevicePixelRatio(device_ratio)
    return pixmap
