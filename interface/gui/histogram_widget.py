import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRect, QRectF
from PySide6.QtGui import QPainter, QColor, QPen, QPainterPath

from core.scopes.histogram import compute_rgb_histogram, compute_luminance_histogram
from interface.gui.theme import BG_FIELD, BORDER, BORDER_LIGHT, TEXT_DIM

CARD_MARGIN = 4
CARD_RADIUS = 6
PLOT_MARGIN = 4
BINS = 256

CHANNEL_COLORS = {
    "R": QColor(232, 99, 95, 160),
    "G": QColor(95, 213, 135, 160),
    "B": QColor(95, 159, 232, 160),
}
LUMA_COLOR = QColor(220, 220, 220, 90)


class HistogramWidget(QWidget):
    """A full RGB + Luminance histogram, drawn as overlaid translucent
    fills (the standard "RGB histogram" look) with a Luminance outline
    behind them - recomputed from whatever image is handed to
    set_image(), so it always reflects the actual processed/rendered
    output rather than a cached approximation.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 110)
        self._rgb_hist = None
        self._luma_hist = None

    def set_image(self, image: np.ndarray):
        """image: float32 RGB, HxWx3, in [0, 1] (the document's rendered,
        display-referred output)."""
        if image is None:
            self._rgb_hist = None
            self._luma_hist = None
        else:
            self._rgb_hist = compute_rgb_histogram(image, bins=BINS)
            self._luma_hist = compute_luminance_histogram(image, bins=BINS)
        self.update()

    def _card_rect(self):
        full = self.rect()
        return QRect(
            full.left() + CARD_MARGIN, full.top() + CARD_MARGIN,
            full.width() - 2 * CARD_MARGIN, full.height() - 2 * CARD_MARGIN,
        )

    def _plot_rect(self):
        return self._card_rect().adjusted(PLOT_MARGIN, PLOT_MARGIN, -PLOT_MARGIN, -PLOT_MARGIN)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        card = self._card_rect()
        card_path = QPainterPath()
        card_path.addRoundedRect(QRectF(card), CARD_RADIUS, CARD_RADIUS)
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.setBrush(QColor(BG_FIELD))
        painter.drawPath(card_path)

        if self._rgb_hist is None:
            painter.setPen(QColor(TEXT_DIM))
            painter.drawText(card, Qt.AlignCenter, "No image")
            return

        painter.save()
        painter.setClipPath(card_path)
        rect = self._plot_rect()

        if self._luma_hist is not None:
            self._draw_fill(painter, rect, self._luma_hist, LUMA_COLOR, outline_only=True)
        for ch in ("R", "G", "B"):
            self._draw_fill(painter, rect, self._rgb_hist[ch], CHANNEL_COLORS[ch])

        painter.restore()
        painter.setPen(QPen(QColor(BORDER_LIGHT), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(card_path)

    def _draw_fill(self, painter, rect, counts, color, outline_only=False):
        peak = max(int(counts.max()), 1)
        # Log scale: a histogram's tallest bin (often flat sky/skin) would
        # otherwise dwarf everything else into invisible flat lines.
        log_counts = np.log1p(counts.astype(np.float64))
        log_peak = max(log_counts.max(), 1e-6)

        n = len(counts)
        path = QPainterPath()
        x0 = rect.left()
        y0 = rect.bottom()
        path.moveTo(x0, y0)
        for i in range(n):
            x = rect.left() + rect.width() * i / (n - 1)
            y = rect.bottom() - rect.height() * (log_counts[i] / log_peak)
            path.lineTo(x, y)
        path.lineTo(rect.right(), rect.bottom())
        path.closeSubpath()

        if outline_only:
            pen = QPen(color, 1.5)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
        else:
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
        painter.drawPath(path)
