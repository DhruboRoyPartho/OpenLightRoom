import math

import cv2
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QPainter, QColor, QPen, QImage, QRadialGradient

from interface.gui.theme import BG_FIELD, BORDER, BORDER_LIGHT, TEXT_DIM, HANDLE

WHEEL_MARGIN = 6
PUCK_RADIUS = 7
PUCK_RADIUS_ACTIVE = 8.5
CENTER_DOT_RADIUS = 2.5


def _hue_wheel_image(diameter: int) -> QImage:
    """A diameter x diameter RGBA image of a full-saturation HSV color
    wheel (hue = angle, saturation = distance from center), used as the
    wheel's background so the puck's position visually communicates the
    hue/chroma it represents. Built once per size with cv2/numpy (fast,
    vectorized) rather than per-pixel Python loops."""
    radius = diameter / 2.0
    yy, xx = np.mgrid[0:diameter, 0:diameter].astype(np.float32)
    dx = xx - radius + 0.5
    dy = yy - radius + 0.5
    dist = np.sqrt(dx * dx + dy * dy)
    angle_deg = (np.degrees(np.arctan2(-dy, dx))) % 360.0

    hue = angle_deg
    sat = np.clip(dist / radius, 0.0, 1.0)
    val = np.ones_like(hue)

    hsv = np.stack([hue, sat, val], axis=-1).astype(np.float32)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgba = np.zeros((diameter, diameter, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(dist <= radius, 255, 0).astype(np.uint8)

    image = QImage(rgba.data, diameter, diameter, diameter * 4, QImage.Format_RGBA8888)
    return image.copy()  # detach from the numpy buffer before it's freed


class ColorWheelWidget(QWidget):
    """A draggable hue/chroma picker, Lightroom-color-grading-style: the
    puck's angle from center is hue, its distance from center is chroma
    (0 at the center, 1 at the rim). Dragging emits valueChanged live;
    editingStarted/editingFinished bracket one gesture so the owner can
    commit exactly one undo step per drag, mirroring CurveWidget.
    """

    editingStarted = Signal()
    valueChanged = Signal(float, float)   # hue_deg, chroma (0..1)
    editingFinished = Signal()

    def __init__(self, label: str = "", parent=None):
        super().__init__(parent)
        self.setMinimumSize(96, 96)
        self._label = label
        self._hue_deg = 0.0
        self._chroma = 0.0     # 0..1
        self._dragging = False
        self._wheel_cache = None
        self._wheel_cache_size = -1

    def set_value(self, hue_deg: float, chroma: float):
        self._hue_deg = hue_deg % 360.0
        self._chroma = max(0.0, min(1.0, chroma))
        self.update()

    def value(self):
        return self._hue_deg, self._chroma

    # --- geometry -------------------------------------------------------

    def _wheel_rect(self):
        full = self.rect()
        label_h = 14 if self._label else 0
        side = min(full.width(), full.height() - label_h) - 2 * WHEEL_MARGIN
        side = max(side, 20)
        x = full.left() + (full.width() - side) // 2
        y = full.top() + WHEEL_MARGIN
        return QRectF(x, y, side, side)

    def _center_and_radius(self, rect: QRectF):
        return rect.center(), rect.width() / 2.0

    def _puck_pos(self, rect: QRectF):
        center, radius = self._center_and_radius(rect)
        theta = math.radians(self._hue_deg)
        r = self._chroma * radius
        return QPointF(center.x() + r * math.cos(theta), center.y() - r * math.sin(theta))

    def _value_from_pos(self, pos: QPointF, rect: QRectF):
        center, radius = self._center_and_radius(rect)
        dx = pos.x() - center.x()
        dy = center.y() - pos.y()
        hue = math.degrees(math.atan2(dy, dx)) % 360.0
        chroma = min(1.0, math.hypot(dx, dy) / max(radius, 1e-6))
        return hue, chroma

    # --- painting ---------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self._wheel_rect()
        diameter = max(int(rect.width()), 2)
        if self._wheel_cache is None or self._wheel_cache_size != diameter:
            self._wheel_cache = _hue_wheel_image(diameter)
            self._wheel_cache_size = diameter

        painter.setPen(QPen(QColor(BORDER), 1))
        painter.setBrush(QColor(BG_FIELD))
        painter.drawEllipse(rect)

        painter.save()
        painter.setClipRegion(_ellipse_region(rect.adjusted(1, 1, -1, -1)))
        painter.drawImage(rect, self._wheel_cache)
        painter.restore()

        painter.setPen(QPen(QColor(BORDER_LIGHT), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(rect)

        center, _ = self._center_and_radius(rect)
        painter.setPen(QPen(QColor(TEXT_DIM), 1))
        painter.setBrush(QColor(TEXT_DIM))
        painter.drawEllipse(center, CENTER_DOT_RADIUS, CENTER_DOT_RADIUS)

        puck = self._puck_pos(rect)
        radius = PUCK_RADIUS_ACTIVE if self._dragging else PUCK_RADIUS
        painter.setPen(QPen(QColor("#1a1a1a"), 2))
        painter.setBrush(QColor(HANDLE))
        painter.drawEllipse(puck, radius, radius)

        if self._label:
            painter.setPen(QColor(TEXT_DIM))
            font = painter.font()
            font.setPointSize(max(font.pointSize() - 2, 7))
            painter.setFont(font)
            label_rect = QRectF(self.rect().left(), rect.bottom() + 1, self.rect().width(), 12)
            painter.drawText(label_rect, Qt.AlignHCenter | Qt.AlignTop, self._label)

    # --- mouse interaction --------------------------------------------------

    def _apply_pos(self, pos):
        rect = self._wheel_rect()
        hue, chroma = self._value_from_pos(pos, rect)
        self._hue_deg, self._chroma = hue, chroma
        self.valueChanged.emit(hue, chroma)
        self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        self._dragging = True
        self.editingStarted.emit()
        self._apply_pos(event.position())

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return
        self._apply_pos(event.position())

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or not self._dragging:
            return
        self._dragging = False
        self.editingFinished.emit()
        self.update()

    def mouseDoubleClickEvent(self, event):
        # Double-click resets to center (hue meaningless at chroma 0).
        self.editingStarted.emit()
        self._hue_deg, self._chroma = 0.0, 0.0
        self.valueChanged.emit(0.0, 0.0)
        self.editingFinished.emit()
        self.update()


def _ellipse_region(rect: QRectF):
    from PySide6.QtGui import QRegion
    return QRegion(rect.toRect(), QRegion.Ellipse)
