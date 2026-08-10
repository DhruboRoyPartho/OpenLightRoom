import math
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QPen, QFont
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from interface.gui.theme import BG_FIELD, BORDER, BORDER_LIGHT, TEXT_DIM, ACCENT

PIXELS_PER_DEGREE = 6
MAJOR_TICK_EVERY = 5
MIN_ANGLE = -45.0
MAX_ANGLE = 45.0


class AngleRuler(QWidget):
    """A horizontal degree ruler for precise straighten control, Lightroom
    style: the current angle always sits under the fixed center indicator,
    and dragging scrolls the tick marks under it like a dial. Complements
    dragging directly on the image (which sets the angle via a computed
    line) and the numeric spinbox (exact entry) - all three stay in sync
    through the shared angle value in ImageViewer."""

    angleChanged = Signal(float)   # live, while dragging
    editingFinished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(34)
        self.setMinimumWidth(180)
        self.setMouseTracking(True)
        self.setCursor(Qt.SizeHorCursor)
        self._value = 0.0
        self._dragging = False
        self._drag_start_x = None
        self._drag_start_value = 0.0

    def value(self) -> float:
        return self._value

    def set_value(self, value: float, emit: bool = True):
        clamped = max(MIN_ANGLE, min(MAX_ANGLE, value))
        if clamped == self._value and not emit:
            return
        self._value = clamped
        self.update()
        if emit:
            self.angleChanged.emit(self._value)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(BG_FIELD))

        mid_x = self.width() / 2.0
        mid_y = self.height() / 2.0

        font = QFont(painter.font())
        font.setPointSize(max(font.pointSize() - 2, 7))
        painter.setFont(font)

        start_deg = self._value - (mid_x / PIXELS_PER_DEGREE)
        end_deg = self._value + (mid_x / PIXELS_PER_DEGREE)
        first_tick = int(math.floor(start_deg))
        last_tick = int(math.ceil(end_deg))

        for deg in range(first_tick, last_tick + 1):
            x = mid_x + (deg - self._value) * PIXELS_PER_DEGREE
            if x < -1 or x > self.width() + 1:
                continue
            is_major = deg % MAJOR_TICK_EVERY == 0
            tick_h = 13 if is_major else 6
            painter.setPen(QPen(QColor(BORDER_LIGHT if is_major else BORDER), 1))
            painter.drawLine(QPointF(x, mid_y - tick_h / 2), QPointF(x, mid_y + tick_h / 2))
            if is_major:
                painter.setPen(QColor(TEXT_DIM))
                painter.drawText(QRectF(x - 14, mid_y + tick_h / 2 + 1, 28, 12), Qt.AlignCenter, str(deg))

        painter.setPen(QPen(QColor(ACCENT), 2))
        painter.drawLine(QPointF(mid_x, 2), QPointF(mid_x, self.height() - 2))

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        self._dragging = True
        self._drag_start_x = event.position().x()
        self._drag_start_value = self._value

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return
        dx = event.position().x() - self._drag_start_x
        # Dragging left scrolls the ruler's ticks right under the fixed
        # center (like spinning a dial), increasing the angle.
        self.set_value(self._drag_start_value - dx / PIXELS_PER_DEGREE)

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            self.editingFinished.emit()

    def mouseDoubleClickEvent(self, event):
        self.set_value(0.0)
        self.editingFinished.emit()
