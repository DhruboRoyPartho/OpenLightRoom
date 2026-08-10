from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal, QPointF, QRect, QRectF
from PySide6.QtGui import QPainter, QColor, QPen, QPainterPath, QFont
from core.processing.curve import build_lut, IDENTITY_POINTS
from interface.gui.theme import BG_FIELD, BG_PANEL, BORDER, BORDER_LIGHT, TEXT, TEXT_DIM, HANDLE

CHANNEL_COLORS = {
    "RGB": QColor(TEXT),
    "Red": QColor("#e8635f"),
    "Green": QColor("#5fd587"),
    "Blue": QColor("#5f9fe8"),
}

GRID_DIVISIONS = 4
POINT_RADIUS = 4.5
POINT_RADIUS_HOVER = 6
POINT_RADIUS_ACTIVE = 6.5
HIT_RADIUS = 10

CARD_MARGIN = 4      # gap between the widget's edge and the rounded card
CARD_RADIUS = 8       # corner radius of the card
PLOT_MARGIN = 16      # gap between the card edge and the plottable area,
                       # room for axis labels and point overflow


class CurveWidget(QWidget):
    """An interactive point-based tone curve editor, Lightroom-style: click
    the curve to add a point, drag a point to reshape it, double-click a
    point to remove it. Endpoints (x=0 and x=255) can be dragged vertically
    (raising/lowering black and white points) but not removed or moved off
    the ends of the input range.

    editingStarted/editingFinished bracket a single drag or click gesture so
    the owner can commit exactly one undo step per gesture, mirroring how
    the sliders elsewhere in this panel behave.
    """

    editingStarted = Signal()
    pointsChanged = Signal(list)   # emitted live, with the active channel's points
    editingFinished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 220)
        self.setMouseTracking(True)
        self._points_by_channel = {ch: list(IDENTITY_POINTS) for ch in CHANNEL_COLORS}
        self._channel = "RGB"
        self._dragging_index = None
        self._hover_index = None
        self._interactive = True
        self._readonly_lut = None   # when set, overrides the point-spline display (Parametric Curve preview)

    def set_interactive(self, enabled: bool):
        """Disables all mouse editing - used for a read-only curve preview
        (e.g. the Parametric Curve's resulting shape)."""
        self._interactive = enabled
        if not enabled:
            self._dragging_index = None
            self._hover_index = None
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def set_preview_lut(self, lut):
        """lut: a 256-entry array of y-values (0-255) to draw directly,
        bypassing the point-spline. No control points are drawn."""
        self._readonly_lut = lut
        self.update()

    def set_channel(self, channel: str):
        self._channel = channel
        self._dragging_index = None
        self._hover_index = None
        self.update()

    def set_points(self, channel: str, points):
        self._points_by_channel[channel] = sorted((int(x), int(y)) for x, y in points)
        if channel == self._channel:
            self.update()

    def points(self, channel: str = None):
        return list(self._points_by_channel[channel or self._channel])

    # --- geometry -----------------------------------------------------

    def _card_rect(self):
        """The largest square that fits the widget, centered, so the curve
        always reads as a proper square graph regardless of the panel's
        actual width."""
        full = self.rect()
        side = min(full.width(), full.height()) - 2 * CARD_MARGIN
        side = max(side, 40)
        x = full.left() + (full.width() - side) // 2
        y = full.top() + (full.height() - side) // 2
        return QRect(x, y, side, side)

    def _plot_rect(self):
        return self._card_rect().adjusted(PLOT_MARGIN, PLOT_MARGIN, -PLOT_MARGIN, -PLOT_MARGIN)

    def _to_widget(self, x, y, rect):
        wx = rect.left() + (x / 255.0) * rect.width()
        wy = rect.bottom() - (y / 255.0) * rect.height()
        return wx, wy

    def _to_value(self, wx, wy, rect):
        x = (wx - rect.left()) / max(rect.width(), 1) * 255.0
        y = (rect.bottom() - wy) / max(rect.height(), 1) * 255.0
        return max(0, min(255, round(x))), max(0, min(255, round(y)))

    def _hit_test(self, pos: QPointF, points, rect):
        for i, (x, y) in enumerate(points):
            wx, wy = self._to_widget(x, y, rect)
            if (pos.x() - wx) ** 2 + (pos.y() - wy) ** 2 <= HIT_RADIUS ** 2:
                return i
        return None

    # --- painting -------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.fillRect(self.rect(), QColor(BG_PANEL))

        card = self._card_rect()
        card_path = QPainterPath()
        card_path.addRoundedRect(QRectF(card), CARD_RADIUS, CARD_RADIUS)
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.setBrush(QColor(BG_FIELD))
        painter.drawPath(card_path)

        painter.save()
        painter.setClipPath(card_path)

        rect = self._plot_rect()
        self._draw_grid(painter, rect)
        self._draw_curve(painter, rect)
        if self._readonly_lut is None:
            self._draw_points(painter, rect)

        painter.restore()

        self._draw_axis_labels(painter, card, rect)
        if self._readonly_lut is None:
            self._draw_readout(painter, card)

    def _draw_grid(self, painter, rect):
        painter.setPen(QPen(QColor(BORDER), 1))
        for i in range(1, GRID_DIVISIONS):
            fx = rect.left() + rect.width() * i / GRID_DIVISIONS
            fy = rect.top() + rect.height() * i / GRID_DIVISIONS
            painter.drawLine(int(fx), rect.top(), int(fx), rect.bottom())
            painter.drawLine(rect.left(), int(fy), rect.right(), int(fy))

        painter.setPen(QPen(QColor(BORDER_LIGHT), 1))
        painter.drawRect(rect)

        pen_ref = QPen(QColor(BORDER_LIGHT), 1)
        pen_ref.setStyle(Qt.DashLine)
        painter.setPen(pen_ref)
        x0, y0 = self._to_widget(0, 0, rect)
        x1, y1 = self._to_widget(255, 255, rect)
        painter.drawLine(int(x0), int(y0), int(x1), int(y1))

    def _draw_curve(self, painter, rect):
        curve_color = CHANNEL_COLORS.get(self._channel, QColor(TEXT))
        if self._readonly_lut is not None:
            lut = self._readonly_lut
        else:
            points = self._points_by_channel[self._channel]
            lut = build_lut(points)

        path = QPainterPath()
        x0, y0 = self._to_widget(0, lut[0], rect)
        path.moveTo(x0, y0)
        for i in range(1, 256):
            path.lineTo(*self._to_widget(i, lut[i], rect))

        # Soft glow under the line for a bit of depth, then a crisp stroke.
        glow_color = QColor(curve_color)
        glow_color.setAlpha(50)
        glow_pen = QPen(glow_color, 5)
        glow_pen.setCapStyle(Qt.RoundCap)
        glow_pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(glow_pen)
        painter.drawPath(path)

        main_pen = QPen(curve_color, 2)
        main_pen.setCapStyle(Qt.RoundCap)
        main_pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(main_pen)
        painter.drawPath(path)

    def _draw_points(self, painter, rect):
        points = self._points_by_channel[self._channel]
        curve_color = CHANNEL_COLORS.get(self._channel, QColor(TEXT))

        for i, (x, y) in enumerate(points):
            wx, wy = self._to_widget(x, y, rect)
            active = i == self._dragging_index
            hovered = i == self._hover_index

            radius = POINT_RADIUS_ACTIVE if active else (POINT_RADIUS_HOVER if hovered else POINT_RADIUS)

            if active or hovered:
                glow = QColor(curve_color)
                glow.setAlpha(70)
                painter.setPen(Qt.NoPen)
                painter.setBrush(glow)
                painter.drawEllipse(QPointF(wx, wy), radius + 5, radius + 5)

            painter.setPen(QPen(curve_color, 2 if (active or hovered) else 1.5))
            painter.setBrush(QColor(HANDLE))
            painter.drawEllipse(QPointF(wx, wy), radius, radius)

    def _draw_axis_labels(self, painter, card, rect):
        font = QFont(painter.font())
        font.setPointSize(max(font.pointSize() - 2, 7))
        painter.setFont(font)
        painter.setPen(QColor(TEXT_DIM))

        painter.drawText(QRectF(card.left() + 2, rect.bottom() + 2, PLOT_MARGIN + 20, PLOT_MARGIN - 2),
                          Qt.AlignLeft | Qt.AlignTop, "0")
        painter.drawText(QRectF(card.right() - PLOT_MARGIN - 22, rect.bottom() + 2, PLOT_MARGIN + 20, PLOT_MARGIN - 2),
                          Qt.AlignRight | Qt.AlignTop, "255")
        painter.drawText(QRectF(card.left() + 2, card.top() + 2, PLOT_MARGIN - 2, PLOT_MARGIN + 20),
                          Qt.AlignLeft | Qt.AlignTop, "255")
        painter.drawText(QRectF(card.left() + 2, rect.bottom() - PLOT_MARGIN - 16, PLOT_MARGIN - 2, PLOT_MARGIN + 20),
                          Qt.AlignLeft | Qt.AlignBottom, "0")

    def _draw_readout(self, painter, card):
        active_index = self._dragging_index if self._dragging_index is not None else self._hover_index
        if active_index is None:
            return
        points = self._points_by_channel[self._channel]
        if active_index >= len(points):
            return
        x, y = points[active_index]

        font = QFont(painter.font())
        font.setFamily("Consolas")
        font.setPointSize(max(font.pointSize() - 1, 8))
        painter.setFont(font)
        painter.setPen(QColor(TEXT))
        text_rect = QRectF(card.left(), card.top() + 4, card.width() - 8, 16)
        painter.drawText(text_rect, Qt.AlignRight | Qt.AlignTop, f"{x}  →  {y}")

    # --- mouse interaction -----------------------------------------------

    def mousePressEvent(self, event):
        if not self._interactive or event.button() != Qt.LeftButton:
            return
        rect = self._plot_rect()
        points = list(self._points_by_channel[self._channel])
        idx = self._hit_test(event.position(), points, rect)

        if idx is None:
            x, y = self._to_value(event.position().x(), event.position().y(), rect)
            if x <= points[0][0] or x >= points[-1][0]:
                return  # can't add a point past the fixed endpoints
            points.append((x, y))
            points.sort(key=lambda p: p[0])
            idx = points.index((x, y))
            self._points_by_channel[self._channel] = points

        self._dragging_index = idx
        self._hover_index = idx
        self.editingStarted.emit()
        self.update()

    def mouseMoveEvent(self, event):
        if not self._interactive:
            return
        rect = self._plot_rect()

        if self._dragging_index is None:
            idx = self._hit_test(event.position(), self._points_by_channel[self._channel], rect)
            if idx != self._hover_index:
                self._hover_index = idx
                self.update()
            self.setCursor(Qt.PointingHandCursor if idx is not None else Qt.CrossCursor)
            return

        points = list(self._points_by_channel[self._channel])
        x, y = self._to_value(event.position().x(), event.position().y(), rect)

        is_endpoint = self._dragging_index in (0, len(points) - 1)
        if is_endpoint:
            x = points[self._dragging_index][0]  # endpoints only move vertically
        else:
            left_x = points[self._dragging_index - 1][0] + 1
            right_x = points[self._dragging_index + 1][0] - 1
            x = max(left_x, min(right_x, x))

        points[self._dragging_index] = (x, y)
        self._points_by_channel[self._channel] = points
        self.pointsChanged.emit(points)
        self.update()

    def mouseReleaseEvent(self, event):
        if not self._interactive or event.button() != Qt.LeftButton or self._dragging_index is None:
            return
        self._dragging_index = None
        self.editingFinished.emit()
        self.update()

    def mouseDoubleClickEvent(self, event):
        if not self._interactive:
            return
        self._dragging_index = None  # cancel the drag state from the preceding press
        rect = self._plot_rect()
        points = self._points_by_channel[self._channel]
        idx = self._hit_test(event.position(), points, rect)
        if idx is None or idx == 0 or idx == len(points) - 1:
            return  # nothing there, or it's a fixed endpoint

        new_points = points[:idx] + points[idx + 1:]
        self._points_by_channel[self._channel] = new_points
        self._hover_index = None
        self.editingStarted.emit()
        self.pointsChanged.emit(new_points)
        self.editingFinished.emit()
        self.update()

    def leaveEvent(self, event):
        if self._dragging_index is None and self._hover_index is not None:
            self._hover_index = None
            self.update()
        super().leaveEvent(event)
