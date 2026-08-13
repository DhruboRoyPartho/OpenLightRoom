import time
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QPainter, QColor, QPen

from interface.gui.theme import TEXT_DIM, ACCENT

# A task that finishes inside SHOW_DELAY_MS never causes so much as a
# flash - only genuinely slow operations (a big render, a RAW decode, an
# export) make the indicator appear at all, which is what keeps it from
# "disturbing the user" on the common case of a near-instant update.
SHOW_DELAY_MS = 200
# Once shown, stays up at least this long from the moment it appeared, so
# a task that finishes just after crossing the show-delay doesn't blink
# for a single frame.
MIN_VISIBLE_MS = 400
SPIN_INTERVAL_MS = 16
SPIN_STEP_DEG = 6
ARC_SPAN_DEG = 100


class _Spinner(QWidget):
    """The actual rotating-arc graphic. Kept separate from LoadingIndicator
    so the repaint timer only runs while the indicator is actually
    visible."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(14, 14)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(SPIN_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)

    def start(self):
        self._angle = 0
        self._timer.start()

    def stop(self):
        self._timer.stop()

    def _tick(self):
        self._angle = (self._angle + SPIN_STEP_DEG) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(2, 2, self.width() - 4, self.height() - 4)

        track_color = QColor(TEXT_DIM)
        track_color.setAlpha(70)
        track_pen = QPen(track_color)
        track_pen.setWidthF(2.0)
        painter.setPen(track_pen)
        painter.drawArc(rect, 0, 360 * 16)

        arc_pen = QPen(QColor(ACCENT))
        arc_pen.setWidthF(2.0)
        arc_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(arc_pen)
        painter.drawArc(rect, -self._angle * 16, ARC_SPAN_DEG * 16)


class LoadingIndicator(QWidget):
    """A small, unobtrusive "working" indicator: a spinning ring plus a
    short label (e.g. "Rendering...", "Importing..."). Meant to live in
    the status bar's permanent-widget area - the one part of the window
    that's otherwise always empty - so it never overlaps or displaces any
    control. Purely informational: it never blocks input, never appears
    as a modal, and only shows itself for operations slow enough to
    actually notice (see SHOW_DELAY_MS/MIN_VISIBLE_MS above).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 10, 0)
        layout.setSpacing(6)

        self._spinner = _Spinner()
        layout.addWidget(self._spinner)

        self._label = QLabel()
        self._label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        layout.addWidget(self._label)

        self.setVisible(False)
        self._shown_at = None

        self._show_timer = QTimer(self)
        self._show_timer.setSingleShot(True)
        self._show_timer.setInterval(SHOW_DELAY_MS)
        self._show_timer.timeout.connect(self._show_now)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._hide_now)

    def attach(self, tracker) -> None:
        tracker.busyChanged.connect(self._on_busy_changed)

    def show_now(self, label: str) -> None:
        """Bypasses the show-delay entirely - for a one-shot, already-known-
        to-be-slow blocking action (import/export/project save-load)
        rather than the frequent, often-instant render-queue traffic
        attach() listens to. Call hide_now() when the action finishes."""
        self._show_timer.stop()
        self._hide_timer.stop()
        self._label.setText(label)
        self._show_now()

    def hide_now(self) -> None:
        self._show_timer.stop()
        self._hide_timer.stop()
        self._hide_now()

    def _on_busy_changed(self, is_busy: bool, label: str) -> None:
        if is_busy:
            self._hide_timer.stop()
            self._label.setText(label)
            if self.isVisible():
                return  # already showing - label just updated above
            if not self._show_timer.isActive():
                self._show_timer.start()
        else:
            self._show_timer.stop()
            if not self.isVisible():
                return  # never crossed the show-delay - nothing to hide
            elapsed_ms = (time.monotonic() - self._shown_at) * 1000.0
            remaining = max(0.0, MIN_VISIBLE_MS - elapsed_ms)
            self._hide_timer.start(int(remaining))

    def _show_now(self) -> None:
        self._spinner.start()
        self.setVisible(True)
        self._shown_at = time.monotonic()

    def _hide_now(self) -> None:
        self._spinner.stop()
        self.setVisible(False)
        self._shown_at = None
