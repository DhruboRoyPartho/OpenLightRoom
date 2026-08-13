from PySide6.QtWidgets import QLabel, QSlider, QSpinBox, QPushButton, QHBoxLayout, QStyle
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QPainter, QColor, QPen

from interface.gui.theme import (
    TRACK, ACCENT, HANDLE, HANDLE_BORDER as HANDLE_BORDER_COLOR,
    TEXT, TEXT_DIM, BORDER_LIGHT, ACCENT as ACCENT_COLOR,
)

SPINBOX_STYLE = f"""
    QSpinBox {{
        font-family: Consolas, "Courier New", monospace;
        color: {TEXT};
        background: transparent;
        border: 1px solid transparent;
        border-radius: 3px;
        padding: 1px 3px;
    }}
    QSpinBox:hover {{
        border: 1px solid {BORDER_LIGHT};
    }}
    QSpinBox:focus {{
        border: 1px solid {ACCENT_COLOR};
    }}
"""

RESET_BUTTON_STYLE = f"""
    QPushButton {{
        color: {TEXT_DIM};
        background: transparent;
        border: none;
        border-radius: 3px;
        font-size: 13px;
        padding: 0px;
    }}
    QPushButton:hover:enabled {{
        color: {TEXT};
        background-color: #3d3d3d;
    }}
    QPushButton:disabled {{
        color: #4a4a4a;
    }}
"""


class CenterFillSlider(QSlider):
    """A slider that fills from its default value's position rather than from
    the left edge (Lightroom-style), so it's obvious at a glance whether an
    attribute is pushed above or below its default. Only paintEvent is
    overridden - all of QSlider's mouse/keyboard handling and signals
    (valueChanged, sliderPressed, sliderReleased) work exactly as normal.

    Shared by ControlsPanel and MasksPanel (and anywhere else a numeric
    slider is needed) so every control in the app looks and behaves
    identically.
    """

    TRACK_COLOR = QColor(TRACK)
    FILL_COLOR = QColor(ACCENT)
    HANDLE_COLOR = QColor(HANDLE)
    HANDLE_BORDER = QColor(HANDLE_BORDER_COLOR)
    HANDLE_DIAMETER = 14
    TRACK_THICKNESS = 4

    def __init__(self, orientation, default_value=0, parent=None):
        super().__init__(orientation, parent)
        self.default_value = default_value
        self.setFixedHeight(22)

    def _value_to_x(self, value, left, available):
        return left + QStyle.sliderPositionFromValue(self.minimum(), self.maximum(), value, available)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        half = self.HANDLE_DIAMETER // 2
        left = half
        available = max(self.width() - self.HANDLE_DIAMETER, 1)
        mid_y = self.height() // 2

        painter.setPen(Qt.NoPen)
        painter.setBrush(self.TRACK_COLOR)
        track_rect = QRect(left, mid_y - self.TRACK_THICKNESS // 2, available, self.TRACK_THICKNESS)
        painter.drawRoundedRect(track_rect, 2, 2)

        default_x = self._value_to_x(self.default_value, left, available)
        value_x = self._value_to_x(self.value(), left, available)
        fill_width = abs(value_x - default_x)
        if fill_width > 0:
            fill_rect = QRect(min(default_x, value_x), mid_y - self.TRACK_THICKNESS // 2, fill_width, self.TRACK_THICKNESS)
            painter.setBrush(self.FILL_COLOR)
            painter.drawRoundedRect(fill_rect, 2, 2)

        painter.setBrush(self.HANDLE_COLOR)
        painter.setPen(QPen(self.HANDLE_BORDER, 1))
        painter.drawEllipse(value_x - half, mid_y - half, self.HANDLE_DIAMETER, self.HANDLE_DIAMETER)


def build_slider_row(layout, display_label, minv, maxv, default,
                      label_width=80, slider_width=150, spinbox_width=50):
    """Constructs a labeled slider + spinbox + reset button row and adds
    it to layout. Returns (slider, spinbox, reset_btn) - the caller wires
    up value-change/commit/reset behavior."""
    row = QHBoxLayout()
    row.setSpacing(4)
    label = QLabel(display_label)
    label.setFixedWidth(label_width)
    row.addWidget(label)

    slider = CenterFillSlider(Qt.Horizontal, default_value=default)
    slider.setRange(minv, maxv)
    slider.setValue(default)
    slider.setFixedWidth(slider_width)
    row.addWidget(slider)

    spinbox = QSpinBox()
    spinbox.setRange(minv, maxv)
    spinbox.setValue(default)
    spinbox.setFixedWidth(spinbox_width)
    spinbox.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    spinbox.setButtonSymbols(QSpinBox.NoButtons)
    spinbox.setStyleSheet(SPINBOX_STYLE)
    row.addWidget(spinbox)

    reset_btn = QPushButton("↺")  # anticlockwise open circle arrow
    reset_btn.setFixedSize(20, 20)
    reset_btn.setToolTip(f"Reset {display_label} to default")
    reset_btn.setCursor(Qt.PointingHandCursor)
    reset_btn.setStyleSheet(RESET_BUTTON_STYLE)
    reset_btn.setEnabled(False)
    row.addWidget(reset_btn)

    layout.addLayout(row)

    slider.valueChanged.connect(spinbox.setValue)
    spinbox.valueChanged.connect(slider.setValue)
    slider.valueChanged.connect(lambda v, btn=reset_btn, d=default: btn.setEnabled(v != d))

    return slider, spinbox, reset_btn
