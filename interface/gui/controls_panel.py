import numpy as np
from PySide6.QtWidgets import QWidget, QSpinBox, QComboBox, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QSlider, QStyle, QStyleOptionSlider
from PySide6.QtCore import Qt, QTimer, QEvent, QRect
from PySide6.QtGui import QPainter, QColor, QPen
from core.adjustment_layers.brightness_layer import BrightnessLayer
from core.adjustment_layers.contrast_layer import ContrastLayer
from core.adjustment_layers.temperature_layer import TemperatureLayer
from core.adjustment_layers.tint_layer import TintLayer
from core.adjustment_layers.exposure_layer import ExposureLayer
from core.adjustment_layers.highlights_layer import HighlightsLayer
from core.adjustment_layers.shadows_layer import ShadowsLayer
from core.adjustment_layers.whites_layer import WhitesLayer
from core.adjustment_layers.blacks_layer import BlacksLayer
from core.adjustment_layers.vibrance_layer import VibranceLayer
from core.adjustment_layers.saturation_layer import SaturationLayer
from core.adjustment_layers.hue_layer import HueLayer
from core.adjustment_layers.curve_layer import CurveLayer, CHANNELS as CURVE_CHANNELS
from core.adjustment_layers.parametric_curve_layer import ParametricCurveLayer
from core.processing.curve import IDENTITY_POINTS
from core.commands.change_layer_command import ChangeLayerCommand
from interface.gui.curve_widget import CurveWidget
from interface.gui.theme import (
    BG_PANEL, BORDER, BORDER_LIGHT, TEXT, TEXT_DIM, TEXT_HEADER,
    ACCENT, TRACK, HANDLE, HANDLE_BORDER as HANDLE_BORDER_COLOR,
)

PARAMETRIC_LAYER_NAME = "Parametric Curve"
PARAMETRIC_FIELDS = [
    ("Highlights", "highlights"),
    ("Lights", "lights"),
    ("Darks", "darks"),
    ("Shadows", "shadows"),
]

# How long (ms) to wait after the last value change before committing an undo
# step, when there's no explicit slider-release to commit on (e.g. spinbox
# typing or keyboard nudging). A mouse drag on the slider instead commits
# immediately on release, so it isn't affected by this delay.
COMMIT_IDLE_MS = 500


class CenterFillSlider(QSlider):
    """A slider that fills from its default value's position rather than from
    the left edge (Lightroom-style), so it's obvious at a glance whether an
    attribute is pushed above or below its default. Only paintEvent is
    overridden - all of QSlider's mouse/keyboard handling and signals
    (valueChanged, sliderPressed, sliderReleased) work exactly as normal."""

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
        border: 1px solid {ACCENT};
    }}
"""

SECTION_HEADER_STYLE = f"""
    QLabel {{
        font-weight: 600;
        font-size: 11px;
        color: {TEXT_HEADER};
        padding-top: 10px;
        padding-bottom: 4px;
        border-bottom: 1px solid {BORDER};
        margin-bottom: 4px;
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


class ControlsPanel(QWidget):
    def __init__(self, document, viewer, layer_stack_panel):
        super().__init__()
        self.document = document
        self.viewer = viewer
        self.layer_stack_panel = layer_stack_panel
        self.control_label_width = 80
        self.slider_width = 150
        self.spinbox_width = 50

        self.setStyleSheet(f"ControlsPanel {{ background-color: {BG_PANEL}; }}")

        self._pending_old_layer = {}   # name -> layer snapshot at gesture start
        self._is_dragging = {}         # name -> True while the slider handle is held
        self._commit_timers = {}       # name -> QTimer that commits idle (non-drag) edits
        self._reset_targets = {}       # QSlider -> (name, layer_cls, transform, default_raw)
        self._adjustments = []         # (name, slider, spinbox, reset_btn, layer_cls, raw->value, value->raw, default_raw)
        self._curve_channel = "RGB"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)

        layout.addWidget(self._section_header("Tone"))
        self._add_row(layout, "Exposure", "exposure", -100, 100, 0,
                       ExposureLayer, lambda v: v, lambda x: round(x))
        self._add_row(layout, "Brightness", "brightness", -100, 100, 0,
                       BrightnessLayer, lambda v: v / 100.0, lambda x: round(x * 100))
        self._add_row(layout, "Contrast", "contrast", 10, 300, 100,
                       ContrastLayer, lambda v: v / 100.0, lambda x: round(x * 100))
        self._add_row(layout, "Highlights", "highlights", -100, 100, 0,
                       HighlightsLayer, lambda v: v, lambda x: round(x))
        self._add_row(layout, "Shadows", "shadows", -100, 100, 0,
                       ShadowsLayer, lambda v: v, lambda x: round(x))
        self._add_row(layout, "Whites", "whites", -100, 100, 0,
                       WhitesLayer, lambda v: v, lambda x: round(x))
        self._add_row(layout, "Blacks", "blacks", -100, 100, 0,
                       BlacksLayer, lambda v: v, lambda x: round(x))

        layout.addWidget(self._section_header("Tone Curve"))
        self._add_curve_section(layout)

        layout.addWidget(self._section_header("Color"))
        self._add_row(layout, "Temperature", "temp", -100, 100, 0,
                       TemperatureLayer, lambda v: v, lambda x: round(x))
        self._add_row(layout, "Tint", "tint", -100, 100, 0,
                       TintLayer, lambda v: v, lambda x: round(x))
        self._add_row(layout, "Vibrance", "vibrance", -100, 100, 0,
                       VibranceLayer, lambda v: v, lambda x: round(x))
        self._add_row(layout, "Saturation", "saturation", -100, 100, 0,
                       SaturationLayer, lambda v: v, lambda x: round(x))
        self._add_row(layout, "Hue", "hue", -180, 180, 0,
                       HueLayer, lambda v: v, lambda x: round(x))

        layout.addSpacing(10)

        button_row = QHBoxLayout()
        self.undo_button = QPushButton("Undo")
        self.redo_button = QPushButton("Redo")
        button_row.addWidget(self.undo_button)
        button_row.addWidget(self.redo_button)
        layout.addLayout(button_row)

        self.delete_layer_button = QPushButton("Delete")
        # layout.addWidget(self.delete_layer_button)

        layout.addStretch(1)

        self.delete_layer_button.clicked.connect(self.delete_layer)
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)

        self._wire_adjustments()

    def _section_header(self, text):
        label = QLabel(text.upper())
        label.setStyleSheet(SECTION_HEADER_STYLE)
        return label

    def _build_slider_row(self, layout, display_label, minv, maxv, default):
        """Constructs a labeled slider + spinbox + reset button row and adds
        it to layout. Returns (slider, spinbox, reset_btn) - the caller
        wires up value-change/commit/reset behavior."""
        row = QHBoxLayout()
        row.setSpacing(4)
        label = QLabel(display_label)
        label.setFixedWidth(self.control_label_width)
        row.addWidget(label)

        slider = CenterFillSlider(Qt.Horizontal, default_value=default)
        slider.setRange(minv, maxv)
        slider.setValue(default)
        slider.setFixedWidth(self.slider_width)
        row.addWidget(slider)

        spinbox = QSpinBox()
        spinbox.setRange(minv, maxv)
        spinbox.setValue(default)
        spinbox.setFixedWidth(self.spinbox_width)
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

    def _add_row(self, layout, display_label, attr_prefix, minv, maxv, default, layer_cls, transform, inverse):
        slider, spinbox, reset_btn = self._build_slider_row(layout, display_label, minv, maxv, default)

        setattr(self, f"{attr_prefix}_slider", slider)
        setattr(self, f"{attr_prefix}_spinbox", spinbox)
        setattr(self, f"{attr_prefix}_reset_button", reset_btn)

        reset_btn.clicked.connect(
            lambda checked=False, name=display_label, layer_cls=layer_cls, transform=transform, default_raw=default:
                self._reset_attribute(name, layer_cls, transform, default_raw)
        )

        self._adjustments.append((display_label, slider, spinbox, reset_btn, layer_cls, transform, inverse, default))

    def _add_curve_section(self, layout):
        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        self.curve_point_mode_btn = QPushButton("Point")
        self.curve_point_mode_btn.setCheckable(True)
        self.curve_point_mode_btn.setChecked(True)
        self.curve_parametric_mode_btn = QPushButton("Parametric")
        self.curve_parametric_mode_btn.setCheckable(True)
        mode_row.addWidget(self.curve_point_mode_btn)
        mode_row.addWidget(self.curve_parametric_mode_btn)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        self.curve_point_mode_btn.clicked.connect(lambda: self._set_curve_mode("point"))
        self.curve_parametric_mode_btn.clicked.connect(lambda: self._set_curve_mode("parametric"))

        # --- Point curve: draggable control points on a spline, per channel ---
        self.point_curve_container = QWidget()
        point_layout = QVBoxLayout(self.point_curve_container)
        point_layout.setContentsMargins(0, 4, 0, 0)
        point_layout.setSpacing(2)

        header_row = QHBoxLayout()
        header_row.setSpacing(4)
        channel_label = QLabel("Channel")
        channel_label.setFixedWidth(self.control_label_width)
        header_row.addWidget(channel_label)

        self.curve_channel_combo = QComboBox()
        self.curve_channel_combo.addItems(list(CURVE_CHANNELS))
        header_row.addWidget(self.curve_channel_combo)
        header_row.addStretch(1)

        self.curve_reset_button = QPushButton("↺")
        self.curve_reset_button.setFixedSize(20, 20)
        self.curve_reset_button.setToolTip("Reset this channel's curve to a straight line")
        self.curve_reset_button.setCursor(Qt.PointingHandCursor)
        self.curve_reset_button.setStyleSheet(RESET_BUTTON_STYLE)
        self.curve_reset_button.setEnabled(False)
        header_row.addWidget(self.curve_reset_button)

        point_layout.addLayout(header_row)

        self.curve_widget = CurveWidget()
        self.curve_widget.setFixedHeight(260)
        point_layout.addWidget(self.curve_widget)

        layout.addWidget(self.point_curve_container)

        self.curve_channel_combo.currentTextChanged.connect(self._on_curve_channel_changed)
        self.curve_widget.editingStarted.connect(lambda: self._ensure_pending("Curve"))
        self.curve_widget.pointsChanged.connect(self._on_curve_points_changed)
        self.curve_widget.editingFinished.connect(self._on_curve_editing_finished)
        self.curve_reset_button.clicked.connect(self._on_curve_reset_clicked)

        # --- Parametric curve: region sliders shaping a smooth curve,
        # visualized read-only in a preview graph ---
        self.parametric_container = QWidget()
        parametric_layout = QVBoxLayout(self.parametric_container)
        parametric_layout.setContentsMargins(0, 4, 0, 0)
        parametric_layout.setSpacing(2)

        self.parametric_preview = CurveWidget()
        self.parametric_preview.setFixedHeight(160)
        self.parametric_preview.set_interactive(False)
        parametric_layout.addWidget(self.parametric_preview)

        self._parametric_rows = {}
        for label, field in PARAMETRIC_FIELDS:
            slider, spinbox, reset_btn = self._build_slider_row(parametric_layout, label, -100, 100, 0)
            self._parametric_rows[field] = (slider, spinbox, reset_btn)

            key = f"parametric_{field}"
            self._is_dragging[key] = False
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(COMMIT_IDLE_MS)
            timer.timeout.connect(lambda f=field: self._on_parametric_commit(f))
            self._commit_timers[key] = timer

            slider.valueChanged.connect(lambda v, f=field: self._on_parametric_value_changed(f, v))
            slider.sliderPressed.connect(lambda f=field: self._on_parametric_drag_start(f))
            slider.sliderReleased.connect(lambda f=field: self._on_parametric_drag_end(f))
            spinbox.editingFinished.connect(lambda f=field: self._on_parametric_commit(f))
            reset_btn.clicked.connect(lambda checked=False, f=field: self._on_parametric_reset(f))

        layout.addWidget(self.parametric_container)
        self.parametric_container.setVisible(False)
        self._update_parametric_preview()

    def _set_curve_mode(self, mode):
        is_point = mode == "point"
        self.curve_point_mode_btn.setChecked(is_point)
        self.curve_parametric_mode_btn.setChecked(not is_point)
        self.point_curve_container.setVisible(is_point)
        self.parametric_container.setVisible(not is_point)

    def _current_curve_layer(self):
        layer = self._current_layer("Curve")
        return layer if layer is not None else CurveLayer()

    def _on_curve_channel_changed(self, channel):
        self._curve_channel = channel
        points = self._current_curve_layer().points_by_channel[channel]
        self.curve_widget.set_points(channel, points)
        self.curve_widget.set_channel(channel)
        self.curve_reset_button.setEnabled(points != IDENTITY_POINTS)

    def _on_curve_points_changed(self, points):
        self._ensure_pending("Curve")
        base_layer = self._pending_old_layer.get("Curve") or CurveLayer()
        new_layer = base_layer.with_channel(self._curve_channel, points)

        self.document.layers = [l for l in self.document.layers if str(l) != "Curve"]
        self.document.layers.append(new_layer)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()
        self.curve_reset_button.setEnabled(list(points) != IDENTITY_POINTS)

    def _on_curve_editing_finished(self):
        if "Curve" not in self._pending_old_layer:
            return
        old_layer = self._pending_old_layer.pop("Curve")
        new_layer = self._current_layer("Curve")

        if old_layer is not None and new_layer is not None and \
                old_layer.points_by_channel == new_layer.points_by_channel:
            return  # gesture ended with no net change

        cmd = ChangeLayerCommand(self.document, "Curve", old_layer, new_layer)
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def _on_curve_reset_clicked(self):
        old_layer = self._current_layer("Curve")
        if old_layer is None:
            return

        reset_points = {ch: list(pts) for ch, pts in old_layer.points_by_channel.items()}
        reset_points[self._curve_channel] = list(IDENTITY_POINTS)

        if all(pts == IDENTITY_POINTS for pts in reset_points.values()):
            new_layer = None  # every channel is back to a straight line - drop the layer
        else:
            new_layer = CurveLayer(reset_points)

        cmd = ChangeLayerCommand(self.document, "Curve", old_layer, new_layer)
        self.document.execute_command(cmd)
        self.curve_widget.set_points(self._curve_channel, IDENTITY_POINTS)
        self.curve_reset_button.setEnabled(False)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def _sync_curve(self):
        layer = self._current_curve_layer()
        for channel, points in layer.points_by_channel.items():
            self.curve_widget.set_points(channel, points)
        self._pending_old_layer.pop("Curve", None)
        active_points = layer.points_by_channel[self._curve_channel]
        self.curve_reset_button.setEnabled(active_points != IDENTITY_POINTS)

    # --- parametric curve --------------------------------------------------

    def _current_parametric_layer(self):
        layer = self._current_layer(PARAMETRIC_LAYER_NAME)
        return layer if layer is not None else ParametricCurveLayer()

    def _parametric_with(self, base, field, value):
        return getattr(base, f"with_{field}")(value)

    def _on_parametric_drag_start(self, field):
        key = f"parametric_{field}"
        self._is_dragging[key] = True
        self._commit_timers[key].stop()
        self._ensure_pending(key, PARAMETRIC_LAYER_NAME)

    def _on_parametric_value_changed(self, field, raw_value):
        key = f"parametric_{field}"
        self._ensure_pending(key, PARAMETRIC_LAYER_NAME)

        base = self._current_parametric_layer()
        new_layer = self._parametric_with(base, field, raw_value)

        self.document.layers = [l for l in self.document.layers if str(l) != PARAMETRIC_LAYER_NAME]
        self.document.layers.append(new_layer)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()
        self._update_parametric_preview()

        if not self._is_dragging.get(key):
            # No slider-release to commit on (spinbox typing, keyboard
            # nudge) - commit after a short idle pause instead.
            self._commit_timers[key].start()

    def _on_parametric_drag_end(self, field):
        key = f"parametric_{field}"
        self._is_dragging[key] = False
        self._commit_timers[key].stop()
        self._on_parametric_commit(field)

    def _on_parametric_commit(self, field):
        key = f"parametric_{field}"
        if key not in self._pending_old_layer:
            return
        old_layer = self._pending_old_layer.pop(key)
        new_layer = self._current_layer(PARAMETRIC_LAYER_NAME)

        old_is_identity = old_layer is None or old_layer.is_identity()
        new_is_identity = new_layer is None or new_layer.is_identity()
        if old_is_identity and new_is_identity:
            return  # no net change over the whole gesture
        if old_layer is not None and new_layer is not None and vars(old_layer) == vars(new_layer):
            return
        if new_is_identity:
            new_layer = None

        cmd = ChangeLayerCommand(self.document, PARAMETRIC_LAYER_NAME, old_layer, new_layer)
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def _on_parametric_reset(self, field):
        old_layer = self._current_layer(PARAMETRIC_LAYER_NAME)
        if old_layer is None:
            return

        new_layer = self._parametric_with(old_layer, field, 0)
        if new_layer.is_identity():
            new_layer = None
        if old_layer is not None and new_layer is not None and vars(old_layer) == vars(new_layer):
            return

        cmd = ChangeLayerCommand(self.document, PARAMETRIC_LAYER_NAME, old_layer, new_layer)
        self.document.execute_command(cmd)
        self._sync_parametric_controls()
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def _update_parametric_preview(self):
        layer = self._current_parametric_layer()
        ramp = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(1, 256, 1)
        ramp_rgb = np.repeat(ramp, 3, axis=2)
        out = layer.apply(ramp_rgb)
        lut = np.clip(out[0, :, 0], 0.0, 1.0) * 255.0
        self.parametric_preview.set_preview_lut(lut)

    def _sync_parametric_controls(self):
        layer = self._current_parametric_layer()
        for field, (slider, spinbox, reset_btn) in self._parametric_rows.items():
            key = f"parametric_{field}"
            self._is_dragging[key] = False
            self._commit_timers[key].stop()
            self._pending_old_layer.pop(key, None)

            value = getattr(layer, field)
            slider.blockSignals(True)
            spinbox.blockSignals(True)
            slider.setValue(value)
            spinbox.setValue(value)
            slider.blockSignals(False)
            spinbox.blockSignals(False)
            reset_btn.setEnabled(value != 0)

        self._update_parametric_preview()

    def _wire_adjustments(self):
        # --- Undo-aware live adjustment wiring ---
        # A slider drag fires valueChanged once per pixel of movement. Each
        # attribute is tracked as a single layer in document.layers; while a
        # gesture (drag, spinbox edit, keyboard nudge) is in progress we only
        # update that layer for a live preview. One ChangeLayerCommand is
        # pushed to the undo history per gesture, not per tick, so a single
        # continuous drag from 0 to 3 undoes in one step back to 0.
        for name, slider, spinbox, reset_btn, layer_cls, transform, inverse, default_raw in self._adjustments:
            self._is_dragging[name] = False

            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(COMMIT_IDLE_MS)
            timer.timeout.connect(
                lambda name=name, layer_cls=layer_cls, transform=transform, slider=slider:
                    self._commit_change(name, layer_cls, transform, slider)
            )
            self._commit_timers[name] = timer

            slider.valueChanged.connect(
                lambda raw, name=name, layer_cls=layer_cls, transform=transform:
                    self._on_value_changed(name, layer_cls, transform, raw)
            )
            slider.sliderPressed.connect(lambda name=name: self._on_drag_start(name))
            slider.sliderReleased.connect(
                lambda name=name, layer_cls=layer_cls, transform=transform, slider=slider:
                    self._on_drag_end(name, layer_cls, transform, slider)
            )
            spinbox.editingFinished.connect(
                lambda name=name, layer_cls=layer_cls, transform=transform, slider=slider:
                    self._commit_change(name, layer_cls, transform, slider)
            )

            # Double-click a slider (Lightroom-style) to reset that attribute.
            self._reset_targets[slider] = (name, layer_cls, transform, default_raw)
            slider.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonDblClick and obj in self._reset_targets:
            name, layer_cls, transform, default_raw = self._reset_targets[obj]
            self._reset_attribute(name, layer_cls, transform, default_raw)
            return True
        return super().eventFilter(obj, event)

    def _current_layer(self, name):
        return next((l for l in self.document.layers if str(l) == name), None)

    def _ensure_pending(self, key, layer_name=None):
        if key not in self._pending_old_layer:
            self._pending_old_layer[key] = self._current_layer(layer_name or key)

    def _on_drag_start(self, name):
        self._is_dragging[name] = True
        self._commit_timers[name].stop()
        self._ensure_pending(name)

    def _on_drag_end(self, name, layer_cls, transform, slider):
        self._is_dragging[name] = False
        self._commit_timers[name].stop()
        self._commit_change(name, layer_cls, transform, slider)

    def _on_value_changed(self, name, layer_cls, transform, raw_value):
        self._ensure_pending(name)

        # Live preview: swap in the new value without touching undo history.
        layer = layer_cls(transform(raw_value))
        self.document.layers = [l for l in self.document.layers if str(l) != name]
        self.document.layers.append(layer)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

        if not self._is_dragging.get(name):
            # No slider-release to commit on (spinbox typing, keyboard nudge,
            # scroll wheel) - commit after a short idle pause instead.
            self._commit_timers[name].start()

    def _commit_change(self, name, layer_cls, transform, slider):
        if name not in self._pending_old_layer:
            return
        old_layer = self._pending_old_layer.pop(name)
        new_layer = layer_cls(transform(slider.value()))

        if old_layer is not None and vars(old_layer) == vars(new_layer):
            return  # value ended up unchanged over the whole gesture

        cmd = ChangeLayerCommand(self.document, name, old_layer, new_layer)
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def _reset_attribute(self, name, layer_cls, transform, default_raw):
        old_layer = self._current_layer(name)
        if old_layer is None:
            return  # already at default, nothing to do
        cmd = ChangeLayerCommand(self.document, name, old_layer, None)
        self.document.execute_command(cmd)
        self._sync_controls()
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def sync_from_document(self):
        """Public entry point to reflect the document's current layer values
        in every slider/spinbox - e.g. after loading a saved project whose
        layers were populated before this panel was wired up."""
        self._sync_controls()

    def _sync_controls(self):
        """Reflect the document's actual current layer values in every slider
        and spinbox, without treating the programmatic change as a new user
        gesture (which would otherwise create bogus undo entries)."""
        for name, slider, spinbox, reset_btn, layer_cls, transform, inverse, default_raw in self._adjustments:
            self._is_dragging[name] = False
            self._commit_timers[name].stop()
            self._pending_old_layer.pop(name, None)

            layer = self._current_layer(name)
            raw = inverse(next(iter(vars(layer).values()))) if layer is not None else default_raw

            slider.blockSignals(True)
            spinbox.blockSignals(True)
            slider.setValue(raw)
            spinbox.setValue(raw)
            slider.blockSignals(False)
            spinbox.blockSignals(False)
            reset_btn.setEnabled(raw != default_raw)

        self._sync_curve()
        self._sync_parametric_controls()

    def delete_layer(self):
        pass

    def undo(self):
        self.document.undo()
        self._sync_controls()
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def redo(self):
        self.document.redo()
        self._sync_controls()
        self.viewer.update_view()
        self.layer_stack_panel.refresh()
