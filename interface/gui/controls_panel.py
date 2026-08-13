import numpy as np
from PySide6.QtWidgets import (
    QWidget, QSpinBox, QComboBox, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QScrollArea,
)
from PySide6.QtCore import Qt, QTimer, QEvent
from interface.gui.common_controls import CenterFillSlider, SPINBOX_STYLE, RESET_BUTTON_STYLE, build_slider_row
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
from core.adjustment_layers.hsl_layer import HSLLayer
from core.processing.hsl_grading import HSL_CHANNELS
from core.adjustment_layers.color_wheels_layer import ColorWheelsLayer
from core.processing.curve import IDENTITY_POINTS
from core.processing.white_balance import (
    estimate_gray_world_white_balance, estimate_white_balance_from_sample, sample_scene_linear_pixel,
)
from core.commands.change_layer_command import ChangeLayerCommand
from core.commands.composite_command import CompositeCommand
from interface.gui.curve_widget import CurveWidget
from interface.gui.color_wheel_widget import ColorWheelWidget
from interface.gui.presets_panel import PresetsPanel
from interface.gui.scopes_panel import ScopesPanel
from interface.gui.masks_panel import MasksPanel
from interface.gui.theme import BG_PANEL, BORDER, TEXT, TEXT_HEADER

PARAMETRIC_LAYER_NAME = "Parametric Curve"
PARAMETRIC_FIELDS = [
    ("Highlights", "highlights"),
    ("Lights", "lights"),
    ("Darks", "darks"),
    ("Shadows", "shadows"),
]

HSL_LAYER_NAME = "HSL"
HSL_AXES = [("Hue", "hue"), ("Saturation", "saturation"), ("Luminance", "luminance")]

COLOR_WHEELS_LAYER_NAME = "Color Wheels"
COLOR_WHEEL_ZONES = ["Shadows", "Midtones", "Highlights", "Global"]
_WHEEL_ZONE_KEY = {"Shadows": "shadows", "Midtones": "midtones", "Highlights": "highlights", "Global": "global"}
_WHEEL_ZONE_ATTR = {"shadows": "shadows", "midtones": "midtones", "highlights": "highlights", "global": "global_"}

# How long (ms) to wait after the last value change before committing an undo
# step, when there's no explicit slider-release to commit on (e.g. spinbox
# typing or keyboard nudging). A mouse drag on the slider instead commits
# immediately on release, so it isn't affected by this delay.
COMMIT_IDLE_MS = 500


SECTION_TOGGLE_STYLE = f"""
    QPushButton {{
        font-weight: 600;
        font-size: 11px;
        color: {TEXT_HEADER};
        background: transparent;
        border: none;
        border-radius: 0px;
        border-bottom: 1px solid {BORDER};
        padding: 9px 2px 7px 2px;
        margin-top: 6px;
        text-align: left;
    }}
    QPushButton:hover {{
        color: {TEXT};
    }}
"""

class CollapsibleSection(QWidget):
    """An accordion-style section: a clickable header (title + a v/> arrow
    prefix) that shows or hides its content area. The develop panel packs
    a lot of tools (Scopes, Presets, Tone, Tone Curve, Color, HSL, Color
    Wheels) into a limited-width sidebar - collapsing the sections you
    aren't using is what keeps that many tools reachable without an
    endless scroll, on top of the panel itself scrolling (see
    ControlsPanel's QScrollArea) for whatever's still expanded.

    Callers build a section's controls into body() exactly like they
    would the top-level panel layout - every _add_row()/_add_*_section()
    helper below just takes whatever QVBoxLayout it's handed.
    """

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self._title = title.upper()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.toggle_button = QPushButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setCursor(Qt.PointingHandCursor)
        self.toggle_button.setStyleSheet(SECTION_TOGGLE_STYLE)
        outer.addWidget(self.toggle_button)

        self.content = QWidget()
        self._content_layout = QVBoxLayout(self.content)
        self._content_layout.setContentsMargins(2, 6, 2, 8)
        self._content_layout.setSpacing(2)
        self.content.setVisible(expanded)
        outer.addWidget(self.content)

        self.toggle_button.toggled.connect(self._on_toggled)
        self._update_button_text(expanded)

    def _on_toggled(self, checked: bool):
        self.content.setVisible(checked)
        self._update_button_text(checked)

    def _update_button_text(self, expanded: bool):
        arrow = "▾" if expanded else "▸"  # ▾ expanded / ▸ collapsed
        self.toggle_button.setText(f"{arrow}  {self._title}")

    def body(self) -> QVBoxLayout:
        """The QVBoxLayout to build this section's controls into."""
        return self._content_layout


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

        # The panel packs 7 tool sections into a fixed-width sidebar - far
        # more vertical content than fits in the available height at once
        # (that mismatch, with no way to reach the overflow, is what made
        # the sidebar look "overlapped/destroyed"). Two fixes, together:
        # the whole panel scrolls (QScrollArea below), and each section is
        # a collapsible accordion (CollapsibleSection) so only the tools
        # actually in use need to take up space.
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet(f"QScrollArea {{ background-color: {BG_PANEL}; border: none; }}")
        outer_layout.addWidget(scroll_area, 1)

        scroll_content = QWidget()
        scroll_content.setStyleSheet(f"background-color: {BG_PANEL};")
        scroll_area.setWidget(scroll_content)

        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(2)

        scopes_section = CollapsibleSection("Scopes", expanded=True)
        layout.addWidget(scopes_section)
        self.scopes_panel = ScopesPanel()
        scopes_section.body().addWidget(self.scopes_panel)
        self.viewer.render_queue.image_rendered.connect(self.scopes_panel.set_image)
        self.scopes_panel.set_image(self.document.render())

        presets_section = CollapsibleSection("Presets", expanded=False)
        layout.addWidget(presets_section)
        self.presets_panel = PresetsPanel(self.document, self.viewer, self.layer_stack_panel,
                                           on_layers_changed=lambda: self._sync_controls())
        presets_section.body().addWidget(self.presets_panel)

        tone_section = CollapsibleSection("Tone", expanded=True)
        layout.addWidget(tone_section)
        tone_body = tone_section.body()
        self._add_row(tone_body, "Exposure", "exposure", -100, 100, 0,
                       ExposureLayer, lambda v: v, lambda x: round(x))
        self._add_row(tone_body, "Brightness", "brightness", -100, 100, 0,
                       BrightnessLayer, lambda v: v / 100.0, lambda x: round(x * 100))
        self._add_row(tone_body, "Contrast", "contrast", 10, 300, 100,
                       ContrastLayer, lambda v: v / 100.0, lambda x: round(x * 100))
        self._add_row(tone_body, "Highlights", "highlights", -100, 100, 0,
                       HighlightsLayer, lambda v: v, lambda x: round(x))
        self._add_row(tone_body, "Shadows", "shadows", -100, 100, 0,
                       ShadowsLayer, lambda v: v, lambda x: round(x))
        self._add_row(tone_body, "Whites", "whites", -100, 100, 0,
                       WhitesLayer, lambda v: v, lambda x: round(x))
        self._add_row(tone_body, "Blacks", "blacks", -100, 100, 0,
                       BlacksLayer, lambda v: v, lambda x: round(x))

        curve_section = CollapsibleSection("Tone Curve", expanded=False)
        layout.addWidget(curve_section)
        self._add_curve_section(curve_section.body())

        color_section = CollapsibleSection("Color", expanded=True)
        layout.addWidget(color_section)
        color_body = color_section.body()
        self._add_row(color_body, "Temperature", "temp", -100, 100, 0,
                       TemperatureLayer, lambda v: v, lambda x: round(x))
        self._add_row(color_body, "Tint", "tint", -100, 100, 0,
                       TintLayer, lambda v: v, lambda x: round(x))
        self._add_white_balance_tools_row(color_body)
        self._add_row(color_body, "Vibrance", "vibrance", -100, 100, 0,
                       VibranceLayer, lambda v: v, lambda x: round(x))
        self._add_row(color_body, "Saturation", "saturation", -100, 100, 0,
                       SaturationLayer, lambda v: v, lambda x: round(x))
        self._add_row(color_body, "Hue", "hue", -180, 180, 0,
                       HueLayer, lambda v: v, lambda x: round(x))

        hsl_section = CollapsibleSection("HSL", expanded=False)
        layout.addWidget(hsl_section)
        self._add_hsl_section(hsl_section.body())

        wheels_section = CollapsibleSection("Color Wheels", expanded=False)
        layout.addWidget(wheels_section)
        self._add_color_wheels_section(wheels_section.body())

        masks_section = CollapsibleSection("Masks", expanded=False)
        layout.addWidget(masks_section)
        self.masks_panel = MasksPanel(self.document, self.viewer, self.layer_stack_panel)
        masks_section.body().addWidget(self.masks_panel)

        layout.addStretch(1)

        # Undo/Redo live outside the scroll area, in a bar that's always
        # visible regardless of scroll position or which sections are
        # expanded - they're needed too often to risk being scrolled away.
        bottom_bar = QWidget()
        bottom_bar.setStyleSheet(f"background-color: {BG_PANEL}; border-top: 1px solid {BORDER};")
        bottom_layout = QVBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(12, 8, 12, 10)
        bottom_layout.setSpacing(0)

        button_row = QHBoxLayout()
        self.undo_button = QPushButton("Undo")
        self.redo_button = QPushButton("Redo")
        button_row.addWidget(self.undo_button)
        button_row.addWidget(self.redo_button)
        bottom_layout.addLayout(button_row)

        outer_layout.addWidget(bottom_bar)

        self.delete_layer_button = QPushButton("Delete")
        # not shown in the layout - kept as a wired-but-unused hook, same
        # as before this refactor

        self.delete_layer_button.clicked.connect(self.delete_layer)
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)

        self._wire_adjustments()

    def _build_slider_row(self, layout, display_label, minv, maxv, default):
        """Constructs a labeled slider + spinbox + reset button row and adds
        it to layout. Returns (slider, spinbox, reset_btn) - the caller
        wires up value-change/commit/reset behavior. Delegates to the
        shared builder (interface/gui/common_controls.py) so every numeric
        control in the app - including MasksPanel's - looks identical."""
        return build_slider_row(layout, display_label, minv, maxv, default,
                                 self.control_label_width, self.slider_width, self.spinbox_width)

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

    def _add_white_balance_tools_row(self, layout):
        row = QHBoxLayout()
        row.setSpacing(4)

        self.auto_wb_button = QPushButton("Auto")
        self.auto_wb_button.setToolTip("Estimate White Balance from the whole image (gray-world assumption)")
        row.addWidget(self.auto_wb_button)

        self.eyedropper_button = QPushButton("Eyedropper")
        self.eyedropper_button.setCheckable(True)
        self.eyedropper_button.setToolTip("Click a neutral gray or white area in the image to set White Balance")
        row.addWidget(self.eyedropper_button)

        row.addStretch(1)
        layout.addLayout(row)

        self.auto_wb_button.clicked.connect(self._on_auto_white_balance)
        self.eyedropper_button.toggled.connect(self._on_eyedropper_toggled)
        self.viewer.pixelPicked.connect(self._on_eyedropper_pixel_picked)
        self.viewer.eyedropperOwnerChanged.connect(self._on_eyedropper_owner_changed)

    def _on_auto_white_balance(self):
        crop_layer = self._current_layer("Crop")
        image = crop_layer.apply(self.document.base_image) if crop_layer is not None else self.document.base_image
        temp, tint = estimate_gray_world_white_balance(image)
        self._commit_white_balance(temp, tint)

    def _on_eyedropper_toggled(self, checked):
        self.viewer.set_eyedropper_mode(checked, purpose="white_balance")

    def _on_eyedropper_owner_changed(self, owner):
        # Another eyedropper (e.g. the Masks panel's Color Range picker)
        # just took over - drop this button's checked state so the UI
        # never shows two "armed" eyedroppers at once.
        if owner != "white_balance" and self.eyedropper_button.isChecked():
            self.eyedropper_button.blockSignals(True)
            self.eyedropper_button.setChecked(False)
            self.eyedropper_button.blockSignals(False)

    def _on_eyedropper_pixel_picked(self, px, py, purpose):
        if purpose != "white_balance":
            return
        r, g, b = sample_scene_linear_pixel(self.document, px, py)
        temp, tint = estimate_white_balance_from_sample(r, g, b)
        self._commit_white_balance(temp, tint)
        self.eyedropper_button.setChecked(False)  # one-shot, like Lightroom's WB eyedropper

    def _commit_white_balance(self, temp, tint):
        """Sets Temperature and Tint together as a single undo step (via
        CompositeCommand) - Auto WB / the eyedropper are one user action
        that happens to touch two layers, so one undo should restore both."""
        old_temp = self._current_layer("Temperature")
        old_tint = self._current_layer("Tint")
        new_temp = TemperatureLayer(temp) if abs(temp) > 1e-9 else None
        new_tint = TintLayer(tint) if abs(tint) > 1e-9 else None

        temp_unchanged = (old_temp is None and new_temp is None) or (
            old_temp is not None and new_temp is not None and vars(old_temp) == vars(new_temp))
        tint_unchanged = (old_tint is None and new_tint is None) or (
            old_tint is not None and new_tint is not None and vars(old_tint) == vars(new_tint))
        if temp_unchanged and tint_unchanged:
            return

        cmd = CompositeCommand([
            ChangeLayerCommand(self.document, "Temperature", old_temp, new_temp),
            ChangeLayerCommand(self.document, "Tint", old_tint, new_tint),
        ])
        self.document.execute_command(cmd)
        self._sync_controls()
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def _add_hsl_section(self, layout):
        self._hsl_channel = HSL_CHANNELS[0]

        header_row = QHBoxLayout()
        header_row.setSpacing(4)
        channel_label = QLabel("Channel")
        channel_label.setFixedWidth(self.control_label_width)
        header_row.addWidget(channel_label)

        self.hsl_channel_combo = QComboBox()
        self.hsl_channel_combo.addItems(HSL_CHANNELS)
        header_row.addWidget(self.hsl_channel_combo)
        header_row.addStretch(1)
        layout.addLayout(header_row)

        self.hsl_channel_combo.currentTextChanged.connect(self._on_hsl_channel_changed)

        self._hsl_rows = {}
        for label, axis in HSL_AXES:
            slider, spinbox, reset_btn = self._build_slider_row(layout, label, -100, 100, 0)
            self._hsl_rows[axis] = (slider, spinbox, reset_btn)

            key = f"hsl_{axis}"
            self._is_dragging[key] = False
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(COMMIT_IDLE_MS)
            timer.timeout.connect(lambda a=axis: self._on_hsl_commit(a))
            self._commit_timers[key] = timer

            slider.valueChanged.connect(lambda v, a=axis: self._on_hsl_value_changed(a, v))
            slider.sliderPressed.connect(lambda a=axis: self._on_hsl_drag_start(a))
            slider.sliderReleased.connect(lambda a=axis: self._on_hsl_drag_end(a))
            spinbox.editingFinished.connect(lambda a=axis: self._on_hsl_commit(a))
            reset_btn.clicked.connect(lambda checked=False, a=axis: self._on_hsl_reset(a))

    def _current_hsl_layer(self):
        layer = self._current_layer(HSL_LAYER_NAME)
        return layer if layer is not None else HSLLayer()

    def _hsl_layers_equal(self, a, b):
        return a.hue == b.hue and a.saturation == b.saturation and a.luminance == b.luminance

    def _on_hsl_channel_changed(self, channel):
        self._hsl_channel = channel
        self._sync_hsl_controls()

    def _on_hsl_drag_start(self, axis):
        key = f"hsl_{axis}"
        self._is_dragging[key] = True
        self._commit_timers[key].stop()
        self._ensure_pending(key, HSL_LAYER_NAME)

    def _on_hsl_value_changed(self, axis, raw_value):
        key = f"hsl_{axis}"
        self._ensure_pending(key, HSL_LAYER_NAME)

        base = self._current_hsl_layer()
        new_layer = base.with_value(axis, self._hsl_channel, raw_value)

        self.document.layers = [l for l in self.document.layers if str(l) != HSL_LAYER_NAME]
        if not new_layer.is_identity():
            self.document.layers.append(new_layer)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

        _, _, reset_btn = self._hsl_rows[axis]
        reset_btn.setEnabled(raw_value != 0)

        if not self._is_dragging.get(key):
            # No slider-release to commit on (spinbox typing, keyboard
            # nudge) - commit after a short idle pause instead.
            self._commit_timers[key].start()

    def _on_hsl_drag_end(self, axis):
        key = f"hsl_{axis}"
        self._is_dragging[key] = False
        self._commit_timers[key].stop()
        self._on_hsl_commit(axis)

    def _on_hsl_commit(self, axis):
        key = f"hsl_{axis}"
        if key not in self._pending_old_layer:
            return
        old_layer = self._pending_old_layer.pop(key)
        new_layer = self._current_layer(HSL_LAYER_NAME)

        old_is_identity = old_layer is None or old_layer.is_identity()
        new_is_identity = new_layer is None or new_layer.is_identity()
        if old_is_identity and new_is_identity:
            return  # no net change over the whole gesture
        if old_layer is not None and new_layer is not None and self._hsl_layers_equal(old_layer, new_layer):
            return
        if new_is_identity:
            new_layer = None

        cmd = ChangeLayerCommand(self.document, HSL_LAYER_NAME, old_layer, new_layer)
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def _on_hsl_reset(self, axis):
        old_layer = self._current_layer(HSL_LAYER_NAME)
        if old_layer is None:
            return

        new_layer = old_layer.with_value(axis, self._hsl_channel, 0)
        if new_layer.is_identity():
            new_layer = None
        if old_layer is not None and new_layer is not None and self._hsl_layers_equal(old_layer, new_layer):
            return

        cmd = ChangeLayerCommand(self.document, HSL_LAYER_NAME, old_layer, new_layer)
        self.document.execute_command(cmd)
        self._sync_hsl_controls()
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def _sync_hsl_controls(self):
        layer = self._current_hsl_layer()
        for axis, (slider, spinbox, reset_btn) in self._hsl_rows.items():
            key = f"hsl_{axis}"
            self._is_dragging[key] = False
            self._commit_timers[key].stop()
            self._pending_old_layer.pop(key, None)

            value = getattr(layer, axis).get(self._hsl_channel, 0)
            slider.blockSignals(True)
            spinbox.blockSignals(True)
            slider.setValue(value)
            spinbox.setValue(value)
            slider.blockSignals(False)
            spinbox.blockSignals(False)
            reset_btn.setEnabled(value != 0)

    def _add_color_wheels_section(self, layout):
        self._wheel_zone = COLOR_WHEEL_ZONES[0]

        header_row = QHBoxLayout()
        header_row.setSpacing(4)
        zone_label = QLabel("Zone")
        zone_label.setFixedWidth(self.control_label_width)
        header_row.addWidget(zone_label)

        self.wheel_zone_combo = QComboBox()
        self.wheel_zone_combo.addItems(COLOR_WHEEL_ZONES)
        header_row.addWidget(self.wheel_zone_combo)
        header_row.addStretch(1)

        self.wheel_reset_button = QPushButton("↺")
        self.wheel_reset_button.setFixedSize(20, 20)
        self.wheel_reset_button.setToolTip("Reset this zone's wheel to center")
        self.wheel_reset_button.setCursor(Qt.PointingHandCursor)
        self.wheel_reset_button.setStyleSheet(RESET_BUTTON_STYLE)
        self.wheel_reset_button.setEnabled(False)
        header_row.addWidget(self.wheel_reset_button)

        layout.addLayout(header_row)

        self.wheel_zone_combo.currentTextChanged.connect(self._on_wheel_zone_changed)
        self.wheel_reset_button.clicked.connect(self._on_wheel_reset)

        self.color_wheel_widget = ColorWheelWidget()
        self.color_wheel_widget.setFixedHeight(150)
        layout.addWidget(self.color_wheel_widget)

        self._is_dragging["ColorWheel"] = False
        self.color_wheel_widget.editingStarted.connect(lambda: self._ensure_pending("ColorWheel", COLOR_WHEELS_LAYER_NAME))
        self.color_wheel_widget.valueChanged.connect(self._on_wheel_value_changed)
        self.color_wheel_widget.editingFinished.connect(self._on_wheel_editing_finished)

        slider, spinbox, reset_btn = self._build_slider_row(layout, "Luminance", -100, 100, 0)
        self._wheel_luminance_row = (slider, spinbox, reset_btn)

        key = "wheel_luminance"
        self._is_dragging[key] = False
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(COMMIT_IDLE_MS)
        timer.timeout.connect(self._on_wheel_luminance_commit)
        self._commit_timers[key] = timer

        slider.valueChanged.connect(self._on_wheel_luminance_changed)
        slider.sliderPressed.connect(self._on_wheel_luminance_drag_start)
        slider.sliderReleased.connect(self._on_wheel_luminance_drag_end)
        spinbox.editingFinished.connect(self._on_wheel_luminance_commit)
        reset_btn.clicked.connect(self._on_wheel_luminance_reset)

    def _current_color_wheels_layer(self):
        layer = self._current_layer(COLOR_WHEELS_LAYER_NAME)
        return layer if layer is not None else ColorWheelsLayer()

    def _wheel_layers_equal(self, a, b):
        return (a.shadows == b.shadows and a.midtones == b.midtones
                and a.highlights == b.highlights and a.global_ == b.global_)

    def _current_wheel_zone_key(self):
        return _WHEEL_ZONE_KEY[self._wheel_zone]

    def _current_wheel_dict(self, layer):
        return getattr(layer, _WHEEL_ZONE_ATTR[self._current_wheel_zone_key()])

    def _on_wheel_zone_changed(self, zone):
        self._wheel_zone = zone
        self._sync_color_wheels_controls()

    def _on_wheel_value_changed(self, hue_deg, chroma):
        self._ensure_pending("ColorWheel", COLOR_WHEELS_LAYER_NAME)

        base = self._current_color_wheels_layer()
        zone = self._current_wheel_zone_key()
        new_layer = base.with_wheel(zone, hue_deg=hue_deg, chroma=chroma * 100.0)

        self.document.layers = [l for l in self.document.layers if str(l) != COLOR_WHEELS_LAYER_NAME]
        if not new_layer.is_identity():
            self.document.layers.append(new_layer)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

        self.wheel_reset_button.setEnabled(chroma != 0.0)

    def _on_wheel_editing_finished(self):
        key = "ColorWheel"
        if key not in self._pending_old_layer:
            return
        old_layer = self._pending_old_layer.pop(key)
        new_layer = self._current_layer(COLOR_WHEELS_LAYER_NAME)

        old_is_identity = old_layer is None or old_layer.is_identity()
        new_is_identity = new_layer is None or new_layer.is_identity()
        if old_is_identity and new_is_identity:
            return  # gesture ended with no net change
        if old_layer is not None and new_layer is not None and self._wheel_layers_equal(old_layer, new_layer):
            return
        if new_is_identity:
            new_layer = None

        cmd = ChangeLayerCommand(self.document, COLOR_WHEELS_LAYER_NAME, old_layer, new_layer)
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def _on_wheel_reset(self):
        old_layer = self._current_layer(COLOR_WHEELS_LAYER_NAME)
        if old_layer is None:
            return
        zone = self._current_wheel_zone_key()
        new_layer = old_layer.with_wheel(zone, hue_deg=0.0, chroma=0.0)
        if new_layer.is_identity():
            new_layer = None
        if old_layer is not None and new_layer is not None and self._wheel_layers_equal(old_layer, new_layer):
            return

        cmd = ChangeLayerCommand(self.document, COLOR_WHEELS_LAYER_NAME, old_layer, new_layer)
        self.document.execute_command(cmd)
        self._sync_color_wheels_controls()
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def _on_wheel_luminance_drag_start(self):
        key = "wheel_luminance"
        self._is_dragging[key] = True
        self._commit_timers[key].stop()
        self._ensure_pending(key, COLOR_WHEELS_LAYER_NAME)

    def _on_wheel_luminance_changed(self, raw_value):
        key = "wheel_luminance"
        self._ensure_pending(key, COLOR_WHEELS_LAYER_NAME)

        base = self._current_color_wheels_layer()
        zone = self._current_wheel_zone_key()
        new_layer = base.with_wheel(zone, luminance=float(raw_value))

        self.document.layers = [l for l in self.document.layers if str(l) != COLOR_WHEELS_LAYER_NAME]
        if not new_layer.is_identity():
            self.document.layers.append(new_layer)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

        _, _, reset_btn = self._wheel_luminance_row
        reset_btn.setEnabled(raw_value != 0)

        if not self._is_dragging.get(key):
            self._commit_timers[key].start()

    def _on_wheel_luminance_drag_end(self):
        key = "wheel_luminance"
        self._is_dragging[key] = False
        self._commit_timers[key].stop()
        self._on_wheel_luminance_commit()

    def _on_wheel_luminance_commit(self):
        key = "wheel_luminance"
        if key not in self._pending_old_layer:
            return
        old_layer = self._pending_old_layer.pop(key)
        new_layer = self._current_layer(COLOR_WHEELS_LAYER_NAME)

        old_is_identity = old_layer is None or old_layer.is_identity()
        new_is_identity = new_layer is None or new_layer.is_identity()
        if old_is_identity and new_is_identity:
            return
        if old_layer is not None and new_layer is not None and self._wheel_layers_equal(old_layer, new_layer):
            return
        if new_is_identity:
            new_layer = None

        cmd = ChangeLayerCommand(self.document, COLOR_WHEELS_LAYER_NAME, old_layer, new_layer)
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def _on_wheel_luminance_reset(self):
        old_layer = self._current_layer(COLOR_WHEELS_LAYER_NAME)
        if old_layer is None:
            return
        zone = self._current_wheel_zone_key()
        new_layer = old_layer.with_wheel(zone, luminance=0.0)
        if new_layer.is_identity():
            new_layer = None
        if old_layer is not None and new_layer is not None and self._wheel_layers_equal(old_layer, new_layer):
            return

        cmd = ChangeLayerCommand(self.document, COLOR_WHEELS_LAYER_NAME, old_layer, new_layer)
        self.document.execute_command(cmd)
        self._sync_color_wheels_controls()
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def _sync_color_wheels_controls(self):
        layer = self._current_color_wheels_layer()
        wheel = self._current_wheel_dict(layer)

        self._is_dragging["ColorWheel"] = False
        self._pending_old_layer.pop("ColorWheel", None)
        self._is_dragging["wheel_luminance"] = False
        self._commit_timers["wheel_luminance"].stop()
        self._pending_old_layer.pop("wheel_luminance", None)

        self.color_wheel_widget.set_value(wheel["hue_deg"], wheel["chroma"] / 100.0)
        self.wheel_reset_button.setEnabled(wheel["chroma"] != 0.0)

        slider, spinbox, reset_btn = self._wheel_luminance_row
        value = round(wheel["luminance"])
        slider.blockSignals(True)
        spinbox.blockSignals(True)
        slider.setValue(value)
        spinbox.setValue(value)
        slider.blockSignals(False)
        spinbox.blockSignals(False)
        reset_btn.setEnabled(value != 0)

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
        self._sync_hsl_controls()
        self._sync_color_wheels_controls()
        self.masks_panel.refresh()

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
