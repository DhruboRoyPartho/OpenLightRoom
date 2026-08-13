from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QDoubleSpinBox
from PySide6.QtCore import Qt
from core.adjustment_layers.geometry_layer import GeometryLayer
from core.commands.change_layer_command import ChangeLayerCommand
from interface.gui.angle_ruler import AngleRuler, MIN_ANGLE, MAX_ANGLE
from interface.gui.theme import BG_PANEL, BORDER, BORDER_LIGHT, TEXT, TEXT_DIM, ACCENT

# (display label, longer-side pixel cap) - None means full resolution.
# Only affects the interactive on-screen preview (see
# core/processing/geometry.py:downscale_to_max_dimension and
# ImageDocument.render's max_dimension) - export always renders at full
# resolution regardless of this setting.
PREVIEW_QUALITY_OPTIONS = [
    ("Full Quality", None),
    ("High (2048px)", 2048),
    ("Balanced (1280px)", 1280),
    ("Fast (800px)", 800),
]

# (display label, ratio) - ratio is width/height, None for free-form, or the
# sentinel "original" to mean "match the current frame's own aspect".
ASPECT_RATIOS = [
    ("Free", None),
    ("Original", "original"),
    ("1:1", 1.0),
    ("16:9", 16 / 9),
    ("9:16", 9 / 16),
    ("4:3", 4 / 3),
    ("3:4", 3 / 4),
    ("5:4", 5 / 4),
    ("3:2", 3 / 2),
]

TOOLBAR_STYLE = f"""
    CanvasToolbar {{
        background-color: {BG_PANEL};
        border-bottom: 1px solid {BORDER};
    }}
    QPushButton {{
        background-color: transparent;
        color: {TEXT};
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 4px 10px;
    }}
    QPushButton:hover {{
        background-color: #3a3a3a;
        border-color: {BORDER_LIGHT};
    }}
    QPushButton:checked {{
        background-color: {ACCENT};
        color: #ffffff;
    }}
    QLabel {{
        color: {TEXT_DIM};
    }}
    QDoubleSpinBox {{
        background-color: {BG_PANEL};
        color: {TEXT};
        border: 1px solid {BORDER_LIGHT};
        border-radius: 3px;
        padding: 1px 3px;
    }}
"""


class CanvasToolbar(QWidget):
    """Canvas-level tools that sit above the image: zoom, before/after,
    rotate/flip, and the interactive crop tool. Mirrors ControlsPanel in
    that it owns committing changes to document.layers/history - the
    ImageViewer itself only handles display and in-progress crop editing."""

    def __init__(self, document, viewer, layer_stack_panel):
        super().__init__()
        self.document = document
        self.viewer = viewer
        self.layer_stack_panel = layer_stack_panel
        self.setStyleSheet(TOOLBAR_STYLE)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(4)

        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setFixedWidth(28)
        self.zoom_out_btn.setToolTip("Zoom Out")
        self.zoom_out_btn.clicked.connect(self.viewer.zoom_out)
        row.addWidget(self.zoom_out_btn)

        self.zoom_label = QLabel("Fit")
        self.zoom_label.setFixedWidth(44)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        row.addWidget(self.zoom_label)

        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedWidth(28)
        self.zoom_in_btn.setToolTip("Zoom In")
        self.zoom_in_btn.clicked.connect(self.viewer.zoom_in)
        row.addWidget(self.zoom_in_btn)

        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setToolTip("Fit image to window")
        self.fit_btn.clicked.connect(self.viewer.set_fit)
        row.addWidget(self.fit_btn)

        self.actual_size_btn = QPushButton("100%")
        self.actual_size_btn.setToolTip("Actual size (1:1 pixels)")
        self.actual_size_btn.clicked.connect(self.viewer.set_actual_size)
        row.addWidget(self.actual_size_btn)

        row.addSpacing(12)

        self.preview_quality_combo = QComboBox()
        for label, _ in PREVIEW_QUALITY_OPTIONS:
            self.preview_quality_combo.addItem(label)
        self.preview_quality_combo.setToolTip(
            "Preview render resolution. Lower settings render faster while\n"
            "editing on large images - this never affects the quality of an\n"
            "exported/saved image, which always renders at full resolution.\n"
            "Switch back to Full Quality before inspecting fine detail at 100%."
        )
        self.preview_quality_combo.currentIndexChanged.connect(self._on_preview_quality_changed)
        row.addWidget(self.preview_quality_combo)

        row.addSpacing(12)

        self.rotate_left_btn = QPushButton("↶")
        self.rotate_left_btn.setToolTip("Rotate Left 90°")
        self.rotate_left_btn.clicked.connect(lambda: self._rotate(-1))
        row.addWidget(self.rotate_left_btn)

        self.rotate_right_btn = QPushButton("↷")
        self.rotate_right_btn.setToolTip("Rotate Right 90°")
        self.rotate_right_btn.clicked.connect(lambda: self._rotate(1))
        row.addWidget(self.rotate_right_btn)

        self.flip_h_btn = QPushButton("⇆")
        self.flip_h_btn.setToolTip("Flip Horizontal")
        self.flip_h_btn.clicked.connect(self._flip_h)
        row.addWidget(self.flip_h_btn)

        self.flip_v_btn = QPushButton("⇅")
        self.flip_v_btn.setToolTip("Flip Vertical")
        self.flip_v_btn.clicked.connect(self._flip_v)
        row.addWidget(self.flip_v_btn)

        row.addSpacing(12)

        self.crop_btn = QPushButton("Crop")
        self.crop_btn.setCheckable(True)
        self.crop_btn.setToolTip("Crop & Straighten")
        self.crop_btn.toggled.connect(self._on_crop_toggled)
        row.addWidget(self.crop_btn)

        row.addStretch(1)

        self.before_after_btn = QPushButton("After")
        self.before_after_btn.setCheckable(True)
        self.before_after_btn.setToolTip("Toggle Before / After (\\)")
        self.before_after_btn.toggled.connect(self._on_before_after_toggled)
        row.addWidget(self.before_after_btn)

        outer.addLayout(row)

        self.crop_row = QWidget()
        crop_outer = QVBoxLayout(self.crop_row)
        crop_outer.setContentsMargins(0, 4, 0, 0)
        crop_outer.setSpacing(4)

        straighten_row = QHBoxLayout()
        straighten_row.setSpacing(4)

        self.straighten_btn = QPushButton("Straighten")
        self.straighten_btn.setCheckable(True)
        self.straighten_btn.setToolTip("Drag a line along a horizon or edge to level it")
        self.straighten_btn.toggled.connect(self._on_straighten_toggled)
        straighten_row.addWidget(self.straighten_btn)

        self.angle_ruler = AngleRuler()
        self.angle_ruler.angleChanged.connect(self._on_ruler_angle_changed)
        straighten_row.addWidget(self.angle_ruler, 1)

        self.angle_spinbox = QDoubleSpinBox()
        self.angle_spinbox.setRange(MIN_ANGLE, MAX_ANGLE)
        self.angle_spinbox.setDecimals(1)
        self.angle_spinbox.setSingleStep(0.5)
        self.angle_spinbox.setSuffix("°")
        self.angle_spinbox.setFixedWidth(70)
        self.angle_spinbox.valueChanged.connect(self._on_spinbox_angle_changed)
        straighten_row.addWidget(self.angle_spinbox)

        self.angle_reset_btn = QPushButton("0°")
        self.angle_reset_btn.setToolTip("Reset angle to 0°")
        self.angle_reset_btn.setFixedWidth(32)
        self.angle_reset_btn.clicked.connect(lambda: self._set_angle(0.0))
        straighten_row.addWidget(self.angle_reset_btn)

        crop_outer.addLayout(straighten_row)

        crop_row_layout = QHBoxLayout()
        crop_row_layout.setSpacing(4)

        crop_row_layout.addWidget(QLabel("Aspect"))
        self.aspect_combo = QComboBox()
        for name, _ in ASPECT_RATIOS:
            self.aspect_combo.addItem(name)
        self.aspect_combo.currentIndexChanged.connect(self._on_aspect_changed)
        crop_row_layout.addWidget(self.aspect_combo)

        crop_row_layout.addStretch(1)

        self.crop_reset_btn = QPushButton("Reset")
        self.crop_reset_btn.clicked.connect(self._reset_crop)
        crop_row_layout.addWidget(self.crop_reset_btn)

        self.crop_cancel_btn = QPushButton("Cancel")
        self.crop_cancel_btn.clicked.connect(self._cancel_crop)
        crop_row_layout.addWidget(self.crop_cancel_btn)

        self.crop_apply_btn = QPushButton("Apply Crop")
        self.crop_apply_btn.clicked.connect(self._apply_crop)
        crop_row_layout.addWidget(self.crop_apply_btn)

        crop_outer.addLayout(crop_row_layout)

        outer.addWidget(self.crop_row)
        self.crop_row.setVisible(False)

        self.viewer.zoomChanged.connect(self._on_zoom_changed)
        self.viewer.angleChanged.connect(self._on_viewer_angle_changed)
        self.viewer.cropModeChanged.connect(self._on_viewer_crop_mode_changed)

    # --- geometry helpers -------------------------------------------------

    def _current_geometry_layer(self):
        return next((l for l in self.document.layers if str(l) == "Crop"), None)

    def _current_geometry_or_default(self):
        layer = self._current_geometry_layer()
        return layer if layer is not None else GeometryLayer()

    def _commit_geometry(self, old_layer, new_layer):
        if new_layer.is_identity():
            new_layer = None
        cmd = ChangeLayerCommand(self.document, "Crop", old_layer, new_layer)
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def _rotate(self, delta_quarters):
        old_layer = self._current_geometry_layer()
        new_layer = self._current_geometry_or_default().with_rotation(delta_quarters)
        self._commit_geometry(old_layer, new_layer)

    def _flip_h(self):
        old_layer = self._current_geometry_layer()
        new_layer = self._current_geometry_or_default().with_flip_h()
        self._commit_geometry(old_layer, new_layer)

    def _flip_v(self):
        old_layer = self._current_geometry_layer()
        new_layer = self._current_geometry_or_default().with_flip_v()
        self._commit_geometry(old_layer, new_layer)

    # --- before/after -------------------------------------------------------

    def _on_before_after_toggled(self, checked):
        self.before_after_btn.setText("Before" if checked else "After")
        self.viewer.set_show_before(checked)

    def toggle_before_after(self):
        self.before_after_btn.setChecked(not self.before_after_btn.isChecked())

    # --- zoom --------------------------------------------------------------

    def _on_zoom_changed(self, zoom, is_fit):
        self.zoom_label.setText("Fit" if is_fit else f"{round(zoom * 100)}%")

    # --- preview quality --------------------------------------------------

    def _on_preview_quality_changed(self, index):
        _, max_dimension = PREVIEW_QUALITY_OPTIONS[index]
        self.viewer.set_preview_quality(max_dimension)

    # --- crop mode ----------------------------------------------------------

    def _on_crop_toggled(self, checked):
        self.crop_row.setVisible(checked)
        if checked:
            current = self._current_geometry_or_default()
            self._set_angle_display(current.angle)
            self.straighten_btn.setChecked(False)
            self.viewer.enter_crop_mode(
                current.crop_rect, current.rotation90, current.flip_h, current.flip_v, current.angle,
            )
            self.viewer.set_crop_aspect(self._selected_ratio())
        else:
            self.straighten_btn.setChecked(False)
            self.viewer.exit_crop_mode()

    def _on_viewer_crop_mode_changed(self, active):
        # Crop mode can now be force-exited by something other than this
        # button (e.g. selecting a mask, which makes it interactive on
        # the canvas) - keep the button/row in sync regardless of who
        # triggered the change. Guarded so a normal click on crop_btn
        # (which already drives enter/exit_crop_mode itself) doesn't
        # double-process.
        if self.crop_btn.isChecked() == active:
            return
        self.crop_btn.blockSignals(True)
        self.crop_btn.setChecked(active)
        self.crop_btn.blockSignals(False)
        self.crop_row.setVisible(active)
        if not active:
            self.straighten_btn.setChecked(False)

    def _selected_ratio(self):
        _, ratio = ASPECT_RATIOS[self.aspect_combo.currentIndex()]
        if ratio == "original":
            size = self.viewer.current_pixmap_size()
            return (size[0] / size[1]) if size else None
        return ratio

    def _on_aspect_changed(self, index):
        if self.crop_btn.isChecked():
            self.viewer.set_crop_aspect(self._selected_ratio())

    def _reset_crop(self):
        self.aspect_combo.setCurrentIndex(0)
        self._set_angle(0.0)
        self.viewer.reset_crop_rect()

    def _cancel_crop(self):
        self.crop_btn.setChecked(False)  # -> _on_crop_toggled(False) -> exit_crop_mode

    def _apply_crop(self):
        old_layer = self._current_geometry_layer()
        new_rect = self.viewer.get_crop_rect()
        new_angle = self.viewer.get_pending_angle()
        new_layer = self._current_geometry_or_default().with_angle_and_crop(new_angle, new_rect)
        self._commit_geometry(old_layer, new_layer)
        self.crop_btn.setChecked(False)

    # --- straighten -----------------------------------------------------

    def _on_straighten_toggled(self, checked):
        self.viewer.set_straighten_mode(checked)

    def _set_angle_display(self, angle):
        """Update the ruler/spinbox display only, without pushing back to
        the viewer (used when initializing from the currently committed
        geometry, which the viewer already knows about)."""
        self.angle_ruler.set_value(angle, emit=False)
        self.angle_spinbox.blockSignals(True)
        self.angle_spinbox.setValue(angle)
        self.angle_spinbox.blockSignals(False)

    def _set_angle(self, angle):
        self._set_angle_display(angle)
        self.viewer.set_pending_angle(angle)

    def _on_ruler_angle_changed(self, value):
        self.angle_spinbox.blockSignals(True)
        self.angle_spinbox.setValue(value)
        self.angle_spinbox.blockSignals(False)
        self.viewer.set_pending_angle(value)

    def _on_spinbox_angle_changed(self, value):
        self.angle_ruler.set_value(value, emit=False)
        self.viewer.set_pending_angle(value)

    def _on_viewer_angle_changed(self, value):
        # Reflects angle changes made by dragging directly on the image
        # (the Straighten tool), which don't go through the ruler/spinbox.
        self._set_angle_display(value)
