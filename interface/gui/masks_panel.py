from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QPushButton,
    QLabel, QMenu, QCheckBox, QInputDialog, QMessageBox, QAbstractItemView,
)
from PySide6.QtCore import Qt, QTimer

from core.adjustment_layers.masked_adjustment_layer import MaskedAdjustmentLayer, ADJUSTMENT_FIELDS
from core.masking.mask import Mask, MaskComponent
from core.commands.change_layer_command import ChangeLayerCommand
from interface.gui.common_controls import build_slider_row, RESET_BUTTON_STYLE
from interface.gui.theme import TEXT, TEXT_DIM, BORDER, BG_FIELD

COMMIT_IDLE_MS = 500

# (menu label, component kind)
MASK_TYPE_OPTIONS = [
    ("Brush", "brush"),
    ("Radial", "radial"),
    ("Linear Gradient", "linear_gradient"),
    ("Rectangle", "rectangle"),
    ("Ellipse", "ellipse"),
    ("Polygon", "polygon"),
    ("Color Range", "color_range"),
    ("Luminance Range", "luminance_range"),
    ("Subject", "subject"),
    ("Sky", "sky"),
    ("Skin", "skin"),
]
KIND_LABELS = {kind: label for label, kind in MASK_TYPE_OPTIONS}

DEFAULT_PARAMS_BY_KIND = {
    "brush": {"strokes": []},
    "radial": {"center_x": 0.5, "center_y": 0.5, "radius_x": 0.25, "radius_y": 0.25, "angle_deg": 0.0, "feather": 50.0},
    "linear_gradient": {"x0": 0.3, "y0": 0.5, "x1": 0.7, "y1": 0.5},
    "rectangle": {"center_x": 0.5, "center_y": 0.5, "half_width": 0.25, "half_height": 0.25, "angle_deg": 0.0, "feather": 0.0},
    "ellipse": {"center_x": 0.5, "center_y": 0.5, "radius_x": 0.25, "radius_y": 0.25, "angle_deg": 0.0, "feather": 0.0},
    "polygon": {"points": [], "feather": 0.0},
    "color_range": {"sample_rgb": (0.5, 0.5, 0.5), "refine": 50.0},
    "luminance_range": {"low": 0.0, "high": 1.0, "feather": 20.0},
    "subject": {},
    "sky": {},
    "skin": {"feather": 20.0},
}

# (param_key, label, ui_min, ui_max, ui_default, kind_hint)
# kind_hint "percent": stored value is a 0..1 fraction, shown as a 0..100
# integer slider; "degrees"/"pct100": stored value already matches the
# integer UI range 1:1.
PARAM_FIELDS_BY_KIND = {
    "radial": [
        ("center_x", "Center X", 0, 100, 50, "percent"),
        ("center_y", "Center Y", 0, 100, 50, "percent"),
        ("radius_x", "Radius X", 1, 100, 25, "percent"),
        ("radius_y", "Radius Y", 1, 100, 25, "percent"),
        ("angle_deg", "Angle", -180, 180, 0, "degrees"),
        ("feather", "Feather", 0, 100, 50, "pct100"),
    ],
    "ellipse": [
        ("center_x", "Center X", 0, 100, 50, "percent"),
        ("center_y", "Center Y", 0, 100, 50, "percent"),
        ("radius_x", "Radius X", 1, 100, 25, "percent"),
        ("radius_y", "Radius Y", 1, 100, 25, "percent"),
        ("angle_deg", "Angle", -180, 180, 0, "degrees"),
        ("feather", "Feather", 0, 100, 0, "pct100"),
    ],
    "rectangle": [
        ("center_x", "Center X", 0, 100, 50, "percent"),
        ("center_y", "Center Y", 0, 100, 50, "percent"),
        ("half_width", "Half Width", 1, 100, 25, "percent"),
        ("half_height", "Half Height", 1, 100, 25, "percent"),
        ("angle_deg", "Angle", -180, 180, 0, "degrees"),
        ("feather", "Feather", 0, 100, 0, "pct100"),
    ],
    "linear_gradient": [
        ("x0", "Start X", 0, 100, 30, "percent"),
        ("y0", "Start Y", 0, 100, 50, "percent"),
        ("x1", "End X", 0, 100, 70, "percent"),
        ("y1", "End Y", 0, 100, 50, "percent"),
    ],
    "polygon": [
        ("feather", "Feather", 0, 100, 0, "pct100"),
    ],
    "luminance_range": [
        ("low", "Low", 0, 100, 0, "percent"),
        ("high", "High", 0, 100, 100, "percent"),
        ("feather", "Feather", 0, 100, 20, "pct100"),
    ],
    "color_range": [
        ("refine", "Refine", 0, 100, 50, "pct100"),
    ],
    "skin": [
        ("feather", "Feather", 0, 100, 20, "pct100"),
    ],
    "brush": [],
    "subject": [],
    "sky": [],
}

INTERACTIVE_KINDS = ("radial", "ellipse", "rectangle", "linear_gradient", "brush", "polygon")

MASK_KIND_HELP = {
    "radial": "Drag directly on the image to move, resize (corner handles) and rotate the shape.",
    "ellipse": "Drag directly on the image to move, resize (corner handles) and rotate the shape.",
    "rectangle": "Drag directly on the image to move, resize (corner handles) and rotate the shape.",
    "linear_gradient": "Drag the two endpoints directly on the image to reposition the gradient.",
    "brush": "Paint directly on the image. Adjust size/hardness/flow below.",
    "polygon": "Click points on the image to trace the selection - click near the first point to close it.",
    "color_range": "Use the Pick Color eyedropper below to sample the reference color; Refine controls how tightly the selection matches it.",
    "luminance_range": "Adjust Low/High below to select a tonal range; the overlay updates live.",
    "subject": "Requires a subject-detection model, not included in this build - selects the whole image until a real SubjectMaskEngine is registered (see core/ai).",
    "sky": "Requires a sky-detection model, not included in this build - selects the whole image until a real SkyMaskEngine is registered (see core/ai).",
    "skin": "Detected automatically from skin-tone heuristics; Feather softens the edge.",
}


def _ui_to_value(kind_hint: str, ui_value: int) -> float:
    if kind_hint == "percent":
        return ui_value / 100.0
    return float(ui_value)


def _value_to_ui(kind_hint: str, value: float) -> int:
    if kind_hint == "percent":
        return round(value * 100)
    return round(value)


class MasksPanel(QWidget):
    """Full non-destructive layer management for local (masked)
    adjustments: an "Add Mask" picker for all 11 mask types, a list of
    existing "Mask N" layers (visibility toggle, rename, duplicate,
    reorder, delete), and - for whichever mask is selected - its stack of
    combined components (Add/Subtract/Intersect, per-component Invert),
    each component's own shape parameters, the mask-level operations
    (Invert/Feather/Blur/Density), and the local Basic/Color adjustment
    sliders (the same math as the equivalent global tools - see
    MaskedAdjustmentLayer).

    Mirrors the rest of this app's control panels: every edit is one
    ChangeLayerCommand, drag gestures on a slider batch into a single
    undo step (sliderPressed..sliderReleased), and non-drag edits
    (spinbox typing, checkboxes, combos) commit after a short idle pause
    or immediately for discrete actions.
    """

    def __init__(self, document, viewer, layer_stack_panel):
        super().__init__()
        self.document = document
        self.viewer = viewer
        self.layer_stack_panel = layer_stack_panel

        self.control_label_width = 80
        self.slider_width = 130
        self.spinbox_width = 50

        self._selected_mask_name = None
        self._selected_component_index = None
        self._pending_old_layer = None
        self._suppress_list_signal = False
        self._dragging = False
        self._dragging_overlay_active = False
        self.pick_color_button = None

        self._commit_timer = QTimer(self)
        self._commit_timer.setSingleShot(True)
        self._commit_timer.setInterval(COMMIT_IDLE_MS)
        self._commit_timer.timeout.connect(self._commit_gesture)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        # --- add-mask row ---------------------------------------------------
        add_row = QHBoxLayout()
        self.add_mask_button = QPushButton("+ Add Mask")
        self.add_mask_button.setMenu(self._build_type_menu(self._add_mask))
        add_row.addWidget(self.add_mask_button)
        add_row.addStretch(1)
        outer.addLayout(add_row)

        # --- mask list --------------------------------------------------------
        self.mask_list = QListWidget()
        self.mask_list.setFixedHeight(110)
        self.mask_list.setSelectionMode(QAbstractItemView.SingleSelection)
        outer.addWidget(self.mask_list)

        list_buttons = QHBoxLayout()
        list_buttons.setSpacing(4)
        self.move_up_button = QPushButton("▲")
        self.move_up_button.setToolTip("Move mask up (applies earlier)")
        self.move_up_button.setFixedWidth(28)
        self.move_down_button = QPushButton("▼")
        self.move_down_button.setToolTip("Move mask down (applies later)")
        self.move_down_button.setFixedWidth(28)
        self.rename_button = QPushButton("Rename")
        self.duplicate_button = QPushButton("Duplicate")
        self.delete_mask_button = QPushButton("Delete")
        for w in (self.move_up_button, self.move_down_button, self.rename_button,
                  self.duplicate_button, self.delete_mask_button):
            list_buttons.addWidget(w)
        outer.addLayout(list_buttons)

        # --- selected-mask editor ----------------------------------------
        self.editor_container = QWidget()
        editor_layout = QVBoxLayout(self.editor_container)
        editor_layout.setContentsMargins(0, 6, 0, 0)
        editor_layout.setSpacing(4)

        # Selecting a mask in the list above immediately makes it live on
        # the canvas - no separate arming step, like Lightroom. This row
        # controls the red "where does this apply" overlay and reports
        # what the canvas is currently doing.
        overlay_row = QHBoxLayout()
        self.show_overlay_checkbox = QCheckBox("Show Mask Overlay (red)")
        self.show_overlay_checkbox.setChecked(False)
        self.show_overlay_checkbox.setToolTip(
            "Tint the canvas red wherever the selected mask currently applies.\n"
            "Off by default so you always see the actual graded image - it also\n"
            "flashes on automatically while you're dragging a shape or painting,\n"
            "so you can see exactly where the boundary falls.")
        overlay_row.addWidget(self.show_overlay_checkbox)
        overlay_row.addStretch(1)
        editor_layout.addLayout(overlay_row)

        self.canvas_status_label = QLabel("")
        self.canvas_status_label.setWordWrap(True)
        self.canvas_status_label.setStyleSheet(
            f"color: {TEXT}; font-size: 10px; font-weight: 600; padding: 2px 0px;")
        editor_layout.addWidget(self.canvas_status_label)

        editor_layout.addWidget(self._divider_label("Components"))
        component_add_row = QHBoxLayout()
        self.add_component_button = QPushButton("+ Add Component")
        self.add_component_button.setMenu(self._build_type_menu(self._add_component))
        component_add_row.addWidget(self.add_component_button)
        component_add_row.addStretch(1)
        editor_layout.addLayout(component_add_row)

        self.component_list = QListWidget()
        self.component_list.setFixedHeight(80)
        self.component_list.setSelectionMode(QAbstractItemView.SingleSelection)
        editor_layout.addWidget(self.component_list)

        component_op_row = QHBoxLayout()
        component_op_row.addWidget(QLabel("Mode"))
        from PySide6.QtWidgets import QComboBox
        self.component_op_combo = QComboBox()
        self.component_op_combo.addItems(["Add", "Subtract", "Intersect"])
        component_op_row.addWidget(self.component_op_combo)
        self.component_invert_check = QCheckBox("Invert this component")
        component_op_row.addWidget(self.component_invert_check)
        component_op_row.addStretch(1)
        self.delete_component_button = QPushButton("Delete")
        component_op_row.addWidget(self.delete_component_button)
        editor_layout.addLayout(component_op_row)

        self.component_help_label = QLabel("")
        self.component_help_label.setWordWrap(True)
        self.component_help_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px; font-style: italic;")
        editor_layout.addWidget(self.component_help_label)

        self.brush_settings_container = QWidget()
        brush_settings_layout = QVBoxLayout(self.brush_settings_container)
        brush_settings_layout.setContentsMargins(0, 4, 0, 4)
        brush_settings_layout.setSpacing(2)
        self.brush_size_slider, self.brush_size_spinbox, _r1 = build_slider_row(
            brush_settings_layout, "Brush Size", 1, 50, 4, self.control_label_width, self.slider_width, self.spinbox_width)
        self.brush_hardness_slider, self.brush_hardness_spinbox, _r2 = build_slider_row(
            brush_settings_layout, "Hardness", 0, 100, 80, self.control_label_width, self.slider_width, self.spinbox_width)
        self.brush_flow_slider, self.brush_flow_spinbox, _r3 = build_slider_row(
            brush_settings_layout, "Flow", 1, 100, 100, self.control_label_width, self.slider_width, self.spinbox_width)
        brush_mode_row = QHBoxLayout()
        brush_mode_row.addWidget(QLabel("Brush Mode"))
        from PySide6.QtWidgets import QComboBox as _QComboBox
        self.brush_mode_combo = _QComboBox()
        self.brush_mode_combo.addItems(["Add", "Subtract (erase)"])
        brush_mode_row.addWidget(self.brush_mode_combo)
        brush_mode_row.addStretch(1)
        brush_settings_layout.addLayout(brush_mode_row)
        editor_layout.addWidget(self.brush_settings_container)
        self.brush_settings_container.setVisible(False)

        self.param_fields_container = QWidget()
        self.param_fields_layout = QVBoxLayout(self.param_fields_container)
        self.param_fields_layout.setContentsMargins(0, 4, 0, 4)
        self.param_fields_layout.setSpacing(2)
        editor_layout.addWidget(self.param_fields_container)

        editor_layout.addWidget(self._divider_label("Mask"))
        self.invert_check = QCheckBox("Invert whole mask")
        editor_layout.addWidget(self.invert_check)
        self.feather_slider, self.feather_spinbox, self.feather_reset = build_slider_row(
            editor_layout, "Feather", 0, 100, 0, self.control_label_width, self.slider_width, self.spinbox_width)
        self.blur_slider, self.blur_spinbox, self.blur_reset = build_slider_row(
            editor_layout, "Blur", 0, 100, 0, self.control_label_width, self.slider_width, self.spinbox_width)
        self.density_slider, self.density_spinbox, self.density_reset = build_slider_row(
            editor_layout, "Density", 0, 100, 100, self.control_label_width, self.slider_width, self.spinbox_width)

        editor_layout.addWidget(self._divider_label("Local Adjustments"))
        self._adjustment_rows = {}
        adjustment_specs = [
            ("exposure", "Exposure", -100, 100, 0),
            ("contrast", "Contrast", -100, 100, 0),
            ("highlights", "Highlights", -100, 100, 0),
            ("shadows", "Shadows", -100, 100, 0),
            ("whites", "Whites", -100, 100, 0),
            ("blacks", "Blacks", -100, 100, 0),
            ("temperature", "Temperature", -100, 100, 0),
            ("tint", "Tint", -100, 100, 0),
            ("saturation", "Saturation", -100, 100, 0),
            ("hue", "Hue", -180, 180, 0),
        ]
        for field, label, minv, maxv, default in adjustment_specs:
            slider, spinbox, reset_btn = build_slider_row(
                editor_layout, label, minv, maxv, default,
                self.control_label_width, self.slider_width, self.spinbox_width)
            self._adjustment_rows[field] = (slider, spinbox, reset_btn)

        outer.addWidget(self.editor_container)
        self.editor_container.setVisible(False)

        self.empty_hint = QLabel("No masks yet - click “+ Add Mask” to create one.")
        self.empty_hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        self.empty_hint.setWordWrap(True)
        outer.addWidget(self.empty_hint)

        self._wire_events()
        self.refresh()

    # --- construction helpers ---------------------------------------------

    def _divider_label(self, text):
        label = QLabel(text.upper())
        label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px; font-weight: 600; "
                             f"border-bottom: 1px solid {BORDER}; padding-top: 4px; padding-bottom: 2px;")
        return label

    def _build_type_menu(self, callback):
        menu = QMenu(self)
        for label, kind in MASK_TYPE_OPTIONS:
            action = menu.addAction(label)
            action.triggered.connect(lambda checked=False, k=kind: callback(k))
        return menu

    def _wire_events(self):
        self.mask_list.currentRowChanged.connect(self._on_mask_row_changed)
        self.mask_list.itemChanged.connect(self._on_mask_item_changed)
        self.move_up_button.clicked.connect(lambda: self._move_selected_mask(-1))
        self.move_down_button.clicked.connect(lambda: self._move_selected_mask(1))
        self.rename_button.clicked.connect(self._rename_selected_mask)
        self.duplicate_button.clicked.connect(self._duplicate_selected_mask)
        self.delete_mask_button.clicked.connect(self._delete_selected_mask)

        self.component_list.currentRowChanged.connect(self._on_component_row_changed)
        self.component_op_combo.currentIndexChanged.connect(self._on_component_op_changed)
        self.component_invert_check.toggled.connect(self._on_component_invert_toggled)
        self.delete_component_button.clicked.connect(self._delete_selected_component)

        self.show_overlay_checkbox.toggled.connect(lambda checked: self._sync_canvas_state())
        self.viewer.cropModeChanged.connect(self._on_viewer_crop_mode_changed)
        self.brush_size_slider.valueChanged.connect(
            lambda v: self.viewer.set_brush_settings(radius=v / 100.0))
        self.brush_hardness_slider.valueChanged.connect(
            lambda v: self.viewer.set_brush_settings(hardness=float(v)))
        self.brush_flow_slider.valueChanged.connect(
            lambda v: self.viewer.set_brush_settings(flow=float(v)))
        self.brush_mode_combo.currentIndexChanged.connect(
            lambda i: self.viewer.set_brush_settings(mode="subtract" if i == 1 else "add"))

        self.viewer.maskDragStarted.connect(self._on_canvas_drag_started)
        self.viewer.maskGeometryChanged.connect(self._on_canvas_geometry_changed)
        self.viewer.maskDragFinished.connect(self._on_canvas_drag_finished)
        self.viewer.maskBrushStrokeFinished.connect(self._on_canvas_brush_stroke)
        self.viewer.maskPolygonFinished.connect(self._on_canvas_polygon_finished)

        self.viewer.pixelPicked.connect(self._on_color_range_pixel_picked)
        self.viewer.eyedropperOwnerChanged.connect(self._on_eyedropper_owner_changed)

        self.invert_check.toggled.connect(lambda checked: self._commit_discrete_mask_field("invert", checked))
        self._bind_slider(self.feather_slider, self.feather_spinbox, self.feather_reset,
                           lambda layer, v: self._with_mask_field(layer, "feather", float(v)), 0)
        self._bind_slider(self.blur_slider, self.blur_spinbox, self.blur_reset,
                           lambda layer, v: self._with_mask_field(layer, "blur", float(v)), 0)
        self._bind_slider(self.density_slider, self.density_spinbox, self.density_reset,
                           lambda layer, v: self._with_mask_field(layer, "density", float(v)), 100)

        for field, (slider, spinbox, reset_btn) in self._adjustment_rows.items():
            default = 0
            self._bind_slider(slider, spinbox, reset_btn,
                               (lambda layer, v, f=field: layer.with_adjustment(f, float(v))), default)

    # --- generic drag-gesture-batched slider binding ------------------------

    def _bind_slider(self, slider, spinbox, reset_btn, build_new_layer, default):
        def on_press():
            self._dragging = True
            self._begin_gesture()

        def on_release():
            self._dragging = False
            self._commit_gesture()

        slider.sliderPressed.connect(on_press)
        slider.sliderReleased.connect(on_release)
        slider.valueChanged.connect(lambda v, b=build_new_layer: self._on_slider_value_changed(b, v))
        spinbox.editingFinished.connect(self._commit_gesture)
        reset_btn.clicked.connect(lambda checked=False, b=build_new_layer, d=default: self._on_reset_clicked(b, d))

    def _on_slider_value_changed(self, build_new_layer, raw_value):
        self._begin_gesture()
        layer = self._current_mask_layer()
        if layer is None:
            return
        new_layer = build_new_layer(layer, raw_value)
        self._stage_layer_change(new_layer)
        # A single shared drag flag (only one slider can be dragged by one
        # mouse at a time) rather than tracking every individual slider -
        # correctly covers the shape-parameter sliders too, which are
        # rebuilt fresh each time the selected component changes.
        if not self._dragging:
            self._commit_timer.start()

    def _on_reset_clicked(self, build_new_layer, default):
        self._begin_gesture()
        layer = self._current_mask_layer()
        if layer is None:
            return
        new_layer = build_new_layer(layer, default)
        self._stage_layer_change(new_layer)
        self._commit_gesture()
        self._sync_editor()

    def _begin_gesture(self):
        if self._pending_old_layer is None:
            self._pending_old_layer = self._current_mask_layer()
        self._commit_timer.stop()

    def _stage_layer_change(self, new_layer):
        name = str(new_layer)
        self.document.layers = [l for l in self.document.layers if str(l) != name]
        self.document.layers.append(new_layer)
        self.viewer.update_view()

    def _commit_gesture(self):
        self._commit_timer.stop()
        if self._pending_old_layer is None:
            return
        old_layer = self._pending_old_layer
        self._pending_old_layer = None
        new_layer = self._current_mask_layer()

        if old_layer is None and new_layer is None:
            return
        # Deliberately NOT gated on is_identity() (unlike ControlsPanel's
        # simple single-value tools): a mask's geometry/settings (invert,
        # feather, a shape's placement) are meaningful, undo-worthy state
        # on their own even before any local adjustment is dialed in -
        # the common workflow is "place and shape the mask, then adjust
        # it" - so only an exact no-op (dragged back to the same value)
        # should be skipped.
        if old_layer is not None and new_layer is not None and self._layers_equal(old_layer, new_layer):
            return

        name = str(new_layer) if new_layer is not None else str(old_layer)
        cmd = ChangeLayerCommand(self.document, name, old_layer, new_layer)
        self.document.execute_command(cmd)
        self.layer_stack_panel.refresh()

    def _commit_discrete_mask_field(self, mask_field, value):
        """For checkboxes/combos - a single click is already one gesture,
        no press/release to bracket."""
        layer = self._current_mask_layer()
        if layer is None:
            return
        self._begin_gesture()
        new_layer = self._with_mask_field(layer, mask_field, value)
        self._stage_layer_change(new_layer)
        self._commit_gesture()

    def _layers_equal(self, a, b):
        if a.mask is not b.mask:
            return False
        return (all(getattr(a, f) == getattr(b, f) for f in ADJUSTMENT_FIELDS)
                and a.visible == b.visible and a.label == b.label)

    # --- mask/layer helpers -------------------------------------------------

    def _current_mask_layer(self):
        if self._selected_mask_name is None:
            return None
        return next((l for l in self.document.layers if str(l) == self._selected_mask_name), None)

    def _all_mask_layers(self):
        return [l for l in self.document.layers if str(l).startswith("Mask ")]

    def _with_mask_field(self, layer, field, value):
        mask = layer.mask
        kwargs = {"components": list(mask.components), "feather": mask.feather,
                  "blur": mask.blur, "density": mask.density, "invert": mask.invert}
        kwargs[field] = value
        return layer.with_mask(Mask(**kwargs))

    def _with_component_param(self, layer, index, param_key, value):
        mask = layer.mask
        components = list(mask.components)
        if index < 0 or index >= len(components):
            return layer
        old = components[index]
        new_params = dict(old.params)
        new_params[param_key] = value
        components[index] = MaskComponent(kind=old.kind, params=new_params, op=old.op, invert=old.invert)
        new_mask = Mask(components=components, feather=mask.feather, blur=mask.blur,
                         density=mask.density, invert=mask.invert)
        return layer.with_mask(new_mask)

    def _with_component_full_params(self, layer, index, new_params):
        """Like _with_component_param, but replaces the whole params dict
        at once - used by canvas dragging, which typically changes several
        keys together (e.g. center_x and center_y in the same drag)."""
        mask = layer.mask
        components = list(mask.components)
        if index < 0 or index >= len(components):
            return layer
        old = components[index]
        components[index] = MaskComponent(kind=old.kind, params=dict(new_params), op=old.op, invert=old.invert)
        new_mask = Mask(components=components, feather=mask.feather, blur=mask.blur,
                         density=mask.density, invert=mask.invert)
        return layer.with_mask(new_mask)

    # --- canvas (interactive) mask editing -----------------------------
    #
    # Selecting a mask (or a component within it) immediately makes it
    # live on the canvas - both the red "where does this apply" overlay
    # and, for geometric shapes, direct drag interactivity - with no
    # separate arming step. This mirrors Lightroom: the mask you're
    # looking at in the panel is the mask you're looking at (and can
    # edit) on the image.

    def _selected_component(self):
        layer = self._current_mask_layer()
        if layer is None or self._selected_component_index is None:
            return None
        components = layer.mask.components
        if 0 <= self._selected_component_index < len(components):
            return components[self._selected_component_index]
        return None

    def _mask_overlay_provider(self, image):
        layer = self._current_mask_layer()
        if layer is None:
            return None
        try:
            return layer.mask.evaluate(image, layer.ai_registry)
        except Exception:
            return None

    def _on_viewer_crop_mode_changed(self, active):
        # Crop force-exits mask-edit mode and clears the overlay (see
        # ImageViewer.enter_crop_mode); once crop ends, restore whichever
        # mask is still selected in this panel.
        if not active:
            self._sync_canvas_state()

    def _should_show_overlay(self):
        # The overlay is off by default so the canvas always shows the
        # actual graded image (with every local adjustment visibly
        # applied) - not a translucent red wash sitting on top of it.
        # It still shows automatically for the moment you're actively
        # shaping a mask (dragging a handle/endpoint or painting a brush
        # stroke, see _set_dragging_overlay below), and the "Show Mask
        # Overlay" checkbox lets you pin it on deliberately, same as
        # Lightroom's "O" toggle.
        return self.show_overlay_checkbox.isChecked() or self._dragging_overlay_active

    def _apply_overlay_state(self):
        layer = self._current_mask_layer()
        if layer is not None and self._should_show_overlay():
            self.viewer.set_mask_overlay_provider(self._mask_overlay_provider, label=layer.label)
        else:
            self.viewer.set_mask_overlay_provider(None)

    def _set_dragging_overlay(self, active):
        if self._dragging_overlay_active == active:
            return
        self._dragging_overlay_active = active
        self._apply_overlay_state()
        component = self._selected_component()
        self._update_canvas_status_label(self._current_mask_layer(), component,
                                          component is not None and component.kind in INTERACTIVE_KINDS)

    def _sync_canvas_state(self):
        layer = self._current_mask_layer()
        component = self._selected_component()
        is_interactive = component is not None and component.kind in INTERACTIVE_KINDS

        self.brush_settings_container.setVisible(component is not None and component.kind == "brush")

        if is_interactive:
            self.viewer.enter_mask_edit_mode(component.kind, component.params)
            if component.kind == "brush":
                self.viewer.set_brush_settings(
                    radius=self.brush_size_slider.value() / 100.0,
                    hardness=float(self.brush_hardness_slider.value()),
                    flow=float(self.brush_flow_slider.value()),
                    mode="subtract" if self.brush_mode_combo.currentIndex() == 1 else "add",
                )
        else:
            self.viewer.exit_mask_edit_mode()

        self._apply_overlay_state()
        self._update_canvas_status_label(layer, component, is_interactive)

    def _update_canvas_status_label(self, layer, component, is_interactive):
        if layer is None:
            self.canvas_status_label.setText("")
            return
        if is_interactive:
            verb = "Paint" if component.kind == "brush" else \
                   "Click points to trace" if component.kind == "polygon" else "Drag"
            text = f"{verb} directly on the image to edit “{layer.label}”."
        else:
            text = f"“{layer.label}” is selected."
        if self._should_show_overlay():
            text += " Shown in red where it applies."
        self.canvas_status_label.setText(text)

    def _on_canvas_drag_started(self):
        self._begin_gesture()
        self._set_dragging_overlay(True)

    def _on_canvas_geometry_changed(self, params):
        layer = self._current_mask_layer()
        if layer is None or self._selected_component_index is None:
            return
        new_layer = self._with_component_full_params(layer, self._selected_component_index, params)
        self._stage_layer_change(new_layer)

    def _on_canvas_drag_finished(self):
        self._commit_gesture()
        self._rebuild_param_fields()  # reflect the dragged result in the numeric fields too
        self._set_dragging_overlay(False)

    def _on_canvas_brush_stroke(self, stroke):
        self._set_dragging_overlay(False)  # this stroke is done, even if the params below turn out to be stale
        layer = self._current_mask_layer()
        if layer is None or self._selected_component_index is None:
            return
        component = self._selected_component()
        if component is None or component.kind != "brush":
            return

        old_layer = layer
        new_params = dict(component.params)
        new_params["strokes"] = list(new_params.get("strokes", [])) + [stroke]
        new_layer = self._with_component_full_params(layer, self._selected_component_index, new_params)

        cmd = ChangeLayerCommand(self.document, self._selected_mask_name, old_layer, new_layer)
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def _on_canvas_polygon_finished(self, points):
        layer = self._current_mask_layer()
        if layer is None or self._selected_component_index is None:
            return
        component = self._selected_component()
        if component is None or component.kind != "polygon":
            return

        old_layer = layer
        new_params = dict(component.params)
        new_params["points"] = list(points)
        new_layer = self._with_component_full_params(layer, self._selected_component_index, new_params)

        cmd = ChangeLayerCommand(self.document, self._selected_mask_name, old_layer, new_layer)
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    # --- color range eyedropper ---------------------------------------------

    def _on_pick_color_toggled(self, checked):
        self.viewer.set_eyedropper_mode(checked, purpose="color_range")

    def _on_eyedropper_owner_changed(self, owner):
        # Another eyedropper (e.g. ControlsPanel's White Balance picker)
        # just took over - drop this button's checked state so the UI
        # never shows two "armed" eyedroppers at once.
        if self.pick_color_button is not None and owner != "color_range" and self.pick_color_button.isChecked():
            self.pick_color_button.blockSignals(True)
            self.pick_color_button.setChecked(False)
            self.pick_color_button.blockSignals(False)

    def _on_color_range_pixel_picked(self, px, py, purpose):
        if purpose != "color_range":
            return
        if self.pick_color_button is not None:
            self.pick_color_button.setChecked(False)  # one-shot, like the White Balance eyedropper

        layer = self._current_mask_layer()
        component = self._selected_component()
        if layer is None or component is None or component.kind != "color_range":
            return

        # Sampled from the actual rendered (display-referred, post-
        # adjustment) image at whatever resolution is currently on
        # screen - Color Range compares against that same image at mask-
        # evaluation time, so the picked reference color must come from
        # the same place for the click to mean what it looks like it means.
        rendered = self.document.render(max_dimension=self.viewer.render_queue.preview_max_dimension)
        h, w = rendered.shape[:2]
        px = max(0, min(w - 1, int(px)))
        py = max(0, min(h - 1, int(py)))
        r, g, b = (float(v) for v in rendered[py, px])

        old_layer = layer
        new_params = dict(component.params)
        new_params["sample_rgb"] = (r, g, b)
        new_layer = self._with_component_full_params(layer, self._selected_component_index, new_params)

        cmd = ChangeLayerCommand(self.document, self._selected_mask_name, old_layer, new_layer)
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()
        self._rebuild_param_fields()

    # --- add / delete / duplicate / move / rename masks ---------------------

    def _add_mask(self, kind):
        name = self.document.next_mask_name()
        component = MaskComponent(kind=kind, params=dict(DEFAULT_PARAMS_BY_KIND[kind]))
        mask = Mask(components=[component])
        layer = MaskedAdjustmentLayer(name, mask=mask, label=f"{name}: {KIND_LABELS[kind]}")

        cmd = ChangeLayerCommand(self.document, name, None, layer)
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

        self._selected_mask_name = name
        self._selected_component_index = 0
        self.refresh()

    def _delete_selected_mask(self):
        layer = self._current_mask_layer()
        if layer is None:
            return
        cmd = ChangeLayerCommand(self.document, self._selected_mask_name, layer, None)
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()
        self._selected_mask_name = None
        self._selected_component_index = None
        self.refresh()

    def _duplicate_selected_mask(self):
        layer = self._current_mask_layer()
        if layer is None:
            return
        new_name = self.document.next_mask_name()
        components = [MaskComponent(kind=c.kind, params=dict(c.params), op=c.op, invert=c.invert)
                      for c in layer.mask.components]
        new_mask = Mask(components=components, feather=layer.mask.feather, blur=layer.mask.blur,
                         density=layer.mask.density, invert=layer.mask.invert)
        adjustments = {f: getattr(layer, f) for f in ADJUSTMENT_FIELDS}
        new_layer = MaskedAdjustmentLayer(new_name, mask=new_mask, label=f"{layer.label} Copy",
                                           visible=layer.visible, **adjustments)

        cmd = ChangeLayerCommand(self.document, new_name, None, new_layer)
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()
        self._selected_mask_name = new_name
        self._selected_component_index = 0 if components else None
        self.refresh()

    def _rename_selected_mask(self):
        layer = self._current_mask_layer()
        if layer is None:
            return
        new_label, ok = QInputDialog.getText(self, "Rename Mask", "Name:", text=layer.label)
        if not ok or not new_label.strip():
            return
        old_layer = layer
        new_layer = layer.with_label(new_label.strip())
        cmd = ChangeLayerCommand(self.document, self._selected_mask_name, old_layer, new_layer)
        self.document.execute_command(cmd)
        self.refresh()

    def _move_selected_mask(self, delta):
        if self._selected_mask_name is None:
            return
        layers = self.document.layers
        indices = [i for i, l in enumerate(layers) if str(l) == self._selected_mask_name]
        if not indices:
            return
        i = indices[0]
        j = i + delta
        mask_names = {str(l) for l in self._all_mask_layers()}
        # Only swap past another mask layer - reordering relative to non-
        # mask layers is meaningless (Masks always run in their own final
        # pipeline stage) and would just be confusing in the layer list.
        while 0 <= j < len(layers) and str(layers[j]) not in mask_names:
            j += delta
        if not (0 <= j < len(layers)):
            return
        layers[i], layers[j] = layers[j], layers[i]
        self.document.history.append(f"Reorder {self._selected_mask_name}")
        self.document.redo_stack.clear()
        self.viewer.update_view()
        self.layer_stack_panel.refresh()
        self.refresh()

    def _on_mask_item_changed(self, item):
        if self._suppress_list_signal:
            return
        name = item.data(Qt.UserRole)
        layer = next((l for l in self.document.layers if str(l) == name), None)
        if layer is None:
            return
        visible = item.checkState() == Qt.Checked
        if layer.visible == visible:
            return
        new_layer = layer.with_visible(visible)
        cmd = ChangeLayerCommand(self.document, name, layer, new_layer)
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def _on_mask_row_changed(self, row):
        if row < 0:
            self._selected_mask_name = None
        else:
            item = self.mask_list.item(row)
            self._selected_mask_name = item.data(Qt.UserRole) if item else None
        self._selected_component_index = 0
        self._sync_editor()

    # --- components -----------------------------------------------------

    def _add_component(self, kind):
        layer = self._current_mask_layer()
        if layer is None:
            return
        old_layer = layer
        component = MaskComponent(kind=kind, params=dict(DEFAULT_PARAMS_BY_KIND[kind]),
                                   op="add" if not layer.mask.components else "add")
        new_components = list(layer.mask.components) + [component]
        new_mask = Mask(components=new_components, feather=layer.mask.feather, blur=layer.mask.blur,
                         density=layer.mask.density, invert=layer.mask.invert)
        new_layer = layer.with_mask(new_mask)
        cmd = ChangeLayerCommand(self.document, self._selected_mask_name, old_layer, new_layer)
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()
        self._selected_component_index = len(new_components) - 1
        self._sync_editor()

    def _delete_selected_component(self):
        layer = self._current_mask_layer()
        if layer is None or self._selected_component_index is None:
            return
        index = self._selected_component_index
        components = list(layer.mask.components)
        if index < 0 or index >= len(components):
            return
        old_layer = layer
        del components[index]
        new_mask = Mask(components=components, feather=layer.mask.feather, blur=layer.mask.blur,
                         density=layer.mask.density, invert=layer.mask.invert)
        new_layer = layer.with_mask(new_mask)
        cmd = ChangeLayerCommand(self.document, self._selected_mask_name, old_layer, new_layer)
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()
        self._selected_component_index = max(0, index - 1) if components else None
        self._sync_editor()

    def _on_component_row_changed(self, row):
        self._selected_component_index = row if row >= 0 else None
        self._rebuild_param_fields()

    def _on_component_op_changed(self, index):
        layer = self._current_mask_layer()
        if layer is None or self._selected_component_index is None:
            return
        ops = ["add", "subtract", "intersect"]
        old_layer = layer
        components = list(layer.mask.components)
        idx = self._selected_component_index
        if idx < 0 or idx >= len(components):
            return
        c = components[idx]
        components[idx] = MaskComponent(kind=c.kind, params=c.params, op=ops[index], invert=c.invert)
        new_mask = Mask(components=components, feather=layer.mask.feather, blur=layer.mask.blur,
                         density=layer.mask.density, invert=layer.mask.invert)
        new_layer = layer.with_mask(new_mask)
        cmd = ChangeLayerCommand(self.document, self._selected_mask_name, old_layer, new_layer)
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()
        self._refresh_component_list()

    def _on_component_invert_toggled(self, checked):
        layer = self._current_mask_layer()
        if layer is None or self._selected_component_index is None:
            return
        old_layer = layer
        components = list(layer.mask.components)
        idx = self._selected_component_index
        if idx < 0 or idx >= len(components):
            return
        c = components[idx]
        if c.invert == checked:
            return
        components[idx] = MaskComponent(kind=c.kind, params=c.params, op=c.op, invert=checked)
        new_mask = Mask(components=components, feather=layer.mask.feather, blur=layer.mask.blur,
                         density=layer.mask.density, invert=layer.mask.invert)
        new_layer = layer.with_mask(new_mask)
        cmd = ChangeLayerCommand(self.document, self._selected_mask_name, old_layer, new_layer)
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    # --- shape parameter fields (rebuilt per selected component's kind) -----

    def _clear_param_fields(self):
        while self.param_fields_layout.count():
            item = self.param_fields_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            layout = item.layout()
            if layout is not None:
                while layout.count():
                    sub = layout.takeAt(0)
                    if sub.widget() is not None:
                        sub.widget().deleteLater()

    def _rebuild_param_fields(self):
        self._clear_param_fields()
        layer = self._current_mask_layer()
        component = None
        if layer is not None and self._selected_component_index is not None:
            components = layer.mask.components
            if 0 <= self._selected_component_index < len(components):
                component = components[self._selected_component_index]

        self.pick_color_button = None
        if component is None:
            return

        self.component_help_label.setText(MASK_KIND_HELP.get(component.kind, ""))

        if component.kind == "color_range":
            self._add_pick_color_row(component)

        fields = PARAM_FIELDS_BY_KIND.get(component.kind, [])
        for param_key, label, minv, maxv, default, kind_hint in fields:
            current = component.params.get(param_key, _ui_to_value(kind_hint, default))
            ui_default = _value_to_ui(kind_hint, current)
            slider, spinbox, reset_btn = build_slider_row(
                self.param_fields_layout, label, minv, maxv, ui_default,
                self.control_label_width, self.slider_width, self.spinbox_width)

            def build_new_layer(mask_layer, raw, key=param_key, hint=kind_hint, idx=self._selected_component_index):
                return self._with_component_param(mask_layer, idx, key, _ui_to_value(hint, raw))

            self._bind_slider(slider, spinbox, reset_btn, build_new_layer, default)

    def _add_pick_color_row(self, component):
        row = QHBoxLayout()
        self.pick_color_button = QPushButton("Pick Color")
        self.pick_color_button.setCheckable(True)
        self.pick_color_button.setToolTip("Click a pixel on the image to use its color as the Color Range reference")
        row.addWidget(self.pick_color_button)

        r, g, b = component.params.get("sample_rgb", (0.5, 0.5, 0.5))
        swatch = QLabel()
        swatch.setFixedSize(20, 20)
        rgb255 = tuple(max(0, min(255, round(c * 255))) for c in (r, g, b))
        swatch.setStyleSheet(f"background-color: rgb{rgb255}; border: 1px solid {BORDER}; border-radius: 3px;")
        row.addWidget(swatch)
        row.addStretch(1)
        self.param_fields_layout.addLayout(row)

        self.pick_color_button.toggled.connect(self._on_pick_color_toggled)

    # --- syncing --------------------------------------------------------

    def refresh(self):
        self._refresh_mask_list()
        self._sync_editor()

    def _refresh_mask_list(self):
        self._suppress_list_signal = True
        self.mask_list.blockSignals(True)
        self.mask_list.clear()

        mask_layers = self._all_mask_layers()
        selected_row = -1
        for row, layer in enumerate(mask_layers):
            item = QListWidgetItem(layer.label)
            item.setData(Qt.UserRole, str(layer))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if layer.visible else Qt.Unchecked)
            self.mask_list.addItem(item)
            if str(layer) == self._selected_mask_name:
                selected_row = row

        self.mask_list.blockSignals(False)
        self._suppress_list_signal = False

        has_masks = bool(mask_layers)
        self.empty_hint.setVisible(not has_masks)

        if selected_row >= 0:
            self.mask_list.setCurrentRow(selected_row)
        elif has_masks:
            self.mask_list.setCurrentRow(0)
        else:
            self._selected_mask_name = None

    def _refresh_component_list(self):
        layer = self._current_mask_layer()
        self.component_list.blockSignals(True)
        self.component_list.clear()
        if layer is not None:
            for i, c in enumerate(layer.mask.components):
                op_label = "" if i == 0 else f" ({c.op.capitalize()})"
                text = f"{KIND_LABELS.get(c.kind, c.kind)}{op_label}{' [inverted]' if c.invert else ''}"
                self.component_list.addItem(text)
        self.component_list.blockSignals(False)

        count = self.component_list.count()
        if self._selected_component_index is None or self._selected_component_index >= count:
            self._selected_component_index = 0 if count else None
        if self._selected_component_index is not None:
            self.component_list.setCurrentRow(self._selected_component_index)

    def _sync_editor(self):
        layer = self._current_mask_layer()
        self.editor_container.setVisible(layer is not None)
        if layer is None:
            self._sync_canvas_state()
            return

        self._refresh_component_list()

        component = None
        if self._selected_component_index is not None and 0 <= self._selected_component_index < len(layer.mask.components):
            component = layer.mask.components[self._selected_component_index]
        self.component_op_combo.blockSignals(True)
        self.component_invert_check.blockSignals(True)
        if component is not None:
            self.component_op_combo.setCurrentIndex({"add": 0, "subtract": 1, "intersect": 2}.get(component.op, 0))
            self.component_op_combo.setEnabled(self._selected_component_index != 0)
            self.component_invert_check.setChecked(component.invert)
        self.component_op_combo.blockSignals(False)
        self.component_invert_check.blockSignals(False)

        self._rebuild_param_fields()

        self.invert_check.blockSignals(True)
        self.invert_check.setChecked(layer.mask.invert)
        self.invert_check.blockSignals(False)

        for slider, spinbox, reset_btn, value, default in (
            (self.feather_slider, self.feather_spinbox, self.feather_reset, layer.mask.feather, 0),
            (self.blur_slider, self.blur_spinbox, self.blur_reset, layer.mask.blur, 0),
            (self.density_slider, self.density_spinbox, self.density_reset, layer.mask.density, 100),
        ):
            slider.blockSignals(True)
            spinbox.blockSignals(True)
            slider.setValue(round(value))
            spinbox.setValue(round(value))
            slider.blockSignals(False)
            spinbox.blockSignals(False)
            reset_btn.setEnabled(round(value) != default)

        for field, (slider, spinbox, reset_btn) in self._adjustment_rows.items():
            value = round(getattr(layer, field))
            slider.blockSignals(True)
            spinbox.blockSignals(True)
            slider.setValue(value)
            spinbox.setValue(value)
            slider.blockSignals(False)
            spinbox.blockSignals(False)
            reset_btn.setEnabled(value != 0)

        self._sync_canvas_state()
