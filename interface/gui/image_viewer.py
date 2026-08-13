import math
from PySide6.QtWidgets import QLabel, QScrollArea
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QPainterPath
from PySide6.QtCore import Qt, Signal, QSize, QRectF, QPointF
import numpy as np
from core.threads.render_queue import RenderQueue
from core.adjustment_layers.geometry_layer import GeometryLayer
from core.processing.geometry import straighten_angle_from_line, downscale_to_max_dimension
from core.processing.color_space import linear_to_display
from interface.gui.theme import BG_CANVAS, TEXT_DIM, ACCENT
import cv2

FULL_RECT = (0.0, 0.0, 1.0, 1.0)

MIN_ZOOM = 0.05
MAX_ZOOM = 8.0
ZOOM_STEP = 1.25

HANDLE_HIT_PX = 14
MIN_CROP_FRACTION = 0.03  # smallest crop dimension, as a fraction of the frame
MAX_STRAIGHTEN_ANGLE = 45.0
MIN_STRAIGHTEN_DRAG_PX = 8  # ignore accidental micro-drags

# Mask-editing shape kinds that place/resize via canvas dragging - the
# rest (Luminance Range, Color Range's Refine, Subject/Sky/Skin) have no
# on-canvas geometry to place.
MASK_SHAPE_KINDS = ("radial", "ellipse", "rectangle")
POLYGON_CLOSE_HIT_PX = 10

# The mask overlay's "rubylith" tint - a translucent red wash over
# whatever the current mask actually selects, exactly like Lightroom's
# Overlay mask visualization (the 'O' key). Strength scales with the
# mask's own alpha, so a half-strength/feathered edge reads as a lighter
# tint rather than a hard red/not-red boundary.
MASK_OVERLAY_COLOR = np.array([1.0, 0.15, 0.15], dtype=np.float32)
MASK_OVERLAY_STRENGTH = 0.55


class ImageViewer(QLabel):
    """The image canvas: renders the document (or, in before/after mode, the
    untouched base image), supports zooming/fitting inside its scroll area,
    and hosts an interactive crop-rectangle overlay when crop mode is active.
    """

    zoomChanged = Signal(float, bool)  # (effective zoom fraction, is_fit_mode)
    cropRectChanged = Signal(tuple)    # normalized (x0, y0, x1, y1), live while dragging
    angleChanged = Signal(float)       # pending straighten angle, live while dragging
    cropModeChanged = Signal(bool)     # fires whenever crop mode is entered/exited, by ANY caller - lets
                                        # CanvasToolbar's Crop button stay in sync even when something else
                                        # (e.g. selecting a mask) force-exits crop mode
    pixelPicked = Signal(int, int, str)   # (x, y) in the rendered/pixmap pixel coordinate space, and which
                                           # eyedropper "owner" requested it (see set_eyedropper_mode)
    eyedropperOwnerChanged = Signal(str)  # fires whenever the active eyedropper purpose changes (""=none),
                                           # so two independent eyedropper toggles (White Balance, Color
                                           # Range) stay mutually exclusive without knowing about each other

    # --- mask editing (see enter_mask_edit_mode) ---------------------------
    maskDragStarted = Signal()          # a shape/gradient handle drag began - snapshot for undo
    maskGeometryChanged = Signal(dict)  # live params update while dragging a shape/gradient handle
    maskDragFinished = Signal()         # drag ended - commit the one undo step for the whole gesture
    maskBrushStrokeFinished = Signal(dict)   # one completed brush stroke: {"points", "radius", "hardness", "flow", "mode"}
    maskPolygonFinished = Signal(list)       # finished polygon: [(x, y), ...] normalized

    def __init__(self, document):
        super().__init__()
        self.document = document
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 300)
        self.setObjectName("ImageCanvas")
        self.setStyleSheet(
            f"QLabel#ImageCanvas {{ background-color: {BG_CANVAS}; color: {TEXT_DIM}; border: none; }}"
        )
        self.setMouseTracking(True)

        self.render_queue = RenderQueue(self.document)
        self.render_queue.image_rendered.connect(self._on_rendered)

        self._base_pixmap = None       # full-resolution QPixmap of the latest render
        self._show_before = False

        self._zoom = 1.0
        self._fit_mode = True
        self._effective_zoom = 1.0
        self._viewport_size = None

        self._crop_mode = False
        self._crop_rect = FULL_RECT
        self._aspect_ratio = None      # width/height float, or None for free
        self._drag_handle = None
        self._drag_start_pos = None
        self._drag_start_rect = FULL_RECT

        # The rotate90/flip carried over from whatever's already committed,
        # plus the in-progress straighten angle - both live-previewed
        # together with crop_rect=full-frame while the user adjusts them.
        self._base_rotation90 = 0
        self._base_flip_h = False
        self._base_flip_v = False
        self._pending_angle = 0.0

        self._straighten_mode = False
        self._straighten_start = None
        self._straighten_end = None

        self._eyedropper_mode = False
        self._eyedropper_purpose = ""

        # --- mask editing state (see enter_mask_edit_mode) -------------
        self._mask_edit_mode = False
        self._mask_edit_kind = None      # "radial" | "ellipse" | "rectangle" | "linear_gradient" | "brush" | "polygon"
        self._mask_edit_params = {}      # live mirror of the selected component's params (normalized coords)
        self._mask_drag_handle = None    # "center" | "corner" | "start" | "end" | None
        self._mask_drag_start_pos = None
        self._mask_drag_start_params = None
        self._brush_stroke_points = []   # accumulating current stroke, normalized coords
        self._brush_radius = 0.04        # fraction of the image diagonal
        self._brush_hardness = 80.0
        self._brush_flow = 100.0
        self._brush_mode = "add"
        self._mouse_pos = None           # last-seen widget-space mouse position, for the brush cursor preview
        self._polygon_points = []        # in-progress polygon, normalized coords

        # --- mask overlay ("where does this mask actually apply") ------
        self._last_rendered_image = None    # cached raw float32 render, so the overlay can be recomposited
        self._mask_overlay_provider = None  # callable(image) -> HxW alpha in [0,1], or None
        self._mask_overlay_label = ""

        self.update_view()

    # --- rendering / display -------------------------------------------

    def update_view(self):
        if self._show_before:
            # base_image is scene-linear; it needs the same display
            # transform render() applies, just with no adjustment layers,
            # so "Before" shows the untouched photo as it actually looks
            # rather than raw linear data (which reads as far too dark).
            # Respects the same preview-quality downscale as the graded
            # ("After") view, so toggling Before/After on a large image
            # doesn't reintroduce the slow full-resolution render it's
            # meant to avoid - this path runs synchronously on the UI
            # thread (unlike the queued/async "After" render), so it's
            # the one place a large, undownscaled image would actually be
            # felt as a freeze.
            base = downscale_to_max_dimension(self.document.base_image, self.render_queue.preview_max_dimension)
            display = np.clip(linear_to_display(base), 0.0, 1.0)
            self._on_rendered(display)
        elif self._crop_mode:
            override = GeometryLayer(
                self._base_rotation90, self._base_flip_h, self._base_flip_v,
                crop_rect=FULL_RECT, angle=self._pending_angle,
            )
            self.render_queue.request_render(geometry_override=override)
        else:
            self.render_queue.request_render()

    def set_show_before(self, show_before: bool):
        self._show_before = show_before
        self.update_view()

    def is_showing_before(self) -> bool:
        return self._show_before

    def set_preview_quality(self, max_dimension):
        """max_dimension: None for full resolution, or a pixel cap on the
        interactive preview's longer side (see
        core/processing/geometry.py:downscale_to_max_dimension). Only
        affects on-screen rendering here and in "Before" mode - export
        always renders the source at full resolution regardless of this
        setting, since it calls document.render() directly with no
        max_dimension."""
        self.render_queue.set_preview_max_dimension(max_dimension)
        self.update_view()

    def preview_quality(self):
        return self.render_queue.preview_max_dimension

    def _on_rendered(self, img):
        if img is None:
            return
        # img: float32, [0, 1], display-referred (from ImageDocument.render()
        # or the before-view transform above). Cached so the mask overlay
        # can be recomposited on top (toggling/updating it shouldn't need a
        # full pipeline re-render).
        self._last_rendered_image = img
        self._refresh_display_pixmap()

    def _refresh_display_pixmap(self):
        img = self._last_rendered_image
        if img is None:
            return
        display = img
        if self._mask_overlay_provider is not None:
            try:
                alpha = self._mask_overlay_provider(img)
            except Exception:
                alpha = None
            if alpha is not None:
                display = self._composite_mask_overlay(img, alpha)
        img_u8 = np.round(np.clip(display, 0.0, 1.0) * 255.0).astype(np.uint8)
        h, w, ch = img_u8.shape
        bytes_per_line = ch * w
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
        qimg = QImage(img_bgr.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self._base_pixmap = QPixmap.fromImage(qimg)
        self._apply_zoom_and_refresh()

    def _composite_mask_overlay(self, image, alpha):
        """Tints `image` red wherever `alpha` (HxW, [0,1]) selects it -
        Lightroom's "Overlay" mask visualization (the 'O' key). Strength
        scales with alpha so a feathered edge reads as a lighter tint
        rather than a hard boundary."""
        alpha = np.asarray(alpha, dtype=np.float32)
        if alpha.shape[:2] != image.shape[:2]:
            alpha = cv2.resize(alpha, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        mix = (alpha * MASK_OVERLAY_STRENGTH)[..., np.newaxis]
        return image * (1.0 - mix) + MASK_OVERLAY_COLOR[np.newaxis, np.newaxis, :] * mix

    def set_mask_overlay_provider(self, provider, label: str = ""):
        """provider: callable(image: np.ndarray) -> HxW alpha in [0,1], or
        None. Pass None to clear. Recomposites immediately from the cached
        last render - no re-render needed just to toggle/update this."""
        self._mask_overlay_provider = provider
        self._mask_overlay_label = label if provider is not None else ""
        self._refresh_display_pixmap()

    def is_mask_overlay_active(self) -> bool:
        return self._mask_overlay_provider is not None

    def mask_overlay_label(self) -> str:
        return self._mask_overlay_label

    # --- zoom / fit -------------------------------------------------------

    def on_viewport_resized(self, viewport_size: QSize):
        self._viewport_size = viewport_size
        if self._fit_mode:
            self._apply_zoom_and_refresh()

    def _fit_scale(self, pixmap: QPixmap) -> float:
        avail = self._viewport_size or self.parentWidget().size() if self.parentWidget() else None
        if not avail or avail.width() <= 0 or avail.height() <= 0 or pixmap.width() <= 0 or pixmap.height() <= 0:
            return 1.0
        return min(avail.width() / pixmap.width(), avail.height() / pixmap.height())

    def _apply_zoom_and_refresh(self):
        pm = self._base_pixmap
        if pm is None or pm.isNull():
            return

        effective_zoom = self._fit_scale(pm) if self._fit_mode else self._zoom
        effective_zoom = max(MIN_ZOOM, effective_zoom)
        self._effective_zoom = effective_zoom

        target_size = QSize(max(1, round(pm.width() * effective_zoom)), max(1, round(pm.height() * effective_zoom)))
        self.setFixedSize(target_size)
        scaled = pm.scaled(target_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

        self.zoomChanged.emit(effective_zoom, self._fit_mode)
        self.update()

    def zoom_in(self):
        self._fit_mode = False
        self._zoom = min(self._effective_zoom * ZOOM_STEP, MAX_ZOOM)
        self._apply_zoom_and_refresh()

    def zoom_out(self):
        self._fit_mode = False
        self._zoom = max(self._effective_zoom / ZOOM_STEP, MIN_ZOOM)
        self._apply_zoom_and_refresh()

    def set_fit(self):
        self._fit_mode = True
        self._apply_zoom_and_refresh()

    def set_actual_size(self):
        self._fit_mode = False
        self._zoom = 1.0
        self._apply_zoom_and_refresh()

    def is_fit_mode(self) -> bool:
        return self._fit_mode

    def current_zoom(self) -> float:
        return self._effective_zoom

    def resizeEvent(self, event):
        # Only relevant when not hosted in a CanvasScrollArea (which drives
        # fit sizing itself via on_viewport_resized).
        super().resizeEvent(event)

    # --- crop mode ----------------------------------------------------

    def is_crop_mode(self) -> bool:
        return self._crop_mode

    def enter_crop_mode(self, initial_rect=None, rotation90=0, flip_h=False, flip_v=False, angle=0.0):
        # Crop, mask-editing and the eyedropper all interpret canvas
        # mouse/keyboard input differently - only one may be active.
        if self._mask_edit_mode:
            self.exit_mask_edit_mode()
        if self._eyedropper_mode:
            self.set_eyedropper_mode(False)
        self.set_mask_overlay_provider(None)
        self._crop_mode = True
        self._crop_rect = tuple(initial_rect) if initial_rect else FULL_RECT
        self._base_rotation90 = rotation90
        self._base_flip_h = flip_h
        self._base_flip_v = flip_v
        self._pending_angle = angle
        self._straighten_mode = False
        self._straighten_start = None
        self._straighten_end = None
        self.set_fit()  # always show the whole frame while cropping
        self.update_view()  # re-render with crop_rect=full-frame at the current angle
        self.update()
        self.cropModeChanged.emit(True)

    def exit_crop_mode(self):
        self._crop_mode = False
        self._drag_handle = None
        self._straighten_mode = False
        self._straighten_start = None
        self._straighten_end = None
        self.update_view()
        self.update()
        self.cropModeChanged.emit(False)

    def get_crop_rect(self):
        return self._crop_rect

    def get_pending_angle(self) -> float:
        return self._pending_angle

    def set_pending_angle(self, angle: float):
        self._pending_angle = max(-MAX_STRAIGHTEN_ANGLE, min(MAX_STRAIGHTEN_ANGLE, angle))
        self.angleChanged.emit(self._pending_angle)
        self.update_view()

    def set_straighten_mode(self, enabled: bool):
        self._straighten_mode = enabled
        self._straighten_start = None
        self._straighten_end = None
        self._drag_handle = None
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self.update()

    def is_straighten_mode(self) -> bool:
        return self._straighten_mode

    # --- eyedropper mode -------------------------------------------------

    def set_eyedropper_mode(self, enabled: bool, purpose: str = ""):
        """purpose: an opaque string identifying who's asking (e.g.
        "white_balance", "color_range") - carried through to pixelPicked
        and eyedropperOwnerChanged so two independent eyedropper toggles
        elsewhere in the UI can stay mutually exclusive (only one can be
        "armed" at a time) without needing a reference to each other."""
        if enabled:
            if self._crop_mode:
                self.exit_crop_mode()
            if self._mask_edit_mode:
                self.exit_mask_edit_mode()
        self._eyedropper_mode = enabled
        self._eyedropper_purpose = purpose if enabled else ""
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self.eyedropperOwnerChanged.emit(self._eyedropper_purpose)

    def is_eyedropper_mode(self) -> bool:
        return self._eyedropper_mode

    def eyedropper_purpose(self) -> str:
        return self._eyedropper_purpose

    # --- mask editing -----------------------------------------------------

    def enter_mask_edit_mode(self, kind: str, params: dict):
        """kind: one of MaskComponent.kind (see core/masking/mask.py) -
        only the geometric ones (radial/ellipse/rectangle/linear_gradient/
        brush/polygon) do anything interactive here; the rest are simply
        not draggable and this becomes a no-op overlay. params: the
        selected component's current params dict, normalized-coordinate,
        mirrored live as the user drags."""
        if self._crop_mode:
            self.exit_crop_mode()
        if self._eyedropper_mode:
            self.set_eyedropper_mode(False)
        self._mask_edit_mode = True
        self._mask_edit_kind = kind
        self._mask_edit_params = dict(params)
        self._mask_drag_handle = None
        self._polygon_points = []
        cursor = Qt.CrossCursor if kind in ("brush", "polygon") else Qt.ArrowCursor
        self.setCursor(cursor)
        self.update()

    def exit_mask_edit_mode(self):
        self._mask_edit_mode = False
        self._mask_edit_kind = None
        self._mask_drag_handle = None
        self._polygon_points = []
        self._mouse_pos = None  # don't leave a stale brush/polygon cursor position behind
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def is_mask_edit_mode(self) -> bool:
        return self._mask_edit_mode

    def set_mask_edit_params(self, params: dict):
        """Pushes updated params into the live overlay - called when a
        numeric field in the Masks panel changes the shape a different
        way than dragging did, so the canvas overlay stays in sync."""
        if self._mask_edit_mode:
            self._mask_edit_params = dict(params)
            self.update()

    def set_brush_settings(self, radius: float = None, hardness: float = None,
                            flow: float = None, mode: str = None):
        if radius is not None:
            self._brush_radius = radius
        if hardness is not None:
            self._brush_hardness = hardness
        if flow is not None:
            self._brush_flow = flow
        if mode is not None:
            self._brush_mode = mode

    def current_pixmap_size(self):
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return None
        return self._base_pixmap.width(), self._base_pixmap.height()

    def reset_crop_rect(self):
        self._crop_rect = FULL_RECT
        self.cropRectChanged.emit(self._crop_rect)
        self.update()

    def set_crop_aspect(self, ratio):
        """ratio: width/height float, or None for free-form."""
        self._aspect_ratio = ratio
        if ratio:
            self._crop_rect = self._rect_fit_to_aspect(self._crop_rect, ratio)
            self.cropRectChanged.emit(self._crop_rect)
            self.update()

    def _rect_fit_to_aspect(self, rect, ratio):
        pm = self._base_pixmap
        if pm is None or pm.isNull():
            return rect
        img_w, img_h = pm.width(), pm.height()
        x0, y0, x1, y1 = rect
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

        width_px = (x1 - x0) * img_w
        height_px = width_px / ratio
        if height_px > img_h:
            height_px = img_h
            width_px = height_px * ratio

        w_norm = width_px / img_w
        h_norm = height_px / img_h
        nx0 = max(0.0, min(cx - w_norm / 2.0, 1.0 - w_norm))
        ny0 = max(0.0, min(cy - h_norm / 2.0, 1.0 - h_norm))
        return (nx0, ny0, nx0 + w_norm, ny0 + h_norm)

    # --- painting -------------------------------------------------------

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._crop_mode and self._base_pixmap is not None:
            self._paint_crop_overlay()
            if self._straighten_mode and self._straighten_start is not None and self._straighten_end is not None:
                self._paint_straighten_line()
        if self._mask_edit_mode and self._base_pixmap is not None:
            self._paint_mask_overlay()

    def _crop_widget_rect(self) -> QRectF:
        x0, y0, x1, y1 = self._crop_rect
        w, h = self.width(), self.height()
        return QRectF(x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h)

    def _paint_crop_overlay(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        full = QRectF(0, 0, self.width(), self.height())
        crop_rect = self._crop_widget_rect()

        dim_path = QPainterPath()
        dim_path.addRect(full)
        inner_path = QPainterPath()
        inner_path.addRect(crop_rect)
        painter.fillPath(dim_path.subtracted(inner_path), QColor(0, 0, 0, 140))

        painter.setPen(QPen(QColor(255, 255, 255, 130), 1))
        for i in (1, 2):
            fx = crop_rect.left() + crop_rect.width() * i / 3
            painter.drawLine(QPointF(fx, crop_rect.top()), QPointF(fx, crop_rect.bottom()))
            fy = crop_rect.top() + crop_rect.height() * i / 3
            painter.drawLine(QPointF(crop_rect.left(), fy), QPointF(crop_rect.right(), fy))

        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(crop_rect)

        handle = 9
        painter.setPen(QPen(QColor(ACCENT), 1.5))
        painter.setBrush(QColor("#ffffff"))
        for cx, cy in self._handle_positions(crop_rect):
            painter.drawRect(QRectF(cx - handle / 2, cy - handle / 2, handle, handle))

    def _paint_straighten_line(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(ACCENT), 2)
        painter.setPen(pen)
        painter.drawLine(self._straighten_start, self._straighten_end)
        for pt in (self._straighten_start, self._straighten_end):
            painter.setBrush(QColor("#ffffff"))
            painter.drawEllipse(pt, 4, 4)

    def _handle_positions(self, crop_rect: QRectF):
        return [
            (crop_rect.left(), crop_rect.top()),
            (crop_rect.right(), crop_rect.top()),
            (crop_rect.left(), crop_rect.bottom()),
            (crop_rect.right(), crop_rect.bottom()),
        ]

    def _handle_names(self):
        return ["tl", "tr", "bl", "br"]

    def _hit_crop_handle(self, pos: QPointF):
        crop_rect = self._crop_widget_rect()
        for name, (hx, hy) in zip(self._handle_names(), self._handle_positions(crop_rect)):
            if (pos.x() - hx) ** 2 + (pos.y() - hy) ** 2 <= HANDLE_HIT_PX ** 2:
                return name
        if crop_rect.contains(pos):
            return "move"
        return None

    # --- crop mouse interaction ----------------------------------------

    def mousePressEvent(self, event):
        if self._mask_edit_mode and event.button() == Qt.LeftButton:
            self._handle_mask_mouse_press(event.position())
            return
        if self._eyedropper_mode and not self._crop_mode and event.button() == Qt.LeftButton:
            if self._effective_zoom > 0:
                px = int(event.position().x() / self._effective_zoom)
                py = int(event.position().y() / self._effective_zoom)
                self.pixelPicked.emit(px, py, self._eyedropper_purpose)
            return
        if self._crop_mode and self._straighten_mode and event.button() == Qt.LeftButton:
            self._straighten_start = event.position()
            self._straighten_end = event.position()
            self.update()
            return
        if self._crop_mode and event.button() == Qt.LeftButton:
            handle = self._hit_crop_handle(event.position())
            if handle:
                self._drag_handle = handle
                self._drag_start_pos = event.position()
                self._drag_start_rect = self._crop_rect
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._mask_edit_mode:
            self._mouse_pos = event.position()
            self._handle_mask_mouse_move(event.position())
            return
        if self._crop_mode and self._straighten_mode and self._straighten_start is not None:
            self._straighten_end = event.position()
            self.update()
            return
        if self._crop_mode and self._drag_handle:
            self._update_crop_drag(event.position())
            return
        if self._crop_mode and self._straighten_mode:
            return  # cursor already set to CrossCursor by set_straighten_mode
        if self._crop_mode:
            handle = self._hit_crop_handle(event.position())
            self.setCursor(Qt.SizeAllCursor if handle else Qt.ArrowCursor)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._mask_edit_mode and event.button() == Qt.LeftButton:
            self._handle_mask_mouse_release()
            return
        if self._crop_mode and self._straighten_mode and self._straighten_start is not None:
            self._finish_straighten_drag()
            return
        if self._crop_mode and self._drag_handle:
            self._drag_handle = None
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        # Without this, the brush cursor's preview circle (and the
        # polygon tool's rubber-band line to the pointer) stay painted at
        # their last position after the mouse leaves the canvas - e.g.
        # moving over to a slider in the Masks panel - which reads as a
        # stuck/leftover mark rather than a live cursor. Only affects the
        # hover preview itself; an in-progress stroke's already-recorded
        # points (_brush_stroke_points) are untouched.
        if self._mouse_pos is not None:
            self._mouse_pos = None
            self.update()
        super().leaveEvent(event)

    def _finish_straighten_drag(self):
        start, end = self._straighten_start, self._straighten_end
        self._straighten_start = None
        self._straighten_end = None
        if start is None or end is None:
            self.update()
            return

        dx, dy = end.x() - start.x(), end.y() - start.y()
        if dx * dx + dy * dy >= MIN_STRAIGHTEN_DRAG_PX ** 2:
            delta = straighten_angle_from_line(dx, dy)
            self.set_pending_angle(self._pending_angle + delta)
        self.update()

    def _update_crop_drag(self, pos: QPointF):
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        dx = (pos.x() - self._drag_start_pos.x()) / w
        dy = (pos.y() - self._drag_start_pos.y()) / h
        x0, y0, x1, y1 = self._drag_start_rect

        if self._drag_handle == "move":
            rw, rh = x1 - x0, y1 - y0
            nx0 = min(max(x0 + dx, 0.0), 1.0 - rw)
            ny0 = min(max(y0 + dy, 0.0), 1.0 - rh)
            new_rect = (nx0, ny0, nx0 + rw, ny0 + rh)
        else:
            nx0, ny0, nx1, ny1 = x0, y0, x1, y1
            if "l" in self._drag_handle:
                nx0 = min(max(x0 + dx, 0.0), x1 - MIN_CROP_FRACTION)
            if "r" in self._drag_handle:
                nx1 = max(min(x1 + dx, 1.0), x0 + MIN_CROP_FRACTION)
            if "t" in self._drag_handle:
                ny0 = min(max(y0 + dy, 0.0), y1 - MIN_CROP_FRACTION)
            if "b" in self._drag_handle:
                ny1 = max(min(y1 + dy, 1.0), y0 + MIN_CROP_FRACTION)

            if self._aspect_ratio and self._base_pixmap is not None:
                img_w, img_h = self._base_pixmap.width(), self._base_pixmap.height()
                ratio = self._aspect_ratio

                anchor_x = x1 if "l" in self._drag_handle else x0
                anchor_y = y1 if "t" in self._drag_handle else y0
                drag_x = nx0 if "l" in self._drag_handle else nx1
                drag_y = ny0 if "t" in self._drag_handle else ny1

                width_px = abs(drag_x - anchor_x) * img_w
                height_px = abs(drag_y - anchor_y) * img_h
                if height_px <= 0 or width_px / max(height_px, 1e-6) >= ratio:
                    height_px = width_px / ratio
                else:
                    width_px = height_px * ratio

                w_norm = width_px / img_w
                h_norm = height_px / img_h

                if "l" in self._drag_handle:
                    nx0 = anchor_x - w_norm
                else:
                    nx1 = anchor_x + w_norm
                if "t" in self._drag_handle:
                    ny0 = anchor_y - h_norm
                else:
                    ny1 = anchor_y + h_norm

            nx0 = max(nx0, 0.0)
            ny0 = max(ny0, 0.0)
            nx1 = min(nx1, 1.0)
            ny1 = min(ny1, 1.0)
            new_rect = (nx0, ny0, nx1, ny1)

        self._crop_rect = new_rect
        self.cropRectChanged.emit(new_rect)
        self.update()

    # --- mask editing: geometry --------------------------------------------

    def _to_normalized(self, pos: QPointF):
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return (0.0, 0.0)
        return (min(max(pos.x() / w, 0.0), 1.0), min(max(pos.y() / h, 0.0), 1.0))

    def _mask_shape_handles_widget(self):
        """{'center', 'corner'} in widget pixel coords, for the current
        _mask_edit_params - radial/ellipse/rectangle only."""
        p = self._mask_edit_params
        w, h = self.width(), self.height()
        cx, cy = p.get("center_x", 0.5) * w, p.get("center_y", 0.5) * h
        if self._mask_edit_kind in ("radial", "ellipse"):
            rx, ry = p.get("radius_x", 0.25) * w, p.get("radius_y", 0.25) * h
        else:  # rectangle
            rx, ry = p.get("half_width", 0.25) * w, p.get("half_height", 0.25) * h
        return {"center": QPointF(cx, cy), "corner": QPointF(cx + rx, cy + ry)}

    def _mask_gradient_handles_widget(self):
        p = self._mask_edit_params
        w, h = self.width(), self.height()
        return {
            "start": QPointF(p.get("x0", 0.3) * w, p.get("y0", 0.5) * h),
            "end": QPointF(p.get("x1", 0.7) * w, p.get("y1", 0.5) * h),
        }

    def _hit_mask_handle(self, pos: QPointF):
        if self._mask_edit_kind in MASK_SHAPE_KINDS:
            handles = self._mask_shape_handles_widget()
        elif self._mask_edit_kind == "linear_gradient":
            handles = self._mask_gradient_handles_widget()
        else:
            return None

        for name, point in handles.items():
            if (pos.x() - point.x()) ** 2 + (pos.y() - point.y()) ** 2 <= HANDLE_HIT_PX ** 2:
                return name

        if self._mask_edit_kind in MASK_SHAPE_KINDS:
            center, corner = handles["center"], handles["corner"]
            rx, ry = abs(corner.x() - center.x()), abs(corner.y() - center.y())
            if rx > 0 and ry > 0:
                dist = ((pos.x() - center.x()) / rx) ** 2 + ((pos.y() - center.y()) / ry) ** 2
                if dist <= 1.0:
                    return "center"  # clicked inside the shape body - drag to move
        return None

    # --- mask editing: mouse interaction -----------------------------------

    def _handle_mask_mouse_press(self, pos: QPointF):
        kind = self._mask_edit_kind
        if kind in MASK_SHAPE_KINDS or kind == "linear_gradient":
            handle = self._hit_mask_handle(pos)
            if handle:
                self._mask_drag_handle = handle
                self._mask_drag_start_pos = pos
                self._mask_drag_start_params = dict(self._mask_edit_params)
                self.maskDragStarted.emit()
            return
        if kind == "brush":
            self._brush_stroke_points = [self._to_normalized(pos)]
            self.maskDragStarted.emit()
            self.update()
            return
        if kind == "polygon":
            norm = self._to_normalized(pos)
            if len(self._polygon_points) >= 3:
                first = self._polygon_points[0]
                fw, fh = self.width(), self.height()
                if (pos.x() - first[0] * fw) ** 2 + (pos.y() - first[1] * fh) ** 2 <= POLYGON_CLOSE_HIT_PX ** 2:
                    self._finish_polygon()
                    return
            self._polygon_points.append(norm)
            self.update()
            return

    def _handle_mask_mouse_move(self, pos: QPointF):
        kind = self._mask_edit_kind
        if kind in MASK_SHAPE_KINDS and self._mask_drag_handle:
            self._update_mask_shape_drag(pos)
            return
        if kind == "linear_gradient" and self._mask_drag_handle:
            self._update_mask_gradient_drag(pos)
            return
        if kind == "brush" and self._brush_stroke_points:
            self._brush_stroke_points.append(self._to_normalized(pos))
            self.update()
            return
        if kind in ("brush", "polygon"):
            self.update()  # redraw the brush cursor / polygon rubber-band line to the new mouse position

    def _handle_mask_mouse_release(self):
        kind = self._mask_edit_kind
        if kind in MASK_SHAPE_KINDS or kind == "linear_gradient":
            if self._mask_drag_handle:
                self._mask_drag_handle = None
                self._mask_drag_start_pos = None
                self._mask_drag_start_params = None
                self.maskDragFinished.emit()
            return
        if kind == "brush":
            if self._brush_stroke_points:
                stroke = {
                    "points": list(self._brush_stroke_points),
                    "radius": self._brush_radius,
                    "hardness": self._brush_hardness,
                    "flow": self._brush_flow,
                    "mode": self._brush_mode,
                }
                self._brush_stroke_points = []
                self.maskBrushStrokeFinished.emit(stroke)
            self.update()
            return

    def _update_mask_shape_drag(self, pos: QPointF):
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0 or self._mask_drag_start_pos is None:
            return
        dx = (pos.x() - self._mask_drag_start_pos.x()) / w
        dy = (pos.y() - self._mask_drag_start_pos.y()) / h
        start = self._mask_drag_start_params
        new_params = dict(self._mask_edit_params)

        if self._mask_drag_handle == "center":
            new_params["center_x"] = min(max(start.get("center_x", 0.5) + dx, 0.0), 1.0)
            new_params["center_y"] = min(max(start.get("center_y", 0.5) + dy, 0.0), 1.0)
        elif self._mask_drag_handle == "corner":
            if self._mask_edit_kind in ("radial", "ellipse"):
                new_params["radius_x"] = max(0.01, start.get("radius_x", 0.25) + dx)
                new_params["radius_y"] = max(0.01, start.get("radius_y", 0.25) + dy)
            else:
                new_params["half_width"] = max(0.01, start.get("half_width", 0.25) + dx)
                new_params["half_height"] = max(0.01, start.get("half_height", 0.25) + dy)

        self._mask_edit_params = new_params
        self.maskGeometryChanged.emit(dict(new_params))
        self.update()

    def _update_mask_gradient_drag(self, pos: QPointF):
        new_params = dict(self._mask_edit_params)
        norm = self._to_normalized(pos)
        if self._mask_drag_handle == "start":
            new_params["x0"], new_params["y0"] = norm
        else:
            new_params["x1"], new_params["y1"] = norm
        self._mask_edit_params = new_params
        self.maskGeometryChanged.emit(dict(new_params))
        self.update()

    def _finish_polygon(self):
        points = list(self._polygon_points)
        self._polygon_points = []
        self.update()
        if len(points) >= 3:
            self.maskPolygonFinished.emit(points)

    # --- mask editing: painting ---------------------------------------------

    def _paint_mask_overlay(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        kind = self._mask_edit_kind

        if kind in MASK_SHAPE_KINDS:
            self._paint_mask_shape(painter)
        elif kind == "linear_gradient":
            self._paint_mask_gradient(painter)
        elif kind == "brush":
            self._paint_mask_brush(painter)
        elif kind == "polygon":
            self._paint_mask_polygon(painter)

    def _paint_mask_shape(self, painter: QPainter):
        handles = self._mask_shape_handles_widget()
        center, corner = handles["center"], handles["corner"]
        rx, ry = abs(corner.x() - center.x()), abs(corner.y() - center.y())
        rect = QRectF(center.x() - rx, center.y() - ry, rx * 2, ry * 2)

        painter.setPen(QPen(QColor(ACCENT), 2))
        painter.setBrush(Qt.NoBrush)
        if self._mask_edit_kind == "rectangle":
            painter.drawRect(rect)
        else:
            painter.drawEllipse(rect)

        handle_r = 5.0
        painter.setPen(QPen(QColor(ACCENT), 1.5))
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(center, 4, 4)
        painter.drawRect(QRectF(corner.x() - handle_r, corner.y() - handle_r, handle_r * 2, handle_r * 2))

    def _paint_mask_gradient(self, painter: QPainter):
        handles = self._mask_gradient_handles_widget()
        start, end = handles["start"], handles["end"]
        painter.setPen(QPen(QColor(ACCENT), 2))
        painter.drawLine(start, end)
        painter.setPen(QPen(QColor(ACCENT), 1.5))
        painter.setBrush(QColor("#ffffff"))
        for pt in (start, end):
            painter.drawEllipse(pt, 5, 5)

    def _paint_mask_brush(self, painter: QPainter):
        w, h = self.width(), self.height()
        diagonal = math.hypot(w, h)
        radius_px = max(self._brush_radius * diagonal, 1.0)

        if len(self._brush_stroke_points) >= 2:
            pen = QPen(QColor(ACCENT), max(2.0, radius_px * 0.15))
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            path = QPainterPath()
            fx, fy = self._brush_stroke_points[0]
            path.moveTo(fx * w, fy * h)
            for x, y in self._brush_stroke_points[1:]:
                path.lineTo(x * w, y * h)
            painter.drawPath(path)

        if self._mouse_pos is not None:
            painter.setPen(QPen(QColor("#ffffff"), 1.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(self._mouse_pos, radius_px, radius_px)

    def _paint_mask_polygon(self, painter: QPainter):
        if not self._polygon_points:
            return
        w, h = self.width(), self.height()
        painter.setPen(QPen(QColor(ACCENT), 2))
        path = QPainterPath()
        fx, fy = self._polygon_points[0]
        path.moveTo(fx * w, fy * h)
        for x, y in self._polygon_points[1:]:
            path.lineTo(x * w, y * h)
        if self._mouse_pos is not None:
            path.lineTo(self._mouse_pos)
        painter.drawPath(path)

        painter.setPen(QPen(QColor(ACCENT), 1.5))
        painter.setBrush(QColor("#ffffff"))
        for x, y in self._polygon_points:
            painter.drawEllipse(QPointF(x * w, y * h), 4, 4)


class CanvasScrollArea(QScrollArea):
    """Hosts the ImageViewer, providing native scrollbar panning when zoomed
    beyond the viewport, and drives the viewer's fit-to-window sizing."""

    def __init__(self, image_viewer: ImageViewer):
        super().__init__()
        self._image_viewer = image_viewer
        self.setWidget(image_viewer)
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignCenter)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setStyleSheet(f"QScrollArea {{ background-color: {BG_CANVAS}; border: none; }}")
        viewport = self.viewport()
        if viewport is not None:
            viewport.setStyleSheet(f"background-color: {BG_CANVAS};")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._image_viewer.on_viewport_resized(self.viewport().size())

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self._image_viewer.zoom_in()
            else:
                self._image_viewer.zoom_out()
            event.accept()
            return
        super().wheelEvent(event)
