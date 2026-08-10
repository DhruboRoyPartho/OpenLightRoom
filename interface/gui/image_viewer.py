import math
from PySide6.QtWidgets import QLabel, QScrollArea
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QPainterPath
from PySide6.QtCore import Qt, Signal, QSize, QRectF, QPointF
import numpy as np
from core.threads.render_queue import RenderQueue
from core.adjustment_layers.geometry_layer import GeometryLayer
from core.processing.geometry import straighten_angle_from_line
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


class ImageViewer(QLabel):
    """The image canvas: renders the document (or, in before/after mode, the
    untouched base image), supports zooming/fitting inside its scroll area,
    and hosts an interactive crop-rectangle overlay when crop mode is active.
    """

    zoomChanged = Signal(float, bool)  # (effective zoom fraction, is_fit_mode)
    cropRectChanged = Signal(tuple)    # normalized (x0, y0, x1, y1), live while dragging
    angleChanged = Signal(float)       # pending straighten angle, live while dragging

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

        self.update_view()

    # --- rendering / display -------------------------------------------

    def update_view(self):
        if self._show_before:
            # base_image is scene-linear; it needs the same display
            # transform render() applies, just with no adjustment layers,
            # so "Before" shows the untouched photo as it actually looks
            # rather than raw linear data (which reads as far too dark).
            display = np.clip(linear_to_display(self.document.base_image), 0.0, 1.0)
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

    def _on_rendered(self, img):
        if img is None:
            return
        # img: float32, [0, 1], display-referred (from ImageDocument.render()
        # or the before-view transform above) - convert to uint8 for display.
        img_u8 = np.round(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        h, w, ch = img_u8.shape
        bytes_per_line = ch * w
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
        qimg = QImage(img_bgr.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self._base_pixmap = QPixmap.fromImage(qimg)
        self._apply_zoom_and_refresh()

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

    def exit_crop_mode(self):
        self._crop_mode = False
        self._drag_handle = None
        self._straighten_mode = False
        self._straighten_start = None
        self._straighten_end = None
        self.update_view()
        self.update()

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
        if self._crop_mode and self._straighten_mode and self._straighten_start is not None:
            self._finish_straighten_drag()
            return
        if self._crop_mode and self._drag_handle:
            self._drag_handle = None
            return
        super().mouseReleaseEvent(event)

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
