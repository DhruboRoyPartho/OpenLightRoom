import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QColor

from core.scopes.waveform import compute_waveform
from core.scopes.rgb_parade import compute_rgb_parade
from core.scopes.vectorscope import compute_vectorscope
from interface.gui.histogram_widget import HistogramWidget
from interface.gui.theme import BG_FIELD, BORDER

MODES = ["Histogram", "Waveform", "RGB Parade", "Vectorscope"]
SCOPE_SIZE = 256


class ScopesPanel(QWidget):
    """Histogram (full quality, see HistogramWidget) plus Waveform/RGB
    Parade/Vectorscope (basic display: each is rendered straight from its
    core/scopes density map into a QImage). All four are generated from
    whatever image set_image() is given - the actual rendered/processed
    output, not a placeholder or a cached approximation - so they read
    correctly through every edit, including a live crop/straighten
    preview.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(MODES)
        layout.addWidget(self.mode_combo)

        self.histogram_widget = HistogramWidget()
        self.histogram_widget.setFixedHeight(140)
        layout.addWidget(self.histogram_widget)

        self.basic_scope_label = QLabel()
        self.basic_scope_label.setFixedHeight(140)
        self.basic_scope_label.setAlignment(Qt.AlignCenter)
        self.basic_scope_label.setStyleSheet(
            f"background-color: {BG_FIELD}; border: 1px solid {BORDER}; border-radius: 6px;"
        )
        self.basic_scope_label.setVisible(False)
        layout.addWidget(self.basic_scope_label)

        self._current_image = None
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self._on_mode_changed(self.mode_combo.currentText())

    def set_image(self, image):
        self._current_image = image
        self.histogram_widget.set_image(image)
        self._refresh_basic_scope()

    def _on_mode_changed(self, mode):
        is_histogram = mode == "Histogram"
        self.histogram_widget.setVisible(is_histogram)
        self.basic_scope_label.setVisible(not is_histogram)
        self._refresh_basic_scope()

    def _refresh_basic_scope(self):
        mode = self.mode_combo.currentText()
        if mode == "Histogram" or self._current_image is None:
            self.basic_scope_label.clear()
            return

        if mode == "Waveform":
            data = compute_waveform(self._current_image, out_width=SCOPE_SIZE, out_height=128)
            qimg = _density_to_qimage(data)
        elif mode == "RGB Parade":
            parade = compute_rgb_parade(self._current_image, out_width=SCOPE_SIZE, out_height=128)
            qimg = _parade_to_qimage(parade)
        elif mode == "Vectorscope":
            data = compute_vectorscope(self._current_image, size=180)
            qimg = _density_to_qimage(data, tint=QColor(210, 210, 210))
        else:
            return

        pixmap = QPixmap.fromImage(qimg).scaled(
            self.basic_scope_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.basic_scope_label.setPixmap(pixmap)


def _normalize_to_u8(data: np.ndarray) -> np.ndarray:
    # log1p scaling: a scope's tallest/densest bin (a flat sky, a gray
    # backdrop) would otherwise dwarf everything else into invisible
    # near-zero-brightness pixels.
    log_data = np.log1p(data.astype(np.float64))
    peak = max(float(log_data.max()), 1e-6)
    return np.clip(log_data / peak * 255.0, 0.0, 255.0).astype(np.uint8)


def _density_to_qimage(data: np.ndarray, tint: QColor = None) -> QImage:
    norm = _normalize_to_u8(data)
    h, w = norm.shape
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    if tint is None:
        rgb[..., 0] = norm
        rgb[..., 1] = norm
        rgb[..., 2] = norm
    else:
        rgb[..., 0] = (norm.astype(np.float32) * tint.redF()).astype(np.uint8)
        rgb[..., 1] = (norm.astype(np.float32) * tint.greenF()).astype(np.uint8)
        rgb[..., 2] = (norm.astype(np.float32) * tint.blueF()).astype(np.uint8)
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return qimg.copy()  # detach from the numpy buffer before it's freed


def _parade_to_qimage(parade: dict) -> QImage:
    r = _normalize_to_u8(parade["R"])
    g = _normalize_to_u8(parade["G"])
    b = _normalize_to_u8(parade["B"])
    rgb = np.ascontiguousarray(np.stack([r, g, b], axis=-1))
    h, w = r.shape
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return qimg.copy()
