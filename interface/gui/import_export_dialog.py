from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QComboBox
from PySide6.QtCore import Qt

from interface.gui.theme import TEXT, TEXT_DIM, ACCENT

FIELD_LABEL_STYLE = f"color: {TEXT_DIM}; font-size: 11px; font-weight: 600;"
HINT_LABEL_STYLE = f"color: {TEXT_DIM}; font-size: 10px; font-style: italic;"
EXPORT_BUTTON_STYLE = f"""
    QPushButton {{
        background-color: {ACCENT};
        border: 1px solid {ACCENT};
        color: #ffffff;
        font-weight: 600;
    }}
    QPushButton:hover {{ background-color: #4a83f5; border-color: #4a83f5; }}
    QPushButton:pressed {{ background-color: #2559c4; border-color: #2559c4; }}
"""

# Which export formats actually use each control - PNG's compression
# level is driven by the same "Quality" slider as JPEG/WEBP (see
# core/io/image_io.py:save_image), which the dialog used to not reflect;
# TIFF ignores quality entirely and is lossless regardless. Bit depth only
# has a 16-bit mode for PNG/TIFF - JPEG/WEBP have none and are always
# 8-bit no matter what's selected.
_SUPPORTS_QUALITY = {"JPG", "PNG", "WEBP"}
_SUPPORTS_16BIT = {"PNG", "TIFF"}


def format_capabilities(fmt: str):
    """(supports_quality, supports_16bit) for an export format name
    (case-insensitive) - the single source of truth for which formats
    support what, shared between ExportDialog and the Preferences
    dialog's Export Defaults section so the two never drift apart."""
    fmt = fmt.upper()
    return fmt in _SUPPORTS_QUALITY, fmt in _SUPPORTS_16BIT


class ExportDialog(QDialog):
    def __init__(self, parent=None, default_format="JPG", default_quality=95, default_bit_depth="8-bit"):
        super().__init__(parent)
        self.setWindowTitle("Export Image")
        self.setModal(True)
        self.setFixedWidth(340)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(6)

        title = QLabel("Export Image")
        title.setStyleSheet(f"color: {TEXT}; font-size: 14px; font-weight: 600;")
        layout.addWidget(title)
        layout.addSpacing(8)

        layout.addWidget(self._field_label("Format"))
        self.format_box = QComboBox()
        self.format_box.addItems(["JPG", "PNG", "TIFF", "WEBP"])
        self.format_box.setCurrentText(default_format)
        layout.addWidget(self.format_box)
        layout.addSpacing(10)

        layout.addWidget(self._field_label("Bit Depth"))
        self.bit_depth_box = QComboBox()
        self.bit_depth_box.addItems(["8-bit", "16-bit"])
        self.bit_depth_box.setCurrentText(default_bit_depth)
        layout.addWidget(self.bit_depth_box)
        self.bit_depth_hint = QLabel()
        self.bit_depth_hint.setStyleSheet(HINT_LABEL_STYLE)
        layout.addWidget(self.bit_depth_hint)
        layout.addSpacing(10)

        quality_row = QHBoxLayout()
        self.quality_label = QLabel("Quality")
        self.quality_label.setStyleSheet(FIELD_LABEL_STYLE)
        quality_row.addWidget(self.quality_label)
        quality_row.addStretch(1)
        self.quality_value_label = QLabel(str(default_quality))
        self.quality_value_label.setStyleSheet(f"color: {TEXT_DIM}; font-family: Consolas, monospace; font-size: 11px;")
        quality_row.addWidget(self.quality_value_label)
        layout.addLayout(quality_row)

        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(10, 100)
        self.quality_slider.setValue(default_quality)
        layout.addWidget(self.quality_slider)
        self.quality_hint = QLabel()
        self.quality_hint.setStyleSheet(HINT_LABEL_STYLE)
        layout.addWidget(self.quality_hint)

        layout.addSpacing(12)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.cancel_btn = QPushButton("Cancel")
        self.export_btn = QPushButton("Export")
        self.export_btn.setDefault(True)
        self.export_btn.setStyleSheet(EXPORT_BUTTON_STYLE)
        button_row.addWidget(self.cancel_btn)
        button_row.addWidget(self.export_btn)
        layout.addLayout(button_row)

        self.cancel_btn.clicked.connect(self.reject)
        self.format_box.currentTextChanged.connect(self._update_field_availability)
        self.quality_slider.valueChanged.connect(lambda v: self.quality_value_label.setText(str(v)))

        self._update_field_availability(self.format_box.currentText())

    def _field_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(FIELD_LABEL_STYLE)
        return label

    def _update_field_availability(self, fmt: str):
        supports_quality, supports_16bit = format_capabilities(fmt)
        fmt = fmt.upper()

        self.bit_depth_box.setEnabled(supports_16bit)
        if not supports_16bit:
            self.bit_depth_box.setCurrentText("8-bit")
            self.bit_depth_hint.setText(f"{fmt} exports as 8-bit only.")
        else:
            self.bit_depth_hint.setText("16-bit is only written for a lossless PNG/TIFF export.")

        self.quality_label.setEnabled(supports_quality)
        self.quality_value_label.setEnabled(supports_quality)
        self.quality_slider.setEnabled(supports_quality)
        self.quality_hint.setText("" if supports_quality else f"{fmt} is always lossless - quality doesn't apply.")
