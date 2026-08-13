"""Edit > Preferences... - the small set of persisted, cross-session
preferences this app actually needs: a default preview render quality,
whether closing the app asks for confirmation, and default export
options so the Export dialog doesn't reset to the same values every
time. Changes only take effect on Save (see AppSettings) - Cancel
discards whatever was changed in this dialog.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QCheckBox, QSlider,
)
from PySide6.QtCore import Qt

from interface.gui.theme import TEXT, TEXT_DIM, BORDER
from interface.gui.app_settings import AppSettings
from interface.gui.canvas_toolbar import PREVIEW_QUALITY_OPTIONS
from interface.gui.import_export_dialog import format_capabilities, EXPORT_BUTTON_STYLE

SECTION_LABEL_STYLE = (
    f"color: {TEXT_DIM}; font-size: 10px; font-weight: 600; "
    f"border-bottom: 1px solid {BORDER}; padding-top: 8px; padding-bottom: 2px;"
)
FIELD_LABEL_STYLE = f"color: {TEXT_DIM}; font-size: 11px; font-weight: 600;"


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Preferences")
        self.setModal(True)
        self.setFixedWidth(360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(6)

        title = QLabel("Preferences")
        title.setStyleSheet(f"color: {TEXT}; font-size: 14px; font-weight: 600;")
        layout.addWidget(title)

        layout.addWidget(self._section_label("Preview"))
        layout.addWidget(self._field_label("Default Preview Quality"))
        self.preview_quality_box = QComboBox()
        for label, _ in PREVIEW_QUALITY_OPTIONS:
            self.preview_quality_box.addItem(label)
        layout.addWidget(self.preview_quality_box)
        hint = QLabel("Applied to newly opened images and this window's current preview.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px; font-style: italic;")
        layout.addWidget(hint)

        layout.addWidget(self._section_label("Behavior"))
        self.confirm_exit_check = QCheckBox("Confirm before exiting the app")
        layout.addWidget(self.confirm_exit_check)

        layout.addWidget(self._section_label("Export Defaults"))
        layout.addWidget(self._field_label("Format"))
        self.format_box = QComboBox()
        self.format_box.addItems(["JPG", "PNG", "TIFF", "WEBP"])
        layout.addWidget(self.format_box)

        layout.addWidget(self._field_label("Bit Depth"))
        self.bit_depth_box = QComboBox()
        self.bit_depth_box.addItems(["8-bit", "16-bit"])
        layout.addWidget(self.bit_depth_box)

        quality_row = QHBoxLayout()
        quality_row.addWidget(self._field_label("Quality"))
        quality_row.addStretch(1)
        self.quality_value_label = QLabel()
        self.quality_value_label.setStyleSheet(
            f"color: {TEXT_DIM}; font-family: Consolas, monospace; font-size: 11px;")
        quality_row.addWidget(self.quality_value_label)
        layout.addLayout(quality_row)
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(10, 100)
        layout.addWidget(self.quality_slider)

        layout.addSpacing(10)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.cancel_btn = QPushButton("Cancel")
        self.save_btn = QPushButton("Save")
        self.save_btn.setDefault(True)
        self.save_btn.setStyleSheet(EXPORT_BUTTON_STYLE)
        button_row.addWidget(self.cancel_btn)
        button_row.addWidget(self.save_btn)
        layout.addLayout(button_row)

        self.cancel_btn.clicked.connect(self.reject)
        self.save_btn.clicked.connect(self._save)
        self.format_box.currentTextChanged.connect(self._update_export_field_availability)
        self.quality_slider.valueChanged.connect(lambda v: self.quality_value_label.setText(str(v)))

        self._load_from_settings()

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setStyleSheet(SECTION_LABEL_STYLE)
        return label

    def _field_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(FIELD_LABEL_STYLE)
        return label

    def _load_from_settings(self):
        self.preview_quality_box.setCurrentText(self.settings.default_preview_quality_label())
        self.confirm_exit_check.setChecked(self.settings.confirm_before_exit())
        self.format_box.setCurrentText(self.settings.default_export_format())
        self.bit_depth_box.setCurrentText(self.settings.default_export_bit_depth())
        self.quality_slider.setValue(self.settings.default_export_quality())
        self.quality_value_label.setText(str(self.quality_slider.value()))
        self._update_export_field_availability(self.format_box.currentText())

    def _update_export_field_availability(self, fmt: str):
        supports_quality, supports_16bit = format_capabilities(fmt)
        self.bit_depth_box.setEnabled(supports_16bit)
        if not supports_16bit:
            self.bit_depth_box.setCurrentText("8-bit")
        self.quality_slider.setEnabled(supports_quality)
        self.quality_value_label.setEnabled(supports_quality)

    def _save(self):
        self.settings.set_default_preview_quality_label(self.preview_quality_box.currentText())
        self.settings.set_confirm_before_exit(self.confirm_exit_check.isChecked())
        self.settings.set_default_export_format(self.format_box.currentText())
        self.settings.set_default_export_bit_depth(self.bit_depth_box.currentText())
        self.settings.set_default_export_quality(self.quality_slider.value())
        self.accept()
