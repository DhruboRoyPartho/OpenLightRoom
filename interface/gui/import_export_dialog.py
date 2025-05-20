from PySide6.QtWidgets import QFileDialog, QDialog, QVBoxLayout, QLabel, QPushButton, QSlider, QComboBox, QHBoxLayout
from PySide6.QtCore import Qt

class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Image")
        
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("Export Format"))
        self.format_box = QComboBox()
        self.format_box.addItems(["JPG", "PNG", "TIFF"])
        layout.addWidget(self.format_box)

        layout.addWidget(QLabel("Quality (for JPEG)"))
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(10, 100)
        self.quality_slider.setValue(95)
        layout.addWidget(self.quality_slider)
        
        self.export_btn = QPushButton("Export")
        layout.addWidget(self.export_btn)