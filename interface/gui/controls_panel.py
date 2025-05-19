from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QSlider
from PySide6.QtCore import Qt
from core.adjustment_layers.brightness_layer import BrightnessLayer
from core.adjustment_layers.contrast_layer import ContrastLayer
from core.commands.add_layer_command import AddLayerCommand

class ControlsPanel(QWidget):
    def __init__(self, document, viewer, layer_stack_panel):
        super().__init__()
        self.document = document
        self.viewer = viewer
        self.layer_stack_panel = layer_stack_panel

        layout = QVBoxLayout(self)

        # Brightness
        layout.addWidget(QLabel("Brightness"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(-100)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        layout.addWidget(self.slider)

        self.apply_button = QPushButton("Apply Brightness")
        layout.addWidget(self.apply_button)

        # Contrast
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(10, 300)  # 10% to 300%
        self.contrast_slider.setValue(100)
        layout.addWidget(QLabel("Contrast"))
        layout.addWidget(self.contrast_slider)

        self.contrast_button = QPushButton("Apply Contrast")
        layout.addWidget(self.contrast_button)

        self.undo_button = QPushButton("Undo")
        layout.addWidget(self.undo_button)

        self.redo_button = QPushButton("Redo")
        layout.addWidget(self.redo_button)

        self.apply_button.clicked.connect(self.apply_brightness)
        self.contrast_button.clicked.connect(self.apply_contrast)
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)

    def apply_brightness(self):
        value = self.slider.value() / 100.0  # Convert to -1.0 to 1.0
        layer = BrightnessLayer(value)
        cmd = AddLayerCommand(self.document, layer)
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def apply_contrast(self):
        value = self.contrast_slider.value() / 100.0    # Convert to 0.1 - 3.0
        layer = ContrastLayer(value)
        cmd = AddLayerCommand(self.document, layer)
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def undo(self):
        self.document.undo()
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def redo(self):
        self.document.redo()
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

