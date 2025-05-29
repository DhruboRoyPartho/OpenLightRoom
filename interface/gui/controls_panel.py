from PySide6.QtWidgets import QWidget, QSpinBox, QDoubleSpinBox, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QSlider
from PySide6.QtCore import Qt
from core.adjustment_layers.brightness_layer import BrightnessLayer
from core.adjustment_layers.contrast_layer import ContrastLayer
from core.adjustment_layers.temperature_layer import TemperatureLayer
from core.adjustment_layers.tint_layer import TintLayer
from core.adjustment_layers.exposure_layer import ExposureLayer
from core.adjustment_layers.highlights_layer import HighlightsLayer
from core.adjustment_layers.shadows_layer import ShadowsLayer
from core.adjustment_layers.whites_layer import WhitesLayer
from core.adjustment_layers.blacks_layer import BlacksLayer
from core.commands.add_layer_command import AddLayerCommand

class ControlsPanel(QWidget):
    def __init__(self, document, viewer, layer_stack_panel):
        super().__init__()
        self.document = document
        self.viewer = viewer
        self.layer_stack_panel = layer_stack_panel
        self.control_label_width = 80
        self.slider_width = 170
        self.spinbox_width = 70

        # control_widget = QWidget()

        layout = QVBoxLayout(self)

        # control_widget.setLayout(layout)

        # control_widget.setFixedWidth(300)

        # Brightness
        brightness_layout = QHBoxLayout()
        brightness_label = QLabel("Brightness")
        brightness_label.setFixedWidth(self.control_label_width)
        brightness_layout.addWidget(brightness_label)
        
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setFixedWidth(self.slider_width)

        self.brightness_spinbox = QSpinBox()
        self.brightness_spinbox.setRange(-100, 100)
        self.brightness_spinbox.setValue(0)
        self.brightness_spinbox.setFixedWidth(self.spinbox_width)

        brightness_layout.addWidget(self.brightness_slider)
        brightness_layout.addWidget(self.brightness_spinbox)
        layout.addLayout(brightness_layout)

        # self.apply_button = QPushButton("Apply Brightness")
        # layout.addWidget(self.apply_button)

        # Contrast
        contrast_layout = QHBoxLayout()
        contrast_label = QLabel("Contrast")
        contrast_label.setFixedWidth(self.control_label_width)
        contrast_layout.addWidget(contrast_label)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(10, 300)  # 10% to 300%
        self.contrast_slider.setValue(100)
        self.contrast_slider.setFixedWidth(self.slider_width)

        self.contrast_spinbox = QSpinBox()
        self.contrast_spinbox.setRange(10, 300)
        self.contrast_spinbox.setValue(100)
        self.contrast_spinbox.setFixedWidth(self.spinbox_width)

        contrast_layout.addWidget(self.contrast_slider)
        contrast_layout.addWidget(self.contrast_spinbox)
        layout.addLayout(contrast_layout)

        # self.contrast_button = QPushButton("Apply Contrast")
        # layout.addWidget(self.contrast_button)

        # White balance
        # temperature
        temp_layout = QHBoxLayout()
        temp_label = QLabel("Temperature")
        temp_label.setFixedWidth(self.control_label_width)
        temp_layout.addWidget(temp_label)

        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(-100, 100)
        self.temp_slider.setValue(0)
        self.temp_slider.setFixedWidth(self.slider_width)

        self.temp_spinbox = QSpinBox()
        self.temp_spinbox.setRange(-100, 100)
        self.temp_spinbox.setValue(0)
        self.temp_spinbox.setFixedWidth(self.spinbox_width)

        temp_layout.addWidget(self.temp_slider)
        temp_layout.addWidget(self.temp_spinbox)
        layout.addLayout(temp_layout)

        # tint
        tint_layout = QHBoxLayout()
        tint_label = QLabel("Tint")
        tint_label.setFixedWidth(self.control_label_width)
        tint_layout.addWidget(tint_label)

        self.tint_slider = QSlider(Qt.Horizontal)
        self.tint_slider.setRange(-100, 100)
        self.tint_slider.setValue(0)
        self.tint_slider.setFixedWidth(self.slider_width)

        self.tint_spinbox = QSpinBox()
        self.tint_spinbox.setRange(-100, 100)
        self.tint_spinbox.setValue(0)
        self.tint_spinbox.setFixedWidth(self.spinbox_width)

        tint_layout.addWidget(self.tint_slider)
        tint_layout.addWidget(self.tint_spinbox)
        layout.addLayout(tint_layout)

        # Exposure
        exposure_layout = QHBoxLayout()
        exposure_label = QLabel("Exposure")
        exposure_label.setFixedWidth(self.control_label_width)
        exposure_layout.addWidget(exposure_label)

        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(-100, 100)
        self.exposure_slider.setValue(0)
        self.exposure_slider.setFixedWidth(self.slider_width)

        self.exposure_spinbox = QSpinBox()
        self.exposure_spinbox.setRange(-100, 100)
        self.exposure_spinbox.setValue(0)
        self.exposure_spinbox.setFixedWidth(self.spinbox_width)

        exposure_layout.addWidget(self.exposure_slider)
        exposure_layout.addWidget(self.exposure_spinbox)
        layout.addLayout(exposure_layout)

        # Highlights
        highlights_layout = QHBoxLayout()
        highlights_label = QLabel("Highlights")
        highlights_label.setFixedWidth(self.control_label_width)
        highlights_layout.addWidget(highlights_label)

        self.highlights_slider = QSlider(Qt.Horizontal)
        self.highlights_slider.setRange(-100, 100)
        self.highlights_slider.setValue(0)
        self.highlights_slider.setFixedWidth(self.slider_width)

        self.highlights_spinbox = QSpinBox()
        self.highlights_spinbox.setRange(-100, 100)
        self.highlights_spinbox.setValue(0)
        self.highlights_spinbox.setFixedWidth(self.spinbox_width)

        highlights_layout.addWidget(self.highlights_slider)
        highlights_layout.addWidget(self.highlights_spinbox)
        layout.addLayout(highlights_layout)

        # Shadows
        shadows_layout = QHBoxLayout()
        shadows_label = QLabel("Shadows")
        shadows_label.setFixedWidth(self.control_label_width)
        shadows_layout.addWidget(shadows_label)

        self.shadows_slider = QSlider(Qt.Horizontal)
        self.shadows_slider.setRange(-100, 100)
        self.shadows_slider.setValue(0)
        self.shadows_slider.setFixedWidth(self.slider_width)

        self.shadows_spinbox = QSpinBox()
        self.shadows_spinbox.setRange(-100, 100)
        self.shadows_spinbox.setValue(0)
        self.shadows_spinbox.setFixedWidth(self.spinbox_width)

        shadows_layout.addWidget(self.shadows_slider)
        shadows_layout.addWidget(self.shadows_spinbox)
        layout.addLayout(shadows_layout)

        # Whites
        whites_layout = QHBoxLayout()
        whites_label = QLabel("Whites")
        whites_label.setFixedWidth(self.control_label_width)
        whites_layout.addWidget(whites_label)

        self.whites_slider = QSlider(Qt.Horizontal)
        self.whites_slider.setRange(-100, 100)
        self.whites_slider.setValue(0)
        self.whites_slider.setFixedWidth(self.slider_width)

        self.whites_spinbox = QSpinBox()
        self.whites_spinbox.setRange(-100, 100)
        self.whites_spinbox.setValue(0)
        self.whites_spinbox.setFixedWidth(self.spinbox_width)

        whites_layout.addWidget(self.whites_slider)
        whites_layout.addWidget(self.whites_spinbox)
        layout.addLayout(whites_layout)

        # Blacks
        blacks_layout = QHBoxLayout()
        blacks_label = QLabel("Blacks")
        blacks_label.setFixedWidth(self.control_label_width)
        blacks_layout.addWidget(blacks_label)

        self.blacks_slider = QSlider(Qt.Horizontal)
        self.blacks_slider.setRange(-100, 100)
        self.blacks_slider.setValue(0)
        self.blacks_slider.setFixedWidth(self.slider_width)

        self.blacks_spinbox = QSpinBox()
        self.blacks_spinbox.setRange(-100, 100)
        self.blacks_spinbox.setValue(0)
        self.blacks_spinbox.setFixedWidth(self.spinbox_width)

        blacks_layout.addWidget(self.blacks_slider)
        blacks_layout.addWidget(self.blacks_spinbox)
        layout.addLayout(blacks_layout)


        self.delete_layer_button = QPushButton("Delete")
        # layout.addWidget(self.delete_layer_button)

        self.undo_button = QPushButton("Undo")
        layout.addWidget(self.undo_button)

        self.redo_button = QPushButton("Redo")
        layout.addWidget(self.redo_button)

        # Slider value change and apply to image
        self.brightness_slider.valueChanged.connect(self.apply_brightness)
        self.contrast_slider.valueChanged.connect(self.apply_contrast)
        self.temp_slider.valueChanged.connect(self.apply_temperature)
        self.tint_slider.valueChanged.connect(self.apply_tint)
        self.exposure_slider.valueChanged.connect(self.apply_exposure)
        self.highlights_slider.valueChanged.connect(self.apply_highlights)
        self.shadows_slider.valueChanged.connect(self.apply_shadows)
        self.whites_slider.valueChanged.connect(self.apply_whites)
        self.blacks_slider.valueChanged.connect(self.apply_blacks)

        # self.apply_button.clicked.connect(self.apply_brightness)
        # self.contrast_button.clicked.connect(self.apply_contrast)
        self.delete_layer_button.clicked.connect(self.delete_layer)
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)

        # Slider and spinbox connection
        self.brightness_slider.valueChanged.connect(self.brightness_spinbox.setValue)
        self.brightness_spinbox.valueChanged.connect(self.brightness_slider.setValue)

        self.contrast_slider.valueChanged.connect(self.contrast_spinbox.setValue)
        self.contrast_spinbox.valueChanged.connect(self.contrast_slider.setValue)    

        self.temp_slider.valueChanged.connect(self.temp_spinbox.setValue)
        self.temp_spinbox.valueChanged.connect(self.temp_slider.setValue)

        self.tint_slider.valueChanged.connect(self.tint_spinbox.setValue)
        self.tint_spinbox.valueChanged.connect(self.tint_slider.setValue)

        self.exposure_slider.valueChanged.connect(self.exposure_spinbox.setValue)
        self.exposure_spinbox.valueChanged.connect(self.exposure_slider.setValue)

        self.highlights_slider.valueChanged.connect(self.highlights_spinbox.setValue)
        self.highlights_spinbox.valueChanged.connect(self.highlights_slider.setValue)   

        self.shadows_slider.valueChanged.connect(self.shadows_spinbox.setValue)
        self.shadows_spinbox.valueChanged.connect(self.shadows_slider.setValue)

        self.whites_slider.valueChanged.connect(self.whites_spinbox.setValue)
        self.whites_spinbox.valueChanged.connect(self.whites_slider.setValue) 

        self.blacks_slider.valueChanged.connect(self.blacks_spinbox.setValue)
        self.blacks_spinbox.valueChanged.connect(self.blacks_slider.setValue)    

    def apply_brightness(self):
        value = self.brightness_slider.value() / 100.0  # Convert to -1.0 to 1.0
        layer = BrightnessLayer(value)
        cmd = AddLayerCommand(self.document, layer, "Brightness")
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def apply_contrast(self):
        value = self.contrast_slider.value() / 100.0    # Convert to 0.1 - 3.0
        layer = ContrastLayer(value)
        cmd = AddLayerCommand(self.document, layer, "Contrast")
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def apply_temperature(self):
        value = self.temp_slider.value()
        layer = TemperatureLayer(value)
        cmd = AddLayerCommand(self.document, layer, "Temperature")
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def apply_tint(self):
        value = self.tint_slider.value()
        layer = TintLayer(value)
        cmd = AddLayerCommand(self.document, layer, "Tint")
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def apply_exposure(self):
        value = self.exposure_slider.value()
        layer = ExposureLayer(value)
        cmd = AddLayerCommand(self.document, layer, "Exposure")
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def apply_highlights(self):
        value = self.highlights_slider.value()
        layer = HighlightsLayer(value)
        cmd = AddLayerCommand(self.document, layer, "Highlights")
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def apply_shadows(self):
        value = self.shadows_slider.value()
        layer = ShadowsLayer(value)
        cmd = AddLayerCommand(self.document, layer, "Shadows")
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def apply_whites(self):
        value = self.whites_slider.value()
        layer = WhitesLayer(value)
        cmd = AddLayerCommand(self.document, layer, "Whites")
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()
        
    def apply_blacks(self):
        value = self.blacks_slider.value()
        layer = BlacksLayer(value)
        cmd = AddLayerCommand(self.document, layer, "Blacks")
        self.document.execute_command(cmd)
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def delete_layer(self):
        pass

    def undo(self):
        self.document.undo()
        self.viewer.update_view()
        self.layer_stack_panel.refresh()

    def redo(self):
        self.document.redo()
        self.viewer.update_view()
        self.layer_stack_panel.refresh()