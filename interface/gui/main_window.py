from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout
from interface.gui.image_viewer import ImageViewer
from interface.gui.controls_panel import ControlsPanel
from core.image_model.image_document import ImageDocument
from interface.gui.layer_stack_panel import LayerStackPanel
import cv2

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Open LighRoom")

        # Load Image
        img = cv2.imread('tests/test.jpg')
        self.document = ImageDocument(img)

        # Setup GUI components
        central_widget = QWidget()
        layout = QHBoxLayout(central_widget)

        self.image_viewer = ImageViewer(self.document)
        # self.controls_panel = ControlsPanel(self.document, self.image_viewer)

        # Layer stack gui
        self.layer_stack_panel = LayerStackPanel(self.document, self.image_viewer)

        self.controls_panel = ControlsPanel(self.document, self.image_viewer, self.layer_stack_panel)

        layout.addWidget(self.image_viewer)
        layout.addWidget(self.controls_panel)
        layout.addWidget(self.layer_stack_panel)

        self.setCentralWidget(central_widget)