from PySide6.QtWidgets import QFileDialog, QMainWindow, QWidget, QHBoxLayout, QMenu, QMenuBar
from PySide6.QtGui import QAction
from interface.gui.image_viewer import ImageViewer
from interface.gui.controls_panel import ControlsPanel
from core.image_model.image_document import ImageDocument
from interface.gui.layer_stack_panel import LayerStackPanel
from interface.gui.import_export_dialog import ExportDialog
from core.io.image_io import save_image
import cv2
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Open LighRoom")

        # Menubar setup
        menu_bar = QMenuBar(self)
        file_menu = QMenu("File", self)
        menu_bar.addMenu(file_menu)
        self.setMenuBar(menu_bar)

        export_action = QAction("Export Image", self)
        export_action.triggered.connect(self.export_image)
        file_menu.addAction(export_action)

        # Load Image
        img = cv2.imread('tests/test.jpg')
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((500, 500, 3), dtype=np.uint8)
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

    def export_image(self):
        dialog = ExportDialog(self)

        def on_export():
            path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.jpg *.jpeg *.png *.tif)")
            if not path:
                return
            format = dialog.format_box.currentText()
            quality = dialog.quality_slider.value()
            image = self.document.render()
            if not path.lower().endswith(f".{format.lower()}"):
                path += f".{format.lower()}"
            save_image(path, image, format=format, quality=quality)
            dialog.accept()
        
        dialog.export_btn.clicked.connect(on_export)
        dialog.exec()