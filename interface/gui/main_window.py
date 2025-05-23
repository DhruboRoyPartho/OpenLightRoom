from PySide6.QtWidgets import QFileDialog, QMainWindow, QWidget, QHBoxLayout, QMenu, QMenuBar
from PySide6.QtGui import QAction
from interface.gui.image_viewer import ImageViewer
from interface.gui.controls_panel import ControlsPanel
from core.image_model.image_document import ImageDocument
from interface.gui.layer_stack_panel import LayerStackPanel
from interface.gui.import_export_dialog import ExportDialog, ImportDialog
from core.io.image_io import save_image
from interface.gui.exit_dialog import ExitDialog
import cv2
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Open LightRoom")

        # Optional Window size
        self.setGeometry(100, 50, 1080, 720)

        # Menubar setup
        menu_bar = QMenuBar(self)
        file_menu = QMenu("File", self)
        menu_bar.addMenu(file_menu)
        self.setMenuBar(menu_bar)

        # Import Action
        import_action = QAction("Import Image", self)
        import_action.triggered.connect(self.import_image)
        file_menu.addAction(import_action)

        # Export Action
        export_action = QAction("Export Image", self)
        export_action.triggered.connect(self.export_image)
        file_menu.addAction(export_action)

        # Exit Action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.exit_program)
        file_menu.addAction(exit_action)

        # Load Image
        self.load_image(None)
        # img = cv2.imread('tests/test.jpg')
        # if img is not None:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # else:
        #     img = np.zeros((500, 500, 3), dtype=np.uint8)
        # self.document = ImageDocument(img)

        # Setup GUI components
        central_widget = QWidget()
        layout = QHBoxLayout(central_widget)

        self.image_viewer = ImageViewer(self.document)
        # self.controls_panel = ControlsPanel(self.document, self.image_viewer)

        # Layer stack gui
        self.layer_stack_panel = LayerStackPanel(self.document, self.image_viewer)

        self.controls_panel = ControlsPanel(self.document, self.image_viewer, self.layer_stack_panel)

        # Fixed size of controls panel and stack
        self.controls_panel.setFixedWidth(350)
        self.layer_stack_panel.setFixedWidth(280)

        layout.addWidget(self.image_viewer)
        layout.addWidget(self.controls_panel)
        layout.addWidget(self.layer_stack_panel)

        self.setCentralWidget(central_widget)

    def exit_program(self):
        dialog = ExitDialog(self)

        dialog.yes_btn.clicked.connect(self.close)
        dialog.no_btn.clicked.connect(dialog.reject)
        dialog.exec()

    def import_image(self):
        dialog = ImportDialog(self)

        def on_import():
            path, _ = QFileDialog.getOpenFileNames(
                self,
                "Open Image",
                "",
                "Image Files (*.jpg *.jpeg *.png *.tif *.tiff *.bmp)"
            )
            print("here is path: ", path)
            
            if path:
                self.load_image(path[0])

            # Close dialog after import
            dialog.close()

        dialog.import_btn.clicked.connect(on_import)
        dialog.cancel_btn.clicked.connect(dialog.reject)
        dialog.exec()

    def load_image(self, path: str = None):
        print(f"path from load image: {path}")
        if path is None:
            img = np.zeros((500, 500, 3), dtype=np.uint8)
            self.document = ImageDocument(img)
            return
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.zeros((500, 500, 3), dtype=np.uint8)
            self.document = ImageDocument(img)
        except Exception as e:
            print(f"Error loading image {e}")
            return
        
        # Clear central widget and recreate UI components
        central_widget = self.centralWidget()
        layout = central_widget.layout()

        # Remove existing widgets if any
        for i in reversed(range(layout.count())): 
            layout.itemAt(i).widget().setParent(None)

        # New components with the new document
        self.image_viewer = ImageViewer(self.document)
        self.layer_stack_panel = LayerStackPanel(self.document, self.image_viewer)
        self.controls_panel = ControlsPanel(self.document, self.image_viewer, self.layer_stack_panel)

        # Fixed size of controls panel and stack
        self.controls_panel.setFixedWidth(350)
        self.layer_stack_panel.setFixedWidth(280)

        layout.addWidget(self.image_viewer)
        layout.addWidget(self.controls_panel)
        layout.addWidget(self.layer_stack_panel)

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