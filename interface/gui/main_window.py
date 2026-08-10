from PySide6.QtWidgets import QFileDialog, QMainWindow, QSplitter, QMenu, QMenuBar, QMessageBox, QWidget, QVBoxLayout
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtCore import Qt
from interface.gui.image_viewer import ImageViewer, CanvasScrollArea
from interface.gui.canvas_toolbar import CanvasToolbar
from interface.gui.controls_panel import ControlsPanel
from core.image_model.image_document import ImageDocument
from interface.gui.layer_stack_panel import LayerStackPanel
from interface.gui.import_export_dialog import ExportDialog, ImportDialog
from core.io.image_io import save_image, load_and_linearize
from core.io.project_io import save_project as write_project_file, load_project as read_project_file
from interface.gui.exit_dialog import ExitDialog
from core.io.raw_io import is_raw_file, load_raw
from core.io.exif_io import read_exif, write_exif
import os
import numpy as np

IMPORT_FILE_FILTER = (
    "Image Files (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp "
    "*.cr2 *.cr3 *.nef *.arw *.dng *.raf *.orf *.rw2)"
)
EXPORT_FILE_FILTER = "Images (*.jpg *.jpeg *.png *.tif *.webp)"

PROJECT_FILE_FILTER = "Open LightRoom Project (*.olrproj)"
PROJECT_EXTENSION = ".olrproj"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window size
        self.setGeometry(80, 40, 1440, 880)
        self.setMinimumSize(1000, 640)

        # Menubar setup
        menu_bar = QMenuBar(self)
        file_menu = QMenu("File", self)
        menu_bar.addMenu(file_menu)
        self.setMenuBar(menu_bar)

        # Project actions
        open_project_action = QAction("Open Project...", self)
        open_project_action.setShortcut(QKeySequence("Ctrl+O"))
        open_project_action.triggered.connect(self.open_project)
        file_menu.addAction(open_project_action)

        save_project_action = QAction("Save Project", self)
        save_project_action.setShortcut(QKeySequence("Ctrl+S"))
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)

        save_project_as_action = QAction("Save Project As...", self)
        save_project_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_project_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(save_project_as_action)

        file_menu.addSeparator()

        # Import Action
        import_action = QAction("Import Image...", self)
        import_action.setShortcut(QKeySequence("Ctrl+I"))
        import_action.triggered.connect(self.import_image)
        file_menu.addAction(import_action)

        # Export Action
        export_action = QAction("Export Image...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self.export_image)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        # Exit Action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.exit_program)
        file_menu.addAction(exit_action)

        # Central layout: [ Layers panel | Canvas | Develop controls panel ],
        # resizable via drag handles like Lightroom's Develop module.
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(1)
        self.splitter.setChildrenCollapsible(False)
        self.setCentralWidget(self.splitter)

        self.document = None
        self.image_viewer = None
        self.canvas_toolbar = None
        self.layer_stack_panel = None
        self.controls_panel = None

        # Path of the source raster image currently being edited, and of the
        # .olrproj sidecar it was last saved to/opened from (if any).
        self.current_image_path = None
        self.current_project_path = None

        # Lightroom-style before/after toggle, available regardless of which
        # widget currently has focus.
        self._before_after_shortcut = QShortcut(QKeySequence("\\"), self)
        self._before_after_shortcut.activated.connect(lambda: self.canvas_toolbar.toggle_before_after())

        self.load_image(None)

    def exit_program(self):
        dialog = ExitDialog(self)

        dialog.yes_btn.clicked.connect(self.close)
        dialog.no_btn.clicked.connect(dialog.reject)
        dialog.exec()

    def import_image(self):
        dialog = ImportDialog(self)

        def on_import():
            path, _ = QFileDialog.getOpenFileNames(self, "Open Image", "", IMPORT_FILE_FILTER)

            if path:
                self.load_image(path[0])

            # Close dialog after import
            dialog.close()

        dialog.import_btn.clicked.connect(on_import)
        dialog.cancel_btn.clicked.connect(dialog.reject)
        dialog.exec()

    def load_image(self, path: str = None):
        """Decodes a RAW or standard image file into this app's
        scene-linear float32 working space (real LibRaw demosaicing for
        RAW; sRGB-decode for everything else, converting from an embedded
        ICC profile first if the source has one other than sRGB)."""
        if path is None:
            img = np.zeros((500, 500, 3), dtype=np.float32)
            image_path = None
            exif_data = {}
        else:
            try:
                img = load_raw(path) if is_raw_file(path) else load_and_linearize(path)
                exif_data = read_exif(path)
            except Exception as e:
                QMessageBox.critical(self, "Import Image", f"Could not open this image:\n{e}")
                return
            image_path = path

        document = ImageDocument(img)
        document.exif_data = exif_data
        self._set_document(document, image_path=image_path, project_path=None)

    def _set_document(self, document, image_path, project_path):
        """Swap in a document (freshly imported, or restored from a saved
        project) and rebuild the panels bound to it."""
        self.document = document
        self.current_image_path = image_path
        self.current_project_path = project_path

        # Drop the previous document's panels, if any, before building fresh
        # ones bound to the new document.
        while self.splitter.count():
            widget = self.splitter.widget(0)
            widget.setParent(None)
            widget.deleteLater()

        self.image_viewer = ImageViewer(self.document)
        self.layer_stack_panel = LayerStackPanel(self.document, self.image_viewer)
        self.controls_panel = ControlsPanel(self.document, self.image_viewer, self.layer_stack_panel)
        # The document may already carry layers restored from a saved
        # project; reflect their values in the freshly-built sliders.
        self.controls_panel.sync_from_document()

        self.canvas_toolbar = CanvasToolbar(self.document, self.image_viewer, self.layer_stack_panel)
        canvas_scroll = CanvasScrollArea(self.image_viewer)

        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)
        canvas_layout.addWidget(self.canvas_toolbar)
        canvas_layout.addWidget(canvas_scroll, 1)

        self.layer_stack_panel.setMinimumWidth(200)
        self.layer_stack_panel.setMaximumWidth(360)
        self.controls_panel.setMinimumWidth(300)
        self.controls_panel.setMaximumWidth(420)

        self.splitter.addWidget(self.layer_stack_panel)
        self.splitter.addWidget(canvas_container)
        self.splitter.addWidget(self.controls_panel)

        # Canvas (index 1) takes all extra space; side panels stay fixed
        # unless the user drags a handle.
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([260, 820, 360])

        self._update_window_title()

    def _update_window_title(self):
        if self.current_project_path:
            name = os.path.basename(self.current_project_path)
        elif self.current_image_path:
            name = os.path.basename(self.current_image_path)
        else:
            name = "Untitled"
        self.setWindowTitle(f"Open LightRoom - {name}")

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", PROJECT_FILE_FILTER)
        if not path:
            return

        try:
            image_path, document = read_project_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Open Project", f"Could not open project:\n{e}")
            return

        self._set_document(document, image_path=image_path, project_path=path)

    def save_project(self):
        if self.current_project_path is None:
            self.save_project_as()
            return
        self._write_project(self.current_project_path)

    def save_project_as(self):
        if self.current_image_path is None:
            QMessageBox.warning(self, "Save Project", "Import an image before saving a project.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Project As", "", PROJECT_FILE_FILTER)
        if not path:
            return
        if not path.lower().endswith(PROJECT_EXTENSION):
            path += PROJECT_EXTENSION

        self._write_project(path)

    def _write_project(self, path):
        try:
            write_project_file(path, self.current_image_path, self.document)
        except Exception as e:
            QMessageBox.critical(self, "Save Project", f"Could not save project:\n{e}")
            return

        self.current_project_path = path
        self._update_window_title()

    def export_image(self):
        dialog = ExportDialog(self)

        def on_export():
            path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", EXPORT_FILE_FILTER)
            if not path:
                return
            format = dialog.format_box.currentText()
            quality = dialog.quality_slider.value()
            bit_depth = 16 if dialog.bit_depth_box.currentText() == "16-bit" else 8
            image = self.document.render()
            try:
                actual_path = save_image(path, image, format=format, quality=quality, bit_depth=bit_depth)
                write_exif(actual_path, self.document.exif_data)
            except Exception as e:
                QMessageBox.critical(self, "Export Image", f"Could not export image:\n{e}")
                return
            dialog.accept()

        dialog.export_btn.clicked.connect(on_export)
        dialog.exec()
