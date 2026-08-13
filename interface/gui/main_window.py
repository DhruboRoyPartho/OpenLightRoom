from PySide6.QtWidgets import (
    QFileDialog, QMainWindow, QSplitter, QMenu, QMenuBar, QMessageBox, QWidget, QVBoxLayout, QDialog, QApplication,
)
from PySide6.QtGui import QAction, QKeySequence, QIcon, QDesktopServices
from PySide6.QtCore import Qt, QUrl
from interface.gui.assets import LOGO_PATH
from interface.gui.app_info import APP_NAME, GITHUB_URL, GITHUB_ISSUES_URL
from interface.gui.app_settings import AppSettings
from interface.gui.about_dialog import AboutDialog
from interface.gui.settings_dialog import SettingsDialog
from interface.gui.image_viewer import ImageViewer, CanvasScrollArea
from interface.gui.canvas_toolbar import CanvasToolbar, PREVIEW_QUALITY_OPTIONS
from interface.gui.controls_panel import ControlsPanel
from core.image_model.image_document import ImageDocument
from interface.gui.layer_stack_panel import LayerStackPanel
from interface.gui.import_export_dialog import ExportDialog
from core.io.image_io import save_image, load_and_linearize
from core.io.project_io import save_project as write_project_file, load_project as read_project_file
from interface.gui.exit_dialog import ExitDialog
from interface.gui.confirm_dialog import ConfirmDialog
from interface.gui.busy_tracker import BusyTracker
from interface.gui.loading_indicator import LoadingIndicator
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

RENDER_BUSY_LABEL = "Rendering..."

class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings = None):
        super().__init__()

        self.settings = settings if settings is not None else AppSettings()

        self.setWindowIcon(QIcon(LOGO_PATH))

        # Window size
        self.setGeometry(80, 40, 1440, 880)
        self.setMinimumSize(1000, 640)

        self._build_menu_bar()

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

        # A small, unobtrusive "working" indicator that lives in the status
        # bar - the one part of the window that's otherwise always empty -
        # so a slow render, import, export, or project save/load is
        # visible without ever covering or displacing a control.
        # busy_tracker.attach()'d for the frequent, often-instant
        # render-queue traffic (debounced so a normal tool tweak never
        # flashes it); the one-shot blocking actions below (import/export/
        # save/open) call show_now()/hide_now() directly instead.
        self.busy_tracker = BusyTracker()
        self.loading_indicator = LoadingIndicator()
        self.loading_indicator.attach(self.busy_tracker)
        status_bar = self.statusBar()
        status_bar.setSizeGripEnabled(False)
        status_bar.addPermanentWidget(self.loading_indicator)

        self.load_image(None)

    # --- menu bar ---------------------------------------------------------

    def _build_menu_bar(self):
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        # --- File ---------------------------------------------------------
        file_menu = QMenu("File", self)
        menu_bar.addMenu(file_menu)

        open_project_action = QAction("Open Project...", self)
        open_project_action.setShortcut(QKeySequence("Ctrl+O"))
        open_project_action.triggered.connect(self.open_project)
        file_menu.addAction(open_project_action)

        self.recent_projects_menu = QMenu("Open Recent", self)
        file_menu.addMenu(self.recent_projects_menu)

        save_project_action = QAction("Save Project", self)
        save_project_action.setShortcut(QKeySequence("Ctrl+S"))
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)

        save_project_as_action = QAction("Save Project As...", self)
        save_project_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_project_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(save_project_as_action)

        file_menu.addSeparator()

        import_action = QAction("Import Image...", self)
        import_action.setShortcut(QKeySequence("Ctrl+I"))
        import_action.triggered.connect(self.import_image)
        file_menu.addAction(import_action)

        export_action = QAction("Export Image...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self.export_image)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        close_project_action = QAction("Close Project", self)
        close_project_action.setShortcut(QKeySequence("Ctrl+W"))
        close_project_action.triggered.connect(self.close_project)
        file_menu.addAction(close_project_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.exit_program)
        file_menu.addAction(exit_action)

        self._rebuild_recent_projects_menu()

        # --- Edit -----------------------------------------------------------
        edit_menu = QMenu("Edit", self)
        menu_bar.addMenu(edit_menu)

        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        self.undo_action.triggered.connect(lambda: self.controls_panel.undo())
        edit_menu.addAction(self.undo_action)

        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut(QKeySequence("Ctrl+Shift+Z"))
        self.redo_action.triggered.connect(lambda: self.controls_panel.redo())
        edit_menu.addAction(self.redo_action)

        edit_menu.addSeparator()

        preferences_action = QAction("Preferences...", self)
        preferences_action.setShortcut(QKeySequence("Ctrl+,"))
        preferences_action.triggered.connect(self.open_preferences)
        edit_menu.addAction(preferences_action)

        edit_menu.aboutToShow.connect(self._sync_edit_menu_state)

        # --- View -----------------------------------------------------------
        view_menu = QMenu("View", self)
        menu_bar.addMenu(view_menu)

        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut(QKeySequence.ZoomIn)
        zoom_in_action.triggered.connect(lambda: self.image_viewer.zoom_in())
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut(QKeySequence.ZoomOut)
        zoom_out_action.triggered.connect(lambda: self.image_viewer.zoom_out())
        view_menu.addAction(zoom_out_action)

        view_menu.addSeparator()

        fit_action = QAction("Fit to Window", self)
        fit_action.setShortcut(QKeySequence("Ctrl+0"))
        fit_action.triggered.connect(lambda: self.image_viewer.set_fit())
        view_menu.addAction(fit_action)

        actual_size_action = QAction("Actual Size (100%)", self)
        actual_size_action.setShortcut(QKeySequence("Ctrl+1"))
        actual_size_action.triggered.connect(lambda: self.image_viewer.set_actual_size())
        view_menu.addAction(actual_size_action)

        view_menu.addSeparator()

        # The sole binding for "\\" - previously also lived on a standalone
        # QShortcut, which would fire alongside this action's own shortcut
        # and trip Qt's "ambiguous shortcut" resolution since both share the
        # same window-level context.
        before_after_action = QAction("Toggle Before / After", self)
        before_after_action.setShortcut(QKeySequence("\\"))
        before_after_action.triggered.connect(lambda: self.canvas_toolbar.toggle_before_after())
        view_menu.addAction(before_after_action)

        # --- Help -----------------------------------------------------------
        help_menu = QMenu("Help", self)
        menu_bar.addMenu(help_menu)

        about_action = QAction(f"About {APP_NAME}...", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        help_menu.addSeparator()

        github_action = QAction("View on GitHub", self)
        github_action.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(GITHUB_URL)))
        help_menu.addAction(github_action)

        report_issue_action = QAction("Report an Issue...", self)
        report_issue_action.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(GITHUB_ISSUES_URL)))
        help_menu.addAction(report_issue_action)

    def _sync_edit_menu_state(self):
        doc = self.document
        self.undo_action.setEnabled(bool(doc and doc.history))
        self.redo_action.setEnabled(bool(doc and doc.redo_stack))

    def show_about_dialog(self):
        AboutDialog(self).exec()

    def open_preferences(self):
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec() == QDialog.Accepted:
            self._apply_settings()

    def _apply_settings(self):
        # Reflects the new default preview quality in the currently open
        # viewer right away, exactly as if it had been picked from the
        # canvas toolbar's own dropdown - not just "starting next launch".
        # Setting the combo box's text (rather than calling
        # viewer.set_preview_quality directly) reuses that existing
        # wiring so the toolbar's displayed selection and the actual
        # applied quality never disagree.
        if self.canvas_toolbar is not None:
            self.canvas_toolbar.preview_quality_combo.setCurrentText(
                self.settings.default_preview_quality_label())

    # --- recent projects ----------------------------------------------------

    def _rebuild_recent_projects_menu(self):
        menu = self.recent_projects_menu
        menu.clear()
        paths = self.settings.recent_projects()

        if not paths:
            empty_action = QAction("No Recent Projects", self)
            empty_action.setEnabled(False)
            menu.addAction(empty_action)
            return

        for path in paths:
            action = QAction(os.path.basename(path), self)
            action.setToolTip(path)
            action.setStatusTip(path)
            action.triggered.connect(lambda checked=False, p=path: self._open_recent_project(p))
            menu.addAction(action)

        menu.addSeparator()
        clear_action = QAction("Clear Recent Projects", self)
        clear_action.triggered.connect(self._clear_recent_projects)
        menu.addAction(clear_action)

    def _open_recent_project(self, path):
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Open Recent", f"This project can no longer be found:\n{path}")
            self.settings.remove_recent_project(path)
            self._rebuild_recent_projects_menu()
            return
        self._load_project_file(path)

    def _clear_recent_projects(self):
        self.settings.clear_recent_projects()
        self._rebuild_recent_projects_menu()

    def closeEvent(self, event):
        if not self.settings.confirm_before_exit():
            event.accept()
            return
        dialog = ExitDialog(self)
        if dialog.exec() == QDialog.Accepted:
            event.accept()
        else:
            event.ignore()

    def exit_program(self):
        self.close()  # routed through closeEvent() above for one consistent confirmation

    def close_project(self):
        if self.current_image_path is None and self.current_project_path is None:
            return  # nothing open - nothing to close
        dialog = ConfirmDialog(
            "Close Project",
            "Close the current project? Any unsaved changes will be lost.",
            confirm_text="Close Project",
            parent=self,
        )
        if dialog.exec() == QDialog.Accepted:
            self.load_image(None)

    def import_image(self):
        path, _ = QFileDialog.getOpenFileNames(self, "Import Image", "", IMPORT_FILE_FILTER)
        if path:
            self.load_image(path[0])

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
            self.loading_indicator.show_now(f"Importing {os.path.basename(path)}...")
            QApplication.processEvents()
            try:
                img = load_raw(path) if is_raw_file(path) else load_and_linearize(path)
                exif_data = read_exif(path)
            except Exception as e:
                QMessageBox.critical(self, "Import Image", f"Could not open this image:\n{e}")
                return
            finally:
                self.loading_indicator.hide_now()
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
        # Each document gets its own ImageViewer/RenderQueue; bridge this
        # one's render lifecycle into the shared busy tracker so the status
        # bar indicator reflects whichever document is currently active.
        self.image_viewer.render_queue.render_started.connect(
            lambda: self.busy_tracker.begin(RENDER_BUSY_LABEL)
        )
        self.image_viewer.render_queue.image_rendered.connect(
            lambda _img: self.busy_tracker.end(RENDER_BUSY_LABEL)
        )
        self.layer_stack_panel = LayerStackPanel(self.document, self.image_viewer)
        self.controls_panel = ControlsPanel(self.document, self.image_viewer, self.layer_stack_panel)
        # The document may already carry layers restored from a saved
        # project; reflect their values in the freshly-built sliders.
        self.controls_panel.sync_from_document()

        self.canvas_toolbar = CanvasToolbar(self.document, self.image_viewer, self.layer_stack_panel)
        # New/reopened documents start at the user's saved default preview
        # quality rather than always resetting to Full Quality.
        self.canvas_toolbar.preview_quality_combo.setCurrentText(
            self.settings.default_preview_quality_label())
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
        self._load_project_file(path)

    def _load_project_file(self, path):
        self.loading_indicator.show_now(f"Opening {os.path.basename(path)}...")
        QApplication.processEvents()
        try:
            image_path, document = read_project_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Open Project", f"Could not open project:\n{e}")
            return
        finally:
            self.loading_indicator.hide_now()

        self._set_document(document, image_path=image_path, project_path=path)
        self.settings.add_recent_project(path)
        self._rebuild_recent_projects_menu()

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
        self.loading_indicator.show_now(f"Saving {os.path.basename(path)}...")
        QApplication.processEvents()
        try:
            write_project_file(path, self.current_image_path, self.document)
        except Exception as e:
            QMessageBox.critical(self, "Save Project", f"Could not save project:\n{e}")
            return
        finally:
            self.loading_indicator.hide_now()

        self.current_project_path = path
        self._update_window_title()
        self.settings.add_recent_project(path)
        self._rebuild_recent_projects_menu()

    def export_image(self):
        dialog = ExportDialog(
            self,
            default_format=self.settings.default_export_format(),
            default_quality=self.settings.default_export_quality(),
            default_bit_depth=self.settings.default_export_bit_depth(),
        )

        def on_export():
            path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", EXPORT_FILE_FILTER)
            if not path:
                return
            format = dialog.format_box.currentText()
            quality = dialog.quality_slider.value()
            bit_depth = 16 if dialog.bit_depth_box.currentText() == "16-bit" else 8

            self.loading_indicator.show_now(f"Exporting {os.path.basename(path)}...")
            QApplication.processEvents()
            try:
                image = self.document.render()
                actual_path = save_image(path, image, format=format, quality=quality, bit_depth=bit_depth)
                write_exif(actual_path, self.document.exif_data)
            except Exception as e:
                QMessageBox.critical(self, "Export Image", f"Could not export image:\n{e}")
                return
            finally:
                self.loading_indicator.hide_now()
            dialog.accept()

        dialog.export_btn.clicked.connect(on_export)
        dialog.exec()
