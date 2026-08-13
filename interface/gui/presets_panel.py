from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, QInputDialog,
    QFileDialog, QMessageBox, QAbstractItemView,
)
from PySide6.QtCore import Qt

from core.io import preset_io
from core.commands.change_layer_command import ChangeLayerCommand
from core.commands.composite_command import CompositeCommand


class PresetsPanel(QWidget):
    """Save/load/delete/duplicate/import/export named presets - reusable
    "looks" built from the current adjustment layers (geometry excluded,
    see preset_io._EXCLUDED_FROM_PRESETS), stored as plain JSON files
    independent of any specific project. Applying a preset replaces
    whichever of its layer types the document already has, as one
    CompositeCommand undo step, so it composes with the rest of the app's
    undo history like any other edit rather than being a separate,
    unreversible action.
    """

    def __init__(self, document, viewer, layer_stack_panel, on_layers_changed=None):
        super().__init__()
        self.document = document
        self.viewer = viewer
        self.layer_stack_panel = layer_stack_panel
        self.on_layers_changed = on_layers_changed

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.setFixedHeight(110)
        layout.addWidget(self.list_widget)

        row1 = QHBoxLayout()
        self.apply_button = QPushButton("Apply")
        self.save_button = QPushButton("Save As New")
        row1.addWidget(self.apply_button)
        row1.addWidget(self.save_button)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.duplicate_button = QPushButton("Duplicate")
        self.delete_button = QPushButton("Delete")
        row2.addWidget(self.duplicate_button)
        row2.addWidget(self.delete_button)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.import_button = QPushButton("Import...")
        self.export_button = QPushButton("Export...")
        row3.addWidget(self.import_button)
        row3.addWidget(self.export_button)
        layout.addLayout(row3)

        self.apply_button.clicked.connect(self._on_apply)
        self.save_button.clicked.connect(self._on_save_as_new)
        self.duplicate_button.clicked.connect(self._on_duplicate)
        self.delete_button.clicked.connect(self._on_delete)
        self.import_button.clicked.connect(self._on_import)
        self.export_button.clicked.connect(self._on_export)
        self.list_widget.itemDoubleClicked.connect(lambda _item: self._on_apply())

        self.refresh()

    def refresh(self):
        selected = self._selected_name()
        self.list_widget.clear()
        self.list_widget.addItems(preset_io.list_presets())
        if selected:
            matches = self.list_widget.findItems(selected, Qt.MatchExactly)
            if matches:
                self.list_widget.setCurrentItem(matches[0])

    def _selected_name(self):
        item = self.list_widget.currentItem()
        return item.text() if item else None

    def _on_apply(self):
        name = self._selected_name()
        if not name:
            return
        try:
            preset_layers = preset_io.load_preset(name)
        except FileNotFoundError:
            QMessageBox.warning(self, "Apply Preset", f"Preset '{name}' no longer exists.")
            self.refresh()
            return

        commands = []
        for layer in preset_layers:
            layer_name = str(layer)
            old_layer = next((l for l in self.document.layers if str(l) == layer_name), None)
            commands.append(ChangeLayerCommand(self.document, layer_name, old_layer, layer))
        if not commands:
            return

        self.document.execute_command(CompositeCommand(commands))
        self.viewer.update_view()
        self.layer_stack_panel.refresh()
        if self.on_layers_changed:
            self.on_layers_changed()

    def _on_save_as_new(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        try:
            preset_io.save_preset(name, self.document.layers)
        except FileExistsError:
            reply = QMessageBox.question(self, "Save Preset", f"A preset named '{name}' already exists. Overwrite it?")
            if reply != QMessageBox.Yes:
                return
            preset_io.save_preset(name, self.document.layers, overwrite=True)
        except ValueError as e:
            QMessageBox.warning(self, "Save Preset", str(e))
            return
        self.refresh()

    def _on_duplicate(self):
        name = self._selected_name()
        if not name:
            return
        new_name, ok = QInputDialog.getText(self, "Duplicate Preset", "New preset name:", text=f"{name} Copy")
        if not ok or not new_name.strip():
            return
        try:
            preset_io.duplicate_preset(name, new_name.strip())
        except (FileNotFoundError, FileExistsError, ValueError) as e:
            QMessageBox.warning(self, "Duplicate Preset", str(e))
            return
        self.refresh()

    def _on_delete(self):
        name = self._selected_name()
        if not name:
            return
        reply = QMessageBox.question(self, "Delete Preset", f"Delete preset '{name}'?")
        if reply != QMessageBox.Yes:
            return
        preset_io.delete_preset(name)
        self.refresh()

    def _on_import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Preset", "", "Preset Files (*.json)")
        if not path:
            return
        try:
            preset_io.import_preset(path)
        except Exception as e:
            QMessageBox.critical(self, "Import Preset", f"Could not import preset:\n{e}")
            return
        self.refresh()

    def _on_export(self):
        name = self._selected_name()
        if not name:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Preset", f"{name}.json", "Preset Files (*.json)")
        if not path:
            return
        try:
            preset_io.export_preset(name, path)
        except Exception as e:
            QMessageBox.critical(self, "Export Preset", f"Could not export preset:\n{e}")
