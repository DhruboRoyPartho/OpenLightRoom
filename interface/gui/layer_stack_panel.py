from PySide6.QtWidgets import QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt

class LayerStackPanel(QWidget):
    def __init__(self, document, viewer):
        super().__init__()
        self.document = document
        self.viewer = viewer

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        self.layer_list = QListWidget()
        self.layout.addWidget(self.layer_list)

        self.refresh()

    def refresh(self):
        self.layer_list.clear()
        for i, layer in enumerate(self.document.layers):
            item = QListWidgetItem()
            item.setText(str(layer))
            self.layer_list.addItem(item)

    def delete_selected(self):
        selected = self.layer_list.currentRow()
        if selected >= 0 and selected < len(self.document.layers):
            self.document.history.append(f"Deleted layer {selected}")
            del self.document.layers[selected]
            self.refresh()
            self.viewer.update_view()