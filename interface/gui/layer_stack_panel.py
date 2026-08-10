from PySide6.QtWidgets import QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QPushButton, QHBoxLayout, QLabel
from PySide6.QtCore import Qt
from interface.gui.theme import BG_PANEL, BG_FIELD, BORDER, TEXT, TEXT_HEADER, ACCENT

class LayerStackPanel(QWidget):
    def __init__(self, document, viewer):
        super().__init__()
        self.document = document
        self.viewer = viewer

        self.setStyleSheet(f"LayerStackPanel {{ background-color: {BG_PANEL}; }}")

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(self.layout)

        header = QLabel("LAYERS")
        header.setStyleSheet(f"""
            QLabel {{
                font-weight: 600;
                font-size: 11px;
                color: {TEXT_HEADER};
                padding-bottom: 6px;
                border-bottom: 1px solid {BORDER};
                margin-bottom: 6px;
            }}
        """)
        self.layout.addWidget(header)

        self.layer_list = QListWidget()
        self.layer_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {BG_FIELD};
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 4px;
                padding: 4px;
            }}
            QListWidget::item {{
                padding: 6px 8px;
                border-radius: 3px;
            }}
            QListWidget::item:selected {{
                background-color: {ACCENT};
                color: #ffffff;
            }}
        """)
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