from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton

from interface.gui.theme import TEXT, TEXT_DIM


class ConfirmDialog(QDialog):
    """A small, consistently-styled confirmation dialog for anything
    destructive or hard to undo (closing the app, closing the current
    project) - one shared look instead of a bespoke QMessageBox each time.
    accept()/reject() are wired internally, so callers just check
    exec() == QDialog.Accepted.
    """

    def __init__(self, title: str, message: str, confirm_text: str = "Close", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedWidth(360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(14)

        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {TEXT}; font-size: 14px; font-weight: 600;")
        layout.addWidget(title_label)

        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px;")
        layout.addWidget(message_label)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        button_row.addStretch(1)

        self.cancel_btn = QPushButton("Cancel")
        self.confirm_btn = QPushButton(confirm_text)
        self.confirm_btn.setDefault(True)
        self.confirm_btn.setStyleSheet("""
            QPushButton {
                background-color: #c0392b;
                border: 1px solid #c0392b;
                color: #ffffff;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #d6483a; border-color: #d6483a; }
            QPushButton:pressed { background-color: #a5311f; border-color: #a5311f; }
        """)
        button_row.addWidget(self.cancel_btn)
        button_row.addWidget(self.confirm_btn)
        layout.addLayout(button_row)

        self.confirm_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        # Aliases kept for callers written against the older Yes/No naming.
        self.yes_btn = self.confirm_btn
        self.no_btn = self.cancel_btn
