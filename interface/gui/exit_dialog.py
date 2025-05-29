from PySide6.QtWidgets import QHBoxLayout, QDialog, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt

class ExitDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Exit Dialog")

        exit_layout = QVBoxLayout(self)
        exit_layout2 = QHBoxLayout()

        exit_layout.addWidget(QLabel("Are you sure?"))
        self.yes_btn = QPushButton("Yes")
        self.no_btn = QPushButton("No")
        exit_layout2.addWidget(self.yes_btn)
        exit_layout2.addWidget(self.no_btn)
        exit_layout.addLayout(exit_layout2)