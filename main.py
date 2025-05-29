"""
Project: Open LightRoom
Author: Dhrubo Roy Partho
Date: 15-05-2025
Version: 1.0v
"""

from PySide6.QtWidgets import QApplication
from interface.gui.main_window import MainWindow
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())