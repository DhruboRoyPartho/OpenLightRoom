from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
import cv2

class ImageViewer(QLabel):
    def __init__(self, document):
        super().__init__()
        self.document = document
        self.setAlignment(Qt.AlignCenter)
        self.update_view()

    def update_view(self):
        img = self.document.render()
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.setPixmap(QPixmap.fromImage(qimg).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self.update_view()
