from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
from core.threads.render_queue import RenderQueue
import cv2

class ImageViewer(QLabel):
    def __init__(self, document):
        super().__init__()
        self.document = document
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)

        self.render_queue = RenderQueue(self.document)
        self.render_queue.image_rendered.connect(self.set_image)

        self.worker = None  # Store reference to prevent GC
        self.current_pixmap = None
        self.update_view()

    def update_view(self):
        self.render_queue.request_render()
    
    def set_image(self, img):
        if img is None:
            return
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.setPixmap(QPixmap.fromImage(qimg).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self.update_view()