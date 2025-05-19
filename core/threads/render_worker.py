from PySide6.QtCore import QThread, Signal
import numpy as np

class RenderWorker(QThread):
    rendered = Signal(np.ndarray)   # Signal emitted when rendering is done

    def __init__(self, document):
        super().__init__()
        self.document = document

    def run(self):
        image = self.document.render()
        assert image is not None, "Rendered image is None"
        self.rendered.emit(image)