from PySide6.QtCore import QThread, Signal
import numpy as np

class RenderWorker(QThread):
    rendered = Signal(np.ndarray)   # Signal emitted when rendering is done

    def __init__(self, document, geometry_override=None):
        super().__init__()
        self.document = document
        self.geometry_override = geometry_override

    def run(self):
        image = self.document.render(geometry_override=self.geometry_override)
        assert image is not None, "Rendered image is None"
        self.rendered.emit(image)