from PySide6.QtCore import QObject, QTimer, Signal
from core.threads.render_worker import RenderWorker

class RenderQueue(QObject):
    image_rendered = Signal(object)

    def __init__(self, document):
        super().__init__()
        self.document = document
        self.worker = None
        self.timer = QTimer()
        self.timer.setInterval(30)     # milliseconds
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._start_render)

    def request_render(self):
        self.timer.start()      # debounce rapid request

    def _start_render(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()

        self.worker = RenderWorker(self.document)
        self.worker.rendered.connect(self.image_rendered)
        self.worker.start()