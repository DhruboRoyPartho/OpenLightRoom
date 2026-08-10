from PySide6.QtCore import QObject, QTimer, Signal
from core.threads.render_worker import RenderWorker

class RenderQueue(QObject):
    image_rendered = Signal(object)

    def __init__(self, document):
        super().__init__()
        self.document = document
        self.worker = None
        self._pending_geometry_override = None
        self._render_requested_while_busy = False
        self.timer = QTimer()
        self.timer.setInterval(50)     # milliseconds
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._start_render)

    def request_render(self, geometry_override=None):
        self._pending_geometry_override = geometry_override
        self.timer.start()      # debounce rapid request

    def _start_render(self):
        if self.worker is not None and self.worker.isRunning():
            # A render is still in flight (e.g. a slow straighten/rotate on
            # a large image can outlast the 50ms debounce during a fast
            # drag). QThread.terminate() can kill a thread mid-operation
            # and corrupt memory or crash the process, so never do that -
            # just remember a fresher render is wanted and start it the
            # moment this one actually finishes.
            self._render_requested_while_busy = True
            return

        self._launch_worker()

    def _launch_worker(self):
        self.worker = RenderWorker(self.document, self._pending_geometry_override)
        self.worker.rendered.connect(self.image_rendered)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_worker_finished(self):
        if self._render_requested_while_busy:
            self._render_requested_while_busy = False
            self._launch_worker()