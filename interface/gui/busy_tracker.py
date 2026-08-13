from PySide6.QtCore import QObject, Signal


class BusyTracker(QObject):
    """A simple, reference-counted "is something happening" tracker shared
    across the app, so any slow operation (a render, a RAW import, a
    project save/load, an export) can report itself without needing to
    know about whatever UI element displays that state. Overlapping
    operations are supported - busyChanged only reports idle once every
    begin() has a matching end().
    """

    busyChanged = Signal(bool, str)  # (is_busy, current label - "" when idle)

    def __init__(self):
        super().__init__()
        self._active = []  # stack of in-flight labels, most-recent last

    def begin(self, label: str) -> None:
        self._active.append(label)
        self.busyChanged.emit(True, label)

    def end(self, label: str) -> None:
        if label in self._active:
            self._active.remove(label)
        if self._active:
            self.busyChanged.emit(True, self._active[-1])
        else:
            self.busyChanged.emit(False, "")

    def is_busy(self) -> bool:
        return bool(self._active)
