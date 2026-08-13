"""Persisted, cross-session app preferences (Edit > Preferences), backed
by QSettings as a plain .ini file under the user's app-data directory -
no registry writes, so it's a single visible/removable file.

AppSettings takes an optional `backing` QSettings instance. Production
code (MainWindow's default) leaves it unset and gets the real, on-disk
store; tests that exercise settings-dependent behavior should construct
their own QSettings pointed at a temp .ini file and pass it in, so test
runs never read or write the real user's saved preferences.
"""

from PySide6.QtCore import QSettings

ORG_NAME = "OpenLightRoom"
APP_NAME = "OpenLightRoom"

MAX_RECENT_PROJECTS = 8

_PREVIEW_QUALITY_KEY = "preview/default_quality_label"
_CONFIRM_EXIT_KEY = "behavior/confirm_before_exit"
_EXPORT_FORMAT_KEY = "export/default_format"
_EXPORT_QUALITY_KEY = "export/default_quality"
_EXPORT_BIT_DEPTH_KEY = "export/default_bit_depth"
_RECENT_PROJECTS_KEY = "recent/projects"

DEFAULT_PREVIEW_QUALITY_LABEL = "Full Quality"
DEFAULT_CONFIRM_BEFORE_EXIT = True
DEFAULT_EXPORT_FORMAT = "JPG"
DEFAULT_EXPORT_QUALITY = 95
DEFAULT_EXPORT_BIT_DEPTH = "8-bit"


def _to_bool(value) -> bool:
    # QSettings round-trips bools as the string "true"/"false" on the
    # INI backend rather than a real bool, depending on platform/Qt
    # version - normalize either representation.
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


class AppSettings:
    def __init__(self, backing: QSettings = None):
        self._settings = backing if backing is not None else QSettings(
            QSettings.IniFormat, QSettings.UserScope, ORG_NAME, APP_NAME)

    # --- preview ------------------------------------------------------
    def default_preview_quality_label(self) -> str:
        return self._settings.value(_PREVIEW_QUALITY_KEY, DEFAULT_PREVIEW_QUALITY_LABEL, type=str)

    def set_default_preview_quality_label(self, label: str):
        self._settings.setValue(_PREVIEW_QUALITY_KEY, label)

    # --- behavior -------------------------------------------------------
    def confirm_before_exit(self) -> bool:
        return _to_bool(self._settings.value(_CONFIRM_EXIT_KEY, DEFAULT_CONFIRM_BEFORE_EXIT))

    def set_confirm_before_exit(self, enabled: bool):
        self._settings.setValue(_CONFIRM_EXIT_KEY, bool(enabled))

    # --- export defaults --------------------------------------------------
    def default_export_format(self) -> str:
        return self._settings.value(_EXPORT_FORMAT_KEY, DEFAULT_EXPORT_FORMAT, type=str)

    def set_default_export_format(self, fmt: str):
        self._settings.setValue(_EXPORT_FORMAT_KEY, fmt)

    def default_export_quality(self) -> int:
        return int(self._settings.value(_EXPORT_QUALITY_KEY, DEFAULT_EXPORT_QUALITY))

    def set_default_export_quality(self, quality: int):
        self._settings.setValue(_EXPORT_QUALITY_KEY, int(quality))

    def default_export_bit_depth(self) -> str:
        return self._settings.value(_EXPORT_BIT_DEPTH_KEY, DEFAULT_EXPORT_BIT_DEPTH, type=str)

    def set_default_export_bit_depth(self, bit_depth: str):
        self._settings.setValue(_EXPORT_BIT_DEPTH_KEY, bit_depth)

    # --- recent projects --------------------------------------------------
    def recent_projects(self) -> list[str]:
        value = self._settings.value(_RECENT_PROJECTS_KEY, [])
        if isinstance(value, str):
            # Some QSettings backends collapse a single-item list back to
            # a bare string on read - restore it as a one-item list.
            return [value] if value else []
        return list(value or [])

    def add_recent_project(self, path: str):
        existing = [p for p in self.recent_projects() if p != path]
        existing.insert(0, path)
        self._settings.setValue(_RECENT_PROJECTS_KEY, existing[:MAX_RECENT_PROJECTS])

    def remove_recent_project(self, path: str):
        remaining = [p for p in self.recent_projects() if p != path]
        self._settings.setValue(_RECENT_PROJECTS_KEY, remaining)

    def clear_recent_projects(self):
        self._settings.setValue(_RECENT_PROJECTS_KEY, [])

    def sync(self):
        self._settings.sync()
