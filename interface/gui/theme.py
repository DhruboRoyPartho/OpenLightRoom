"""Shared dark, Lightroom-style color palette and application theming.

Widgets that need custom-painted or per-widget styling (sliders, panel
headers, the canvas background) import the color constants below so the
whole app reads as one consistent design instead of a patchwork of ad-hoc
colors. apply_theme() sets the base look for every standard Qt widget
(menus, dialogs, buttons, scrollbars) so new dialogs inherit it for free.
"""

from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt

# --- Palette ---
BG_WINDOW = "#2b2b2b"      # outer window / menu bar
BG_PANEL = "#262626"       # side panels (controls, layers)
BG_CANVAS = "#1b1b1b"      # image canvas surround
BG_FIELD = "#333333"       # inputs, list items
BORDER = "#3f3f3f"
BORDER_LIGHT = "#555555"

TEXT = "#d8d8d8"
TEXT_DIM = "#8a8a8a"
TEXT_HEADER = "#9a9a9a"

ACCENT = "#2f6fed"
ACCENT_HOVER = "#4a83f5"

TRACK = "#565656"
HANDLE = "#f2f2f2"
HANDLE_BORDER = "#9a9a9a"


def apply_theme(app):
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG_WINDOW))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    palette.setColor(QPalette.Base, QColor(BG_FIELD))
    palette.setColor(QPalette.AlternateBase, QColor(BG_PANEL))
    palette.setColor(QPalette.ToolTipBase, QColor(BG_PANEL))
    palette.setColor(QPalette.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.Button, QColor(BG_PANEL))
    palette.setColor(QPalette.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.Link, QColor(ACCENT))
    palette.setColor(QPalette.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(TEXT_DIM))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(TEXT_DIM))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(TEXT_DIM))
    app.setPalette(palette)

    app.setStyleSheet(f"""
        QMainWindow {{ background-color: {BG_WINDOW}; }}

        QMenuBar {{
            background-color: {BG_WINDOW};
            color: {TEXT};
            border-bottom: 1px solid {BORDER};
        }}
        QMenuBar::item:selected {{ background-color: {BG_FIELD}; }}
        QMenu {{
            background-color: {BG_PANEL};
            color: {TEXT};
            border: 1px solid {BORDER};
        }}
        QMenu::item:selected {{ background-color: {ACCENT}; }}

        QPushButton {{
            background-color: {BG_FIELD};
            color: {TEXT};
            border: 1px solid {BORDER};
            border-radius: 4px;
            padding: 5px 12px;
        }}
        QPushButton:hover {{ background-color: #3d3d3d; border-color: {BORDER_LIGHT}; }}
        QPushButton:pressed {{ background-color: #2a2a2a; }}
        QPushButton:disabled {{ color: {TEXT_DIM}; }}

        QDialog {{ background-color: {BG_WINDOW}; }}

        QComboBox {{
            background-color: {BG_FIELD};
            color: {TEXT};
            border: 1px solid {BORDER};
            border-radius: 3px;
            padding: 3px 6px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {BG_PANEL};
            color: {TEXT};
            selection-background-color: {ACCENT};
        }}

        QSplitter::handle {{ background-color: {BG_WINDOW}; }}
        QSplitter::handle:hover {{ background-color: {ACCENT}; }}

        QScrollBar:vertical, QScrollBar:horizontal {{
            background: {BG_PANEL};
            border: none;
        }}
        QScrollBar::handle {{
            background: {BORDER_LIGHT};
            border-radius: 4px;
        }}
        QScrollBar::handle:hover {{ background: {TEXT_DIM}; }}

        QToolTip {{
            background-color: {BG_PANEL};
            color: {TEXT};
            border: 1px solid {BORDER};
        }}
    """)
