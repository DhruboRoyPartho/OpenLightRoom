"""The app's Credits / About page - who built it, that it's free and
open source, and where to find the code - reached via Help > About
Open LightRoom."""

from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

from interface.gui.theme import TEXT, TEXT_DIM, ACCENT
from interface.gui.assets import LOGO_PATH
from interface.gui.app_info import (
    APP_NAME, APP_VERSION, APP_TAGLINE, AUTHOR_NAME, AUTHOR_EMAIL, GITHUB_URL,
)

LOGO_DISPLAY_SIZE = 88


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {APP_NAME}")
        self.setModal(True)
        self.setFixedWidth(380)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 26, 28, 20)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignHCenter)

        logo_label = QLabel()
        pixmap = QPixmap(LOGO_PATH)
        if not pixmap.isNull():
            logo_label.setPixmap(pixmap.scaled(
                LOGO_DISPLAY_SIZE, LOGO_DISPLAY_SIZE,
                Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo_label.setAlignment(Qt.AlignHCenter)
        layout.addWidget(logo_label)
        layout.addSpacing(12)

        name_label = QLabel(APP_NAME)
        name_label.setAlignment(Qt.AlignHCenter)
        name_label.setStyleSheet(f"color: {TEXT}; font-size: 18px; font-weight: 700;")
        layout.addWidget(name_label)

        tagline_label = QLabel(APP_TAGLINE)
        tagline_label.setAlignment(Qt.AlignHCenter)
        tagline_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        layout.addWidget(tagline_label)
        layout.addSpacing(4)

        version_label = QLabel(f"Version {APP_VERSION}")
        version_label.setAlignment(Qt.AlignHCenter)
        version_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        layout.addWidget(version_label)
        layout.addSpacing(16)

        oss_label = QLabel("Free and open source, released for anyone to use,\nstudy, and build on.")
        oss_label.setAlignment(Qt.AlignHCenter)
        oss_label.setWordWrap(True)
        oss_label.setStyleSheet(f"color: {TEXT}; font-size: 12px;")
        layout.addWidget(oss_label)
        layout.addSpacing(4)

        self.github_label = QLabel(f'<a href="{GITHUB_URL}" style="color:{ACCENT};">{GITHUB_URL}</a>')
        self.github_label.setAlignment(Qt.AlignHCenter)
        self.github_label.setOpenExternalLinks(True)
        self.github_label.setTextFormat(Qt.RichText)
        self.github_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.github_label)
        layout.addSpacing(20)

        self.credit_label = QLabel(
            "Created with passion by<br>"
            f'<span style="color:{TEXT}; font-weight:600;">{AUTHOR_NAME}</span><br>'
            f'<a href="mailto:{AUTHOR_EMAIL}" style="color:{ACCENT};">{AUTHOR_EMAIL}</a>'
        )
        self.credit_label.setAlignment(Qt.AlignHCenter)
        self.credit_label.setOpenExternalLinks(True)
        self.credit_label.setTextFormat(Qt.RichText)
        self.credit_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; line-height: 160%;")
        layout.addWidget(self.credit_label)
        layout.addSpacing(22)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        self.close_btn = QPushButton("Close")
        self.close_btn.setDefault(True)
        self.close_btn.clicked.connect(self.accept)
        close_row.addWidget(self.close_btn)
        close_row.addStretch(1)
        layout.addLayout(close_row)
