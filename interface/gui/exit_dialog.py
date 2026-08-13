from interface.gui.confirm_dialog import ConfirmDialog


class ExitDialog(ConfirmDialog):
    def __init__(self, parent=None):
        super().__init__(
            "Close Application",
            "Do you want to close the app? Any unsaved changes will be lost.",
            confirm_text="Close",
            parent=parent,
        )
