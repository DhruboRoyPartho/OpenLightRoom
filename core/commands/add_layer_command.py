from core.commands.base_command import Command

class AddLayerCommand(Command):
    def __init__(self, document, layer):
        self.document = document
        self.layer = layer

    def execute(self):
        self.document.layers.append(self.layer)

    def undo(self):
        if self.layer in self.document.layers:
            self.document.layers.remove(self.layer)