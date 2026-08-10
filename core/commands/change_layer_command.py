from core.commands.base_command import Command

class ChangeLayerCommand(Command):
    """Replaces the current layer of a given type (e.g. "Brightness") with a new
    value as a single undo step, restoring the previous layer (or removing it
    entirely if there wasn't one) on undo. Either old_layer or new_layer may be
    None, meaning "no adjustment of this type" (e.g. resetting to default)."""

    def __init__(self, document, layer_name: str, old_layer, new_layer):
        self.document = document
        self.layer_name = layer_name
        self.old_layer = old_layer
        self.new_layer = new_layer

    def _remove_existing(self):
        self.document.layers = [l for l in self.document.layers if str(l) != self.layer_name]

    def execute(self):
        self._remove_existing()
        if self.new_layer is not None:
            self.document.layers.append(self.new_layer)

    def undo(self):
        self._remove_existing()
        if self.old_layer is not None:
            self.document.layers.append(self.old_layer)
