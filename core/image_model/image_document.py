import numpy as np

class ImageDocument:
    def __init__(self, base_image: np.ndarray):
        self.base_image = base_image
        self.layers = []

        self.history = []
        self.redo_stack = []

    def add_layer(self, layer):
        self.layers.append(layer)
        self.history.append(f"Added {str(layer)}")
        self.redo_stack.clear()

    def render(self) -> np.ndarray:
        image = self.base_image.copy()
        for layer in self.layers:
            image = layer.apply(image)
        return image
    
    def execute_command(self, command):
        command.execute()
        self.history.append(command)
        self.redo_stack.clear()

    def undo(self):
        if self.history:
            command = self.history.pop()
            command.undo()
            self.redo_stack.append(command)

    def redo(self):
        if self.redo_stack:
            command = self.redo_stack.pop()
            command.execute()
            self.history.append(command)
