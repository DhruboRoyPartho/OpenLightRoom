import numpy as np

class ImageDocument:
    def __init__(self, base_image: np.ndarray):
        self.base_image = base_image
        self.layers = []

        self.history = []
        self.redo_stack = []

    def add_layer(self, layer):
        # if 'Brightness' in layer.__str__():
        #     self.layers = [layer]
        # self.layers = [layer if x.__str__() == 'Brightness' else x for x in self.layers]
        print("Add layer called from document")
        self.layers.append(layer)
        print(self.layers)
        self.history.append(f"{str(layer)}")
        self.redo_stack.clear()

    def render(self) -> np.ndarray:
        image = self.base_image.copy()

        seen = set()
        result = []

        for layer in reversed(self.layers):
            name = str(layer)
            if name not in seen:
                seen.add(name)
                result.append(layer)
        
        result = reversed(result)
        print("Layers: ", self.layers)
        print("refine layer: ", result)
        for layer in result:
            image = layer.apply(image)
        return image
        # for layer in self.layers:
        #     image = layer.apply(image)
        # return image

    
    def execute_command(self, command):
        command.execute()
        print("execute command executed")
        self.history.append(command)
        print("history appended")
        print("command: ", command)
        print("history: ", self.history)
        self.redo_stack.clear()

    def undo(self):
        print("Undo command called from document")
        if self.history:
            command = self.history.pop()
            command.undo()
            self.redo_stack.append(command)
        # if self.layers:
        #     command = self.layers.pop()
        #     command.undo()
        #     self.redo_stack.append(command)

    def redo(self):
        print("Redo command called from document")
        if self.redo_stack:
            command = self.redo_stack.pop()
            command.execute()
            self.history.append(command)
            # self.layers.apped(command)
