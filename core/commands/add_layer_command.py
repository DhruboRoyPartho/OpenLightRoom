from core.commands.base_command import Command

class AddLayerCommand(Command):
    def __init__(self, document, layer, layer_name: str):
        self.document = document
        self.layer = layer
        self.layer_name = layer_name

    def execute(self):
        # print("add layer working")
        # for layer in self.document.layers:
        #     if layer.__str__() == "Brightness":
        #         self.document.layers = [layer if x.__str__() == "Brightness" else x for x in self.document.layers]
        #         return
        # self.document.layers.append(self.layer)
        
        # index = 0
        # for layer in self.document.layers:
        #     if layer.__str__() != self.layer_name:
        #         self.document.layers[index] = self.layer
        
        # self.document.layers = [layer for layer in self.document.layers if layer.__str__() != self.layer_name]
        self.document.layers.append(self.layer)
        # print(self.document.layers)

    def undo(self):
        if self.layer in self.document.layers:
            self.document.layers.remove(self.layer)