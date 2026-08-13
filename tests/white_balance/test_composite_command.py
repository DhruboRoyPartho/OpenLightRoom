"""Tests for core/commands/composite_command.py."""

from core.commands.composite_command import CompositeCommand


class _RecordingCommand:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def execute(self):
        self.log.append(("execute", self.name))

    def undo(self):
        self.log.append(("undo", self.name))


def test_execute_runs_commands_in_order():
    log = []
    composite = CompositeCommand([_RecordingCommand("A", log), _RecordingCommand("B", log)])
    composite.execute()
    assert log == [("execute", "A"), ("execute", "B")]


def test_undo_runs_commands_in_reverse_order():
    log = []
    composite = CompositeCommand([_RecordingCommand("A", log), _RecordingCommand("B", log)])
    composite.execute()
    log.clear()
    composite.undo()
    assert log == [("undo", "B"), ("undo", "A")]


def test_works_with_a_real_document_and_change_layer_command():
    import numpy as np
    from core.image_model.image_document import ImageDocument
    from core.adjustment_layers.temperature_layer import TemperatureLayer
    from core.adjustment_layers.tint_layer import TintLayer
    from core.commands.change_layer_command import ChangeLayerCommand

    doc = ImageDocument(np.zeros((2, 2, 3), dtype=np.float32))
    cmd = CompositeCommand([
        ChangeLayerCommand(doc, "Temperature", None, TemperatureLayer(20.0)),
        ChangeLayerCommand(doc, "Tint", None, TintLayer(-10.0)),
    ])
    doc.execute_command(cmd)

    names = {str(l) for l in doc.layers}
    assert names == {"Temperature", "Tint"}
    assert len(doc.history) == 1  # one undo step for both layers

    doc.undo()
    assert doc.layers == []
