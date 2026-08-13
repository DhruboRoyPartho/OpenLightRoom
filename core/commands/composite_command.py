from core.commands.base_command import Command


class CompositeCommand(Command):
    """Bundles several commands into a single undo step - executed in order,
    undone in reverse order. Used where one user action legitimately
    changes more than one layer at once (e.g. Auto White Balance sets both
    Temperature and Tint together; one undo should restore both)."""

    def __init__(self, commands: list):
        self.commands = list(commands)

    def execute(self):
        for command in self.commands:
            command.execute()

    def undo(self):
        for command in reversed(self.commands):
            command.undo()
