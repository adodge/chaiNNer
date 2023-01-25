from __future__ import annotations

import datetime

from . import category as UtilityCategory
from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties.inputs import TextInput
from ...properties.outputs import TextOutput


@NodeFactory.register("chainner:utility:current_timestamp")
class TextValueNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = "Outputs the current timestamp as a string, formatted according to strftime."
        self.inputs = [
            TextInput("Format", default="%Y%m%d-%H%M%S"),
        ]
        self.outputs = [
            TextOutput("Text"),
        ]

        self.category = UtilityCategory
        self.name = "Current Timestamp"
        self.icon = "MdTextFields"
        self.sub = "Date & Time"

    def run(self, format: str) -> str:
        return datetime.datetime.now().strftime(format)
