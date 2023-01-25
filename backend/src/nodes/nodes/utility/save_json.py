from __future__ import annotations

import os
from sanic.log import logger

from . import category as UtilityCategory
from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties.inputs import (
    ImageInput,
    DirectoryInput,
    TextInput,
)


@NodeFactory.register("chainner:utility:save_json")
class ImWriteNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = "Save JSON to file at a specified directory."
        self.inputs = [
            TextInput(label="JSON"),
            DirectoryInput(has_handle=True),
            TextInput("Subdirectory Path").make_optional(),
            TextInput("File Name"),
        ]
        self.category = UtilityCategory
        self.name = "Save JSON"
        self.outputs = []
        self.icon = "MdSave"
        self.sub = "Input & Output"

        self.side_effects = True

    def run(
            self,
            value: str,
            base_directory: str,
            relative_path: Union[str, None],
            filename: str,
    ) -> None:
        full_name = filename+".json" if not filename.endswith(".json") else filename

        if relative_path and relative_path != ".":
            base_directory = os.path.join(base_directory, relative_path)
        full_path = os.path.join(base_directory, full_name)

        logger.debug(f"Writing JSON to path: {full_path}")
        os.makedirs(base_directory, exist_ok=True)

        open(full_path, 'w').write(value)
