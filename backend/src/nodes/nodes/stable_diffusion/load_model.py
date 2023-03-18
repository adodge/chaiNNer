from __future__ import annotations

import os
from typing import Tuple

from ...impl.stable_diffusion.types import (
    CLIPModel,
    StableDiffusionModel,
    VAEModel,
    load_checkpoint,
)
from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties.inputs import CkptFileInput
from ...properties.outputs import DirectoryOutput, FileNameOutput
from ...properties.outputs.stable_diffusion_outputs import (
    CLIPModelOutput,
    StableDiffusionModelOutput,
    VAEModelOutput,
)
from ...utils.utils import split_file_path
from . import category as StableDiffusionCategory


@NodeFactory.register("chainner:stable_diffusion:load_checkpoint")
class LoadModelNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = ""
        self.inputs = [CkptFileInput(primary_input=True)]
        self.outputs = [
            StableDiffusionModelOutput(kind="stable-diffusion", should_broadcast=True),
            CLIPModelOutput(kind="clip", should_broadcast=True),
            VAEModelOutput(should_broadcast=True),
            DirectoryOutput("Model Directory", of_input=0),
            FileNameOutput("Model Name", of_input=0),
        ]

        self.category = StableDiffusionCategory
        self.name = "Load Checkpoint"
        self.icon = "PyTorch"
        self.sub = "Input & Output"

    def run(
        self, path: str
    ) -> Tuple[StableDiffusionModel, CLIPModel, VAEModel, str, str]:
        assert os.path.exists(path), f"Model file at location {path} does not exist"
        assert os.path.isfile(path), f"Path {path} is not a file"

        sd, clip, vae = load_checkpoint(
            checkpoint_filepath=path, embedding_directory=None
        )

        dirname, basename, _ = split_file_path(path)
        return sd, clip, vae, dirname, basename
