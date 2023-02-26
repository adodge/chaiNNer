from __future__ import annotations

import os
from typing import Tuple

import comfy

from ...impl.stable_diffusion.types import StableDiffusionModel, VAEModel, CLIPModel

from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties.inputs import PthFileInput, CkptFileInput
from ...properties.outputs import DirectoryOutput, FileNameOutput
from ...properties.outputs.stable_diffusion_outputs import StableDiffusionModelOutput, CLIPModelOutput, VAEModelOutput
from ...utils.utils import split_file_path
from . import category as StableDiffusionCategory


@NodeFactory.register("chainner:stable_diffusion:load_checkpoint")
class LoadModelNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = ""
        self.inputs = [CkptFileInput(primary_input=True)]
        self.outputs = [
            StableDiffusionModelOutput(),
            CLIPModelOutput(),
            VAEModelOutput(),
            DirectoryOutput("Model Directory", of_input=0),
            FileNameOutput("Model Name", of_input=0),
        ]

        self.category = StableDiffusionCategory
        self.name = "Load Checkpoint"
        self.icon = "PyTorch"
        self.sub = "Input & Output"

    def run(self, path: str) -> Tuple[StableDiffusionModel, CLIPModel, VAEModel, str, str]:
        assert os.path.exists(path), f"Model file at location {path} does not exist"
        assert os.path.isfile(path), f"Path {path} is not a file"

        config = comfy.CheckpointConfig.from_built_in(comfy.BuiltInCheckpointConfigName.V1)

        sd, clip, vae = comfy.load_checkpoint(config=config, checkpoint_filepath=path, embedding_directory=None)

        dirname, basename, _ = split_file_path(path)
        return sd, clip, vae, dirname, basename