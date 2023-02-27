from __future__ import annotations

import os
from typing import Tuple

from ...impl.stable_diffusion.types import VAEModel
from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties.inputs import StableDiffusionPtFileInput
from ...properties.outputs import DirectoryOutput, FileNameOutput
from ...properties.outputs.stable_diffusion_outputs import VAEModelOutput
from ...utils.utils import split_file_path
from . import category as StableDiffusionCategory


@NodeFactory.register("chainner:stable_diffusion:load_vae")
class LoadModelNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = ""
        self.inputs = [StableDiffusionPtFileInput(primary_input=True)]
        self.outputs = [
            VAEModelOutput(),
            DirectoryOutput("Model Directory", of_input=0),
            FileNameOutput("Model Name", of_input=0),
        ]

        self.category = StableDiffusionCategory
        self.name = "Load VAE"
        self.icon = "PyTorch"
        self.sub = "Input & Output"

    def run(self, path: str) -> Tuple[VAEModel, str, str]:
        assert os.path.exists(path), f"Model file at location {path} does not exist"
        assert os.path.isfile(path), f"Path {path} is not a file"

        vae = VAEModel.from_model(path)

        dirname, basename, _ = split_file_path(path)
        return vae, dirname, basename
