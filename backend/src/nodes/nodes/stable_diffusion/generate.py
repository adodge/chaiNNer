from __future__ import annotations

import numpy as np

from . import category as StableDiffusionCategory
from nodes.impl.stable_diffusion.types import SDKitModel
from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties.inputs import TextInput, SDKitModelInput
from ...properties.outputs import ImageOutput


@NodeFactory.register("chainner:stable_diffusion:generate")
class GenerateNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = ""
        self.inputs = [
            SDKitModelInput("Model").with_id(0),
            TextInput("Prompt"),
        ]
        self.outputs = [
            ImageOutput(
                "Image",
            )
        ]

        self.category = StableDiffusionCategory
        self.name = "Generate"
        self.icon = "BsFillImageFill"
        self.sub = "Text to Text"

    def run(
        self,
        model: SDKitModel,
        prompt: str,
    ) -> np.ndarray:
        image = model.sd.forward(prompt)
        return image
