from __future__ import annotations

from typing import Tuple, Optional


from ...impl.stable_diffusion.types import CLIPModel, Conditioning, LatentImage

from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties.inputs import TextAreaInput, SliderInput
from ...properties.inputs.stable_diffusion_inputs import CLIPModelInput
from . import category as StableDiffusionCategory
from ...properties.outputs.stable_diffusion_outputs import ConditioningOutput, LatentImageOutput


@NodeFactory.register("chainner:stable_diffusion:empty_latent_image")
class EmptyLatentImageNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = ""
        self.inputs = [
            SliderInput(
                "Width",
                minimum=64,
                default=512,
                maximum=2048,
                slider_step=8,
                controls_step=8,
            ),
            SliderInput(
                "Height",
                minimum=64,
                default=512,
                maximum=2048,
                slider_step=8,
                controls_step=8,
            ),
        ]
        self.outputs = [
            LatentImageOutput(),
        ]

        self.category = StableDiffusionCategory
        self.name = "Empty Latent Image"
        self.icon = "PyTorch"
        self.sub = "Latent"

    def run(self, width: int, height: int) -> LatentImage:
        img = LatentImage.empty(width, height)
        return img