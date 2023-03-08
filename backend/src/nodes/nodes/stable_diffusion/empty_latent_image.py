from __future__ import annotations

from ...impl.stable_diffusion.types import LatentImage
from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties.inputs import SliderInput
from ...properties.outputs.stable_diffusion_outputs import LatentImageOutput
from . import category as StableDiffusionCategory


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
        return LatentImage.empty(width, height)
