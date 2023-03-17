from __future__ import annotations

from ...impl.external_stable_diffusion import nearest_valid_size
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
                slider_step=64,
                controls_step=64,
            ),
            SliderInput(
                "Height",
                minimum=64,
                default=512,
                maximum=2048,
                slider_step=64,
                controls_step=64,
            ),
        ]
        self.outputs = [
            LatentImageOutput(
                image_type="""def nearest_valid(n: number) = int & floor(n / 64) * 64;
                LatentImage {
                    width: nearest_valid(Input0),
                    height: nearest_valid(Input1)
                }""",
            ),
        ]

        self.category = StableDiffusionCategory
        self.name = "Empty Latent Image"
        self.icon = "PyTorch"
        self.sub = "Latent"

    def run(self, width: int, height: int) -> LatentImage:
        width, height = nearest_valid_size(
            width, height, step=64
        )

        return LatentImage.empty(width, height)
