from __future__ import annotations

from ...impl.external_stable_diffusion import nearest_valid_size
from ...impl.stable_diffusion.types import CropMethod, LatentImage, UpscaleMethod
from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties.inputs import EnumInput, SliderInput
from ...properties.inputs.stable_diffusion_inputs import LatentImageInput
from ...properties.outputs.stable_diffusion_outputs import LatentImageOutput
from . import category as StableDiffusionCategory


@NodeFactory.register("chainner:stable_diffusion:latent_upscale")
class LatentUpscaleNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = ""
        self.inputs = [
            LatentImageInput(),
            EnumInput(
                UpscaleMethod,
                default_value=UpscaleMethod.BILINEAR,
            ),
            EnumInput(
                CropMethod,
                default_value=CropMethod.DISABLED,
            ),
            SliderInput(
                "width",
                unit="px",
                minimum=64,
                maximum=4096,
                default=512,
                slider_step=64,
                controls_step=64,
            ),
            SliderInput(
                "height",
                unit="px",
                minimum=64,
                maximum=4096,
                default=512,
                slider_step=64,
                controls_step=64,
            ),
        ]
        self.outputs = [
            LatentImageOutput(
                image_type="""def nearest_valid(n: number) = int & floor(n / 64) * 64;
                LatentImage {
                    width: nearest_valid(Input3),
                    height: nearest_valid(Input4)
                }"""
            ),
        ]

        self.category = StableDiffusionCategory
        self.name = "Upscale"
        self.icon = "PyTorch"
        self.sub = "Latent"

    def run(
        self,
        latent_image: LatentImage,
        upscale_method: UpscaleMethod,
        crop_method: CropMethod,
        width: int,
        height: int,
    ) -> LatentImage:
        width, height = nearest_valid_size(width, height, step=64)

        return latent_image.upscale(width, height, upscale_method, crop_method)
