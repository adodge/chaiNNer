from __future__ import annotations

import numpy as np
import torch
from comfy.latent_image import RGBImage

from ...impl.external_stable_diffusion import nearest_valid_size
from ...impl.pil_utils import InterpolationMethod, resize
from ...impl.stable_diffusion.types import LatentImage, VAEModel
from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties.inputs import ImageInput
from ...properties.inputs.stable_diffusion_inputs import VAEModelInput
from ...properties.outputs.stable_diffusion_outputs import LatentImageOutput
from ...utils.utils import get_h_w_c
from . import category as StableDiffusionCategory


@NodeFactory.register("chainner:stable_diffusion:vae_encode")
class VAEEncodeNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = ""
        self.inputs = [
            ImageInput(channels=3),
            VAEModelInput(),
        ]
        self.outputs = [
            LatentImageOutput(
                image_type="""def nearest_valid(n: number) = int & floor(n / 64) * 64;
                LatentImage {
                    width: nearest_valid(Input0.width),
                    height: nearest_valid(Input0.height)
                }""",
            ),
        ]

        self.category = StableDiffusionCategory
        self.name = "VAE Encode"
        self.icon = "PyTorch"
        self.sub = "Input & Output"

    @torch.no_grad()
    def run(self, image: np.ndarray, vae: VAEModel) -> LatentImage:
        height, width, _ = get_h_w_c(image)

        width1, height1 = nearest_valid_size(
            width, height, step=64
        )  # This cooperates with the "image_type" of the ImageOutput

        if width1 != width or height1 != height:
            image = resize(image, (width1, height1), InterpolationMethod.AUTO)

        try:
            vae.cuda()
            img = RGBImage.from_array(image, device="cuda")
            latent = vae.encode(img)
        finally:
            vae.cpu()
        return latent.cpu()
