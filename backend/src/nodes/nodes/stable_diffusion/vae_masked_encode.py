from __future__ import annotations

import numpy as np
from PIL import Image
from comfy.latent_image import RGBImage, GreyscaleImage

from ...impl.stable_diffusion.types import LatentImage, VAEModel
from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties.inputs import ImageInput
from ...properties.inputs.stable_diffusion_inputs import VAEModelInput
from ...properties.outputs.stable_diffusion_outputs import LatentImageOutput
from . import category as StableDiffusionCategory


def _array_to_image(arr: np.ndarray) -> Image:
    arr = (np.clip(arr, 0, 1) * 255).round().astype("uint8")
    return Image.fromarray(arr)


@NodeFactory.register("chainner:stable_diffusion:vae_masked_encode")
class VAEMaskedEncodeNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = ""
        self.inputs = [
            VAEModelInput(),
            ImageInput(channels=3),
            ImageInput(channels=1),
        ]
        self.outputs = [
            LatentImageOutput(),
        ]

        self.category = StableDiffusionCategory
        self.name = "VAE Masked Encode"
        self.icon = "PyTorch"
        self.sub = "Input & Output"

    def run(self, vae: VAEModel, image: np.ndarray, mask: np.ndarray) -> LatentImage:

        try:
            vae.to("cuda")
            img = RGBImage.from_image(_array_to_image(image)).to("cuda")
            msk = GreyscaleImage.from_image(_array_to_image(mask)).to("cuda")
            latent = vae.masked_encode(img, msk)
        finally:
            vae.to("cpu")
        return latent.to("cpu")
