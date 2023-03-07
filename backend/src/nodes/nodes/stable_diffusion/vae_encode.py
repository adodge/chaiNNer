from __future__ import annotations

import numpy as np
from PIL import Image
from comfy.latent_image import RGBImage

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


@NodeFactory.register("chainner:stable_diffusion:vae_encode")
class VAEEncodeNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = ""
        self.inputs = [
            VAEModelInput(),
            ImageInput(),
        ]
        self.outputs = [
            LatentImageOutput(),
        ]

        self.category = StableDiffusionCategory
        self.name = "VAE Encode"
        self.icon = "PyTorch"
        self.sub = "Input & Output"

    def run(self, vae: VAEModel, image: np.ndarray) -> LatentImage:
        try:
            vae.to("cuda")
            img = RGBImage.from_image(_array_to_image(image)).to("cuda")
            latent = vae.encode(img)
        finally:
            vae.to("cpu")
        return latent.to("cpu")
