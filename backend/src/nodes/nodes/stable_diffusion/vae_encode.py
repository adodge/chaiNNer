from __future__ import annotations

import comfy
import numpy as np
from PIL import Image

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

    def run(self, vae: comfy.VAEModel, image: np.ndarray) -> np.ndarray:
        img = _array_to_image(image)
        latent = vae.encode(img)
        return latent
