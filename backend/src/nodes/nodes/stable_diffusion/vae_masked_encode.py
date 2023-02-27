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

    def run(
        self, vae: comfy.VAEModel, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        img = Image.fromarray(image)
        mask_img = Image.fromarray(mask)
        latent = vae.masked_encode(img, mask_img)
        return latent
