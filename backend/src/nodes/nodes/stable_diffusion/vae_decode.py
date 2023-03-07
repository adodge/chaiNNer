from __future__ import annotations

import cv2
import numpy as np
import torch

from ...impl.stable_diffusion.types import LatentImage, VAEModel
from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties.inputs.stable_diffusion_inputs import LatentImageInput, VAEModelInput
from ...properties.outputs import ImageOutput
from . import category as StableDiffusionCategory


@NodeFactory.register("chainner:stable_diffusion:vae_decode")
class VAEDecodeNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = ""
        self.inputs = [
            VAEModelInput(),
            LatentImageInput(),
        ]
        self.outputs = [
            ImageOutput(),
        ]

        self.category = StableDiffusionCategory
        self.name = "VAE Decode"
        self.icon = "PyTorch"
        self.sub = "Input & Output"

    @torch.no_grad()
    def run(self, vae: VAEModel, latent_image: LatentImage) -> np.ndarray:
        try:
            vae.to("cuda")
            latent_image.to("cuda")
            img = vae.decode(latent_image)
        finally:
            vae.to("cpu")
            latent_image.to("cpu")
        arr = np.array(img.to_image())
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr
