from __future__ import annotations

import cv2
import numpy as np
import torch

from ...impl.stable_diffusion.types import LatentImage, VAEModel, image_to_array
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
            LatentImageInput(),
            VAEModelInput(),
        ]
        self.outputs = [
            ImageOutput(image_type="""Image{ width: Input0.width, height: Input0.height }""", channels=3),
        ]

        self.category = StableDiffusionCategory
        self.name = "VAE Decode"
        self.icon = "PyTorch"
        self.sub = "Input & Output"

    @torch.no_grad()
    def run(self, latent_image: LatentImage, vae: VAEModel) -> np.ndarray:
        try:
            vae.cuda()
            latent_image.cuda()
            img = vae.decode(latent_image)
        finally:
            vae.cpu()
            latent_image.cpu()

        return cv2.cvtColor(img.to_array(), cv2.COLOR_RGB2BGR)
