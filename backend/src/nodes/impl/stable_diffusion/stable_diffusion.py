from typing import List
import numpy as np
import sdkit
import torch
from PIL import Image
from sdkit.models import load_model, unload_model
from sdkit.generate import generate_images


class StableDiffusion:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_file(cls, path: str):
        context = sdkit.Context()
        context.vram_usage_level = 'high'
        context.model_paths["stable-diffusion"] = path
        load_model(context, "stable-diffusion")
        return cls(context.models['stable-diffusion'])

    @torch.inference_mode()
    def forward(self, prompt: str) -> np.ndarray:

        context = sdkit.Context()
        context.vram_usage_level = 'high'
        context.models['stable-diffusion'] = self.model

        images: List[Image] = generate_images(context=context, prompt=prompt)
        image = np.array(images[0])

        return image
