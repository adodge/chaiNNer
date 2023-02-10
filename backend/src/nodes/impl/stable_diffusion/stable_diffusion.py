from typing import List
import numpy as np
import sdkit
import torch
from PIL import Image
from sdkit.models import load_model, unload_model
from sdkit.generate import generate_images


class StableDiffusion:
    def __init__(self, context: sdkit.Context):
        self.context = context

    @classmethod
    def from_file(cls, path: str):
        context = sdkit.Context()
        context.model_paths['stable-diffusion'] = path
        return cls(context)

    @torch.inference_mode()
    def forward(self, prompt: str) -> np.ndarray:
        load_model(self.context, 'stable-diffusion')

        images: List[Image] = generate_images(context=self.context, prompt=prompt)
        image = np.array(images[0])

        unload_model(self.context, 'stable-diffusion')

        return image