from __future__ import annotations

import cv2
import numpy as np
from comfy.clip import CLIPModel
from comfy.conditioning import Conditioning
from comfy.latent_image import (
    CropMethod,
    GreyscaleImage,
    LatentImage,
    RGBImage,
    UpscaleMethod,
)
from comfy.stable_diffusion import (
    BuiltInCheckpointConfigName,
    CheckpointConfig,
    Sampler,
    Scheduler,
    StableDiffusionModel,
    load_checkpoint,
)
from comfy.vae import VAEModel
from PIL import Image

__all__ = [
    "CLIPModel",
    "Conditioning",
    "CropMethod",
    "LatentImage",
    "UpscaleMethod",
    "BuiltInCheckpointConfigName",
    "CheckpointConfig",
    "Sampler",
    "Scheduler",
    "StableDiffusionModel",
    "load_checkpoint",
    "VAEModel",
    "RGBImage",
    "GreyscaleImage",
]


def array_to_image(arr: np.ndarray) -> Image:
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    arr = (np.clip(arr, 0, 1) * 255).round().astype("uint8")
    return Image.fromarray(arr)
