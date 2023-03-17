from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch

from ...group import group
from ...impl.stable_diffusion.types import (
    CLIPModel,
    LatentImage,
    Sampler,
    Scheduler,
    StableDiffusionModel,
    VAEModel,
)
from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties.inputs import EnumInput, NumberInput, SliderInput, TextAreaInput
from ...properties.inputs.stable_diffusion_inputs import (
    CLIPModelInput,
    StableDiffusionModelInput,
    VAEModelInput,
)
from ...properties.outputs import ImageOutput
from . import category as StableDiffusionCategory


@NodeFactory.register("chainner:stable_diffusion:text2image")
class KSamplerNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = ""
        self.inputs = [
            StableDiffusionModelInput(),
            CLIPModelInput(),
            VAEModelInput(),
            SliderInput(
                "Width",
                minimum=64,
                default=512,
                maximum=2048,
                slider_step=8,
                controls_step=8,
            ),
            SliderInput(
                "Height",
                minimum=64,
                default=512,
                maximum=2048,
                slider_step=8,
                controls_step=8,
            ),
            TextAreaInput("Prompt").make_optional(),
            TextAreaInput("Negative Prompt").make_optional(),
            group("seed")(
                NumberInput("Seed", minimum=0, default=42, maximum=4294967296)
            ),
            SliderInput("Steps", minimum=1, default=20, maximum=150),
            EnumInput(
                Sampler,
                default_value=Sampler.SAMPLE_EULER,
            ),
            EnumInput(
                Scheduler,
                default_value=Scheduler.NORMAL,
            ),
            SliderInput(
                "CFG Scale",
                minimum=1,
                default=7,
                maximum=20,
                controls_step=0.1,
                precision=1,
            ),
        ]
        self.outputs = [
            ImageOutput(),
        ]

        self.category = StableDiffusionCategory
        self.name = "Text to Image"
        self.icon = "PyTorch"
        self.sub = "Abstract"

    @torch.no_grad()
    def run(
        self,
        model: StableDiffusionModel,
        clip: CLIPModel,
        vae: VAEModel,
        width: int,
        height: int,
        positive: Optional[str],
        negative: Optional[str],
        seed: int,
        steps: int,
        sampler: Sampler,
        scheduler: Scheduler,
        cfg_scale: float,
    ) -> np.ndarray:
        positive = positive or ""
        negative = negative or ""

        try:
            clip.cuda()
            pos = clip.encode(positive)
            neg = clip.encode(negative)
        finally:
            clip.cpu()

        try:
            model.cuda()
            latent = LatentImage.empty(width, height, device="cuda")
            img = model.sample(
                positive=pos,
                negative=neg,
                latent_image=latent,
                seed=seed,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler=sampler,
                scheduler=scheduler,
                denoise_strength=1.0,
            )
            del latent, pos, neg
        finally:
            model.cpu()

        try:
            vae.cuda()
            out = vae.decode(img)
            del img
        finally:
            vae.cpu()

        return out.to_array()
