from __future__ import annotations

from typing import Tuple, Optional

import comfy

from ...impl.stable_diffusion.types import StableDiffusionModel, Conditioning, LatentImage

from ...node_base import NodeBase, group
from ...node_factory import NodeFactory
from ...properties.inputs import TextAreaInput, SliderInput, NumberInput, EnumInput
from ...properties.inputs.stable_diffusion_inputs import CLIPModelInput, LatentImageInput, ConditioningInput, \
    StableDiffusionModelInput
from . import category as StableDiffusionCategory
from ...properties.outputs.stable_diffusion_outputs import ConditioningOutput, LatentImageOutput


@NodeFactory.register("chainner:stable_diffusion:k_sampler")
class KSamplerNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = ""
        self.inputs = [
            StableDiffusionModelInput(),
            ConditioningInput("Positive Conditioning"),
            ConditioningInput("Negative Conditioning"),
            LatentImageInput(),
            SliderInput(
                "Denoising Strength",
                minimum=0,
                default=0.75,
                maximum=1,
                slider_step=0.01,
                controls_step=0.1,
                precision=2,
            ),
            group("seed")(
                NumberInput("Seed", minimum=0, default=42, maximum=4294967296)
            ),
            SliderInput("Steps", minimum=1, default=20, maximum=150),
            EnumInput(
                comfy.Sampler,
                default_value=comfy.Sampler.SAMPLE_EULER,
            ),
            EnumInput(
                comfy.Scheduler,
                default_value=comfy.Scheduler.NORMAL,
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
            LatentImageOutput(),
        ]

        self.category = StableDiffusionCategory
        self.name = "K-Sampler"
        self.icon = "PyTorch"
        self.sub = "Latent"

    def run(self, model: StableDiffusionModel, positive: Conditioning, negative: Conditioning, latent_image: LatentImage,
            denoising_strength: float, seed: int, steps: int, sampler: comfy.Sampler, scheduler: comfy.Scheduler,
            cfg_strength: float) -> LatentImage:

        img = model.sample(
            positive=positive, negative=negative, latent_image=latent_image,
            seed=seed, steps=steps, cfg_strength=cfg_strength,
            sampler=sampler, scheduler=scheduler, denoise_strength=denoising_strength,
        )

        return img
