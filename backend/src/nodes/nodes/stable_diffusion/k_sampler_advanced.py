from __future__ import annotations

import torch

from ...impl.stable_diffusion.types import (
    Conditioning,
    LatentImage,
    Sampler,
    Scheduler,
    StableDiffusionModel,
)
from ...node_base import NodeBase, group
from ...node_factory import NodeFactory
from ...properties.inputs import BoolInput, EnumInput, NumberInput, SliderInput
from ...properties.inputs.stable_diffusion_inputs import (
    ConditioningInput,
    LatentImageInput,
    StableDiffusionModelInput,
)
from ...properties.outputs.stable_diffusion_outputs import LatentImageOutput
from . import category as StableDiffusionCategory


@NodeFactory.register("chainner:stable_diffusion:k_sampler_advanced")
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
                default=1,
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
            SliderInput(
                "Start at step",
                minimum=0,
                default=0,
                maximum=10000,
            ),
            SliderInput(
                "End at step",
                minimum=0,
                default=10000,
                maximum=10000,
            ),
            BoolInput("Add Noise"),
            BoolInput("Return with leftover noise"),
        ]
        self.outputs = [
            LatentImageOutput(),
        ]

        self.category = StableDiffusionCategory
        self.name = "Advanced Sample"
        self.icon = "PyTorch"
        self.sub = "Latent"

    @torch.no_grad()
    def run(
        self,
        model: StableDiffusionModel,
        positive: Conditioning,
        negative: Conditioning,
        latent_image: LatentImage,
        denoising_strength: float,
        seed: int,
        steps: int,
        sampler: Sampler,
        scheduler: Scheduler,
        cfg_scale: float,
        start_at: int,
        end_at: int,
        add_noise: bool,
        return_with_leftover_noise: bool,
    ) -> LatentImage:

        try:
            model.to("cuda")
            positive.to("cuda")
            negative.to("cuda")
            latent_image.to("cuda")

            img = model.advanced_sample(
                positive=positive,
                negative=negative,
                latent_image=latent_image,
                seed=seed,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler=sampler,
                scheduler=scheduler,
                denoise_strength=denoising_strength,
                start_at_step=start_at,
                end_at_step=end_at,
                add_noise=add_noise,
                return_with_leftover_noise=return_with_leftover_noise,
            )
        finally:
            model.to("cpu")
            positive.to("cpu")
            negative.to("cpu")
            latent_image.to("cpu")

        return img.to("cpu")
