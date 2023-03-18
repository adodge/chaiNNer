from __future__ import annotations

import torch

from ...group import group
from ...impl.stable_diffusion.types import (
    Conditioning,
    LatentImage,
    Sampler,
    Scheduler,
    StableDiffusionModel,
)
from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties import expression
from ...properties.inputs import EnumInput, NumberInput, SliderInput
from ...properties.inputs.stable_diffusion_inputs import (
    ConditioningInput,
    LatentImageInput,
    StableDiffusionModelInput,
)
from ...properties.outputs.stable_diffusion_outputs import LatentImageOutput
from . import category as StableDiffusionCategory


@NodeFactory.register("chainner:stable_diffusion:k_sampler")
class KSamplerNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = ""
        self.inputs = [
            LatentImageInput(),
            StableDiffusionModelInput(),
            ConditioningInput(
                "Positive Conditioning",
                input_type=expression.Conditioning(arch_as="Input1"),
            ),
            ConditioningInput(
                "Negative Conditioning",
                input_type=expression.Conditioning(arch_as="Input1"),
            ),
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
        ]
        self.outputs = [
            LatentImageOutput(
                image_type="""LatentImage {
                    width: Input0.width,
                    height: Input0.height
                }"""
            ),
        ]

        self.category = StableDiffusionCategory
        self.name = "Sample"
        self.icon = "PyTorch"
        self.sub = "Latent"

    @torch.no_grad()
    def run(
        self,
        latent_image: LatentImage,
        model: StableDiffusionModel,
        positive: Conditioning,
        negative: Conditioning,
        denoising_strength: float,
        seed: int,
        steps: int,
        sampler: Sampler,
        scheduler: Scheduler,
        cfg_scale: float,
    ) -> LatentImage:
        try:
            model.cuda()
            positive.cuda()
            negative.cuda()
            latent_image.cuda()

            img = model.sample(
                positive=positive,
                negative=negative,
                latent_image=latent_image,
                seed=seed,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler=sampler,
                scheduler=scheduler,
                denoise_strength=denoising_strength,
            )

        finally:
            model.cpu()
            positive.cpu()
            negative.cpu()
            latent_image.cpu()

        return img.cpu()
