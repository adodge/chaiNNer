from __future__ import annotations

import comfy

from ...impl.stable_diffusion.types import (
    Conditioning,
    LatentImage,
    StableDiffusionModel,
)
from ...node_base import NodeBase, group
from ...node_factory import NodeFactory
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
        self.name = "Sample"
        self.icon = "PyTorch"
        self.sub = "Latent"

    def run(
        self,
        model: StableDiffusionModel,
        positive: Conditioning,
        negative: Conditioning,
        latent_image: LatentImage,
        denoising_strength: float,
        seed: int,
        steps: int,
        sampler: comfy.Sampler,
        scheduler: comfy.Scheduler,
        cfg_scale: float,
    ) -> LatentImage:
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

        return img
