from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

from . import category as ExternalStableDiffusionCategory
from ...impl.external_stable_diffusion import (
    decode_base64_image,
    SamplerName,
    SAMPLER_NAME_LABELS,
    STABLE_DIFFUSION_IMG2IMG_URL,
    post,
    encode_base64_image,
    nearest_valid_size,
    ResizeMode,
    RESIZE_MODE_LABELS, InpaintingFill,
)
from ...node_base import NodeBase, group
from ...node_factory import NodeFactory
from ...properties import expression
from ...properties.inputs import (
    TextInput,
    NumberInput,
    SliderInput,
    EnumInput,
    ImageInput, BoolInput,
)
from ...properties.outputs import ImageOutput
from ...utils.utils import get_h_w_c


class InpaintArea(Enum):
    WholePicture="WholePicture"
    OnlyMasked="OnlyMasked"


@NodeFactory.register("chainner:external_stable_diffusion:inpainting")
class InpaintingNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = "Inpaint an image using Automatic1111"
        self.inputs = [
            ImageInput(),
            ImageInput("Mask", channels=1, image_type=expression.Image(size_as="Input0")),
            TextInput("Prompt", default="an astronaut riding a horse"),
            TextInput("Negative Prompt").make_optional(),
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
                SamplerName,
                default_value=SamplerName.EULER,
                option_labels=SAMPLER_NAME_LABELS,
            ),
            SliderInput(
                "CFG Scale",
                minimum=1,
                default=7,
                maximum=20,
                controls_step=0.1,
                precision=1,
            ),
            EnumInput(
                ResizeMode,
                default_value=ResizeMode.JUST_RESIZE,
                option_labels=RESIZE_MODE_LABELS,
            ),
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
            SliderInput(
                "Mask Blur",
                minimum=0,
                default=4,
                maximum=64,
                unit="px",
            ),
            EnumInput(InpaintingFill, default_value=InpaintingFill.FILL),
            EnumInput(InpaintArea, default_value=InpaintArea.WholePicture),
            group(
                "conditional-enum",
                {
                    "enum": 14,
                    "conditions": [
                        InpaintArea.OnlyMasked.value,
                    ],
                },
            )(
                SliderInput(
                    "Only masked padding",
                    minimum=0,
                    default=32,
                    maximum=256,
                    slider_step=4,
                    controls_step=4,
                    unit="px",
                ),
            )
        ]
        self.outputs = [
            ImageOutput(
                image_type="""def nearest_valid(n: number) = int & floor(n / 8) * 8;
                Image {
                    width: nearest_valid(Input10),
                    height: nearest_valid(Input11)
                }""",
                channels=3,
            ),
        ]

        self.category = ExternalStableDiffusionCategory
        self.name = "Image to Image"
        self.icon = "MdChangeCircle"
        self.sub = "Automatic1111"

    def run(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        negative_prompt: Optional[str],
        denoising_strength: float,
        seed: int,
        steps: int,
        sampler_name: SamplerName,
        cfg_scale: float,
        resize_mode: ResizeMode,
        width: int,
        height: int,
        mask_blur: float,
        inpainting_fill: InpaintingFill,
        inpaint_area: InpaintArea,
        inpaint_full_res_padding: int,
    ) -> np.ndarray:
        width, height = nearest_valid_size(
            width, height
        )  # This cooperates with the "image_type" of the ImageOutput
        request_data = {
            "init_images": [encode_base64_image(image)],
            "mask": encode_base64_image(mask),
            "inpainting_fill": inpainting_fill.value,
            "mask_blur": mask_blur,
            "inpaint_full_res": inpaint_area == InpaintArea.OnlyMasked,
            "inpaint_full_res_padding": inpaint_full_res_padding,
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "denoising_strength": denoising_strength,
            "seed": seed,
            "steps": steps,
            "sampler_name": sampler_name.value,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "resize_mode": resize_mode.value,
        }
        response = post(url=STABLE_DIFFUSION_IMG2IMG_URL, json_data=request_data)
        result = decode_base64_image(response["images"][0])
        h, w, _ = get_h_w_c(result)
        assert (w, h) == (
            width,
            height,
        ), f"Expected the returned image to be {width}x{height}px but found {w}x{h}px instead "
        return result
