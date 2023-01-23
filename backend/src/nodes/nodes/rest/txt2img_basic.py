from __future__ import annotations

from sanic.log import logger

from . import category as RESTCategory
from ...impl.rest import decode_base64_image, STABLE_DIFFUSION_TEXT2IMG_URL, post_async
from ...node_base import AsyncNodeBase
from ...node_factory import NodeFactory
from ...properties.inputs import (
    TextInput,
    NumberInput,
)
from ...properties.outputs import LargeImageOutput

@NodeFactory.register("chainner:rest:sd_txt2img_basic")
class BasicTxt2Img(AsyncNodeBase):
    def __init__(self):
        super().__init__()
        self.description = ""
        self.inputs = [
            TextInput("Prompt", default="an astronaut riding a horse"),
            NumberInput("Seed", minimum=0, default=42, maximum=4294967296),
        ]
        self.outputs = [
            LargeImageOutput(),
        ]

        self.category = RESTCategory
        self.name = "Text2Image (Basic)"
        self.icon = "BsFillImageFill"
        self.sub = "SD Text-to-Image"

    async def run_async(self, prompt: str, seed: int) -> np.ndarray:
        request_data = {
            "prompt": prompt,
            "seed": seed,
        }
        response = await post_async(url=STABLE_DIFFUSION_TEXT2IMG_URL, json_data=request_data)
        return decode_base64_image(response['images'][0])
