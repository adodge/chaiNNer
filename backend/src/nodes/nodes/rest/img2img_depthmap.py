from __future__ import annotations

from sanic.log import logger

from . import category as RESTCategory
from ...impl.rest import decode_base64_image, SamplerName, STABLE_DIFFUSION_IMG2IMG_URL, post_async, encode_base64_image
from ...node_base import AsyncNodeBase
from ...node_factory import NodeFactory
from ...properties.inputs import (
    TextInput,
    NumberInput,
    EnumInput,
    ImageInput,
)
from ...properties.outputs import LargeImageOutput

from ...utils.utils import get_h_w_c

@NodeFactory.register("chainner:rest:sd_img2img_depthmap")
class Img2ImgDepthMap(AsyncNodeBase):
    def __init__(self):
        super().__init__()
        self.description = ""
        self.inputs = [
            ImageInput(),
        ]
        self.outputs = [
            LargeImageOutput(),
            LargeImageOutput(),
        ]

        self.category = RESTCategory
        self.name = "Tmage2Image (Depth Map)"
        self.icon = "BsFillImageFill"
        self.sub = "SD Image-to-Image"

    async def run_async(self, image: np.ndarray) -> np.ndarray:
        h,w,_ = get_h_w_c(image)
        request_data = {
            'init_images': [encode_base64_image(image)],
            "denoising_strength": 0,
            "script_name": "DepthMap v0.3.8",
            "script_args": list({
                'compute_device': 0,
                'model_type': 0,
                'net_width': w,
                'net_height': h,
                'match_size': True,
                'invert_depth': False,
                'boost': True,
                'save_depth': True,
                'show_depth': True,
                'show_heat': False,
                'combine_output': False,
                'combine_output_axis': 1,
                'gen_stereo': False,
                'gen_anaglyph': False,
                'stereo_divergence': False,
                'stereo_fill': "none",
                'stereo_balance': 0,
                'clipdepth': False,
                'clipthreshold_far': 0,
                'clipthreshold_near': 1,
                'inpaint': False,
                'inpaint_vids': False,
                'background_removal_model': "u2net",
                'background_removal': False,
                'pre_depth_background_removal': False,
                'save_background_removal_masks': False,
                'gen_normal': False,
            }.values())
        }
        response = await post_async(url=STABLE_DIFFUSION_IMG2IMG_URL, json_data=request_data)
        return [decode_base64_image(img) for img in response['images']]
