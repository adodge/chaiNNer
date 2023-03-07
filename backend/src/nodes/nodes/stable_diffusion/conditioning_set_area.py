from __future__ import annotations

from ...impl.stable_diffusion.types import Conditioning
from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties.inputs import SliderInput
from ...properties.inputs.stable_diffusion_inputs import ConditioningInput
from ...properties.outputs.stable_diffusion_outputs import ConditioningOutput
from . import category as StableDiffusionCategory


# @NodeFactory.register("chainner:stable_diffusion:conditioning_set_area")
# class ConditinoingComposeNode(NodeBase):
#     def __init__(self):
#         super().__init__()
#         self.description = ""
#         self.inputs = [
#             ConditioningInput(),
#             SliderInput(
#                 "strength",
#                 minimum=0,
#                 maximum=10,
#                 default=1,
#                 slider_step=0.01,
#                 controls_step=0.01,
#             ),
#             SliderInput(
#                 "width",
#                 unit="px",
#                 minimum=64,
#                 maximum=4096,
#                 default=512,
#                 slider_step=64,
#                 controls_step=64,
#             ),
#             SliderInput(
#                 "height",
#                 unit="px",
#                 minimum=64,
#                 maximum=4096,
#                 default=512,
#                 slider_step=64,
#                 controls_step=64,
#             ),
#             SliderInput(
#                 "x",
#                 unit="px",
#                 minimum=64,
#                 maximum=4096,
#                 default=512,
#                 slider_step=64,
#                 controls_step=64,
#             ),
#             SliderInput(
#                 "y",
#                 unit="px",
#                 minimum=64,
#                 maximum=4096,
#                 default=512,
#                 slider_step=64,
#                 controls_step=64,
#             ),
#         ]
#         self.outputs = [
#             ConditioningOutput(),
#         ]
#
#         self.category = StableDiffusionCategory
#         self.name = "Set Area"
#         self.icon = "PyTorch"
#         self.sub = "Conditioning"
#
#     def run(
#         self,
#         cond: Conditioning,
#         strength: float,
#         width: int,
#         height: int,
#         x: int,
#         y: int,
#     ) -> Conditioning:
#         return cond.set_area(width, height, x, y, strength)
