# from __future__ import annotations
#
# from nodes.impl.stable_diffusion.types import LatentImage
# from nodes.node_base import NodeBase
# from nodes.node_factory import NodeFactory
# from nodes.properties.inputs import SliderInput
# from nodes.properties.inputs.stable_diffusion_inputs import LatentImageInput
# from nodes.properties.outputs.stable_diffusion_outputs import LatentImageOutput
# from nodes.nodes.stable_diffusion import category as StableDiffusionCategory
#
#
# @NodeFactory.register("chainner:stable_diffusion:latent_compose")
# class LatentComposeNode(NodeBase):
#     def __init__(self):
#         super().__init__()
#         self.description = ""
#         self.inputs = [
#             LatentImageInput("to"),
#             LatentImageInput("from"),
#             SliderInput("x", unit="px", minimum=0, maximum=4096, default=0),
#             SliderInput("y", unit="px", minimum=0, maximum=4096, default=0),
#             SliderInput("Feather", unit="px", minimum=0, maximum=4096, default=0),
#         ]
#         self.outputs = [
#             LatentImageOutput(),
#         ]
#
#         self.category = StableDiffusionCategory
#         self.name = "Compose"
#         self.icon = "PyTorch"
#         self.sub = "Latent"
#
#     def run(
#         self,
#         latent_to: LatentImage,
#         latent_from: LatentImage,
#         x: int,
#         y: int,
#         feather: int,
#     ) -> LatentImage:
#         return LatentImage.combine(latent_to, latent_from, x, y, feather)
