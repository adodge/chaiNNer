# from __future__ import annotations
#
# import numpy as np
# import torch
#
# from nodes.impl.stable_diffusion.types import LatentImage, VAEModel, image_to_array
# from nodes.node_base import NodeBase
# from nodes.node_factory import NodeFactory
# from nodes.properties.inputs import ImageInput
# from nodes.properties.inputs.stable_diffusion_inputs import LatentImageInput, VAEModelInput
# from nodes.properties.outputs import ImageOutput
# from nodes.nodes.stable_diffusion import category as StableDiffusionCategory
# from nodes.properties.outputs.stable_diffusion_outputs import LatentImageOutput
#
#
# @NodeFactory.register("chainner:stable_diffusion:latent_to_image")
# class LatentToImageNode(NodeBase):
#     def __init__(self):
#         super().__init__()
#         self.description = ""
#         self.inputs = [
#             LatentImageInput(),
#         ]
#         self.outputs = [
#             ImageOutput(),
#         ]
#
#         self.category = StableDiffusionCategory
#         self.name = "Latent to Image"
#         self.icon = "PyTorch"
#         self.sub = "Latent"
#
#     @torch.no_grad()
#     def run(self, latent_image: LatentImage) -> np.ndarray:
#         img, msk = latent_image.to_arrays()
#         return img
#
#
# @NodeFactory.register("chainner:stable_diffusion:image_to_latent")
# class ImageToLatentNode(NodeBase):
#     def __init__(self):
#         super().__init__()
#         self.description = ""
#         self.inputs = [
#             ImageInput(),
#         ]
#         self.outputs = [
#             LatentImageOutput(),
#         ]
#
#         self.category = StableDiffusionCategory
#         self.name = "Image to Latent"
#         self.icon = "PyTorch"
#         self.sub = "Latent"
#
#     @torch.no_grad()
#     def run(self, img: np.ndarray) -> LatentImage:
#         latent = LatentImage.from_arrays(img, None)
#         return latent
